"""
Core SUMO/TraCI wrapper for the multi-junction traffic light control problem.

This class is intentionally NOT a Gymnasium-API-conformant single-agent env,
even though it borrows `spaces` from gymnasium for convenience. Its `step()`
takes one action per junction and advances the *whole* SUMO simulation by one
step. The adapter that makes this look like N parallel single-agent envs
(what Stable-Baselines3 needs for shared-parameter training) lives in
`sumo_vec_env.py`.

Key fixes vs. the original version:
- Fixed SUMO binary path resolution on Windows (explicit full path vs. PATH lookup)
- `gymnasium` import typo fixed (was `gymanisum`).
- `step()` now returns the modern 5-tuple (obs, reward, terminated, truncated,
  info) instead of the old 4-tuple (obs, reward, done, info).
- One TraCI snapshot is gathered per simulation step and reused across
  observation, reward, and metrics computation.
- `get_metrics()` now actually accumulates and returns values.
- Reward sign bugs fixed: queue length and global waiting time are correctly penalized.
"""

import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from pathlib import Path


class SumoEnvironment(gym.Env):
    def __init__(self, net_file, route_file, gui=False, max_episode_steps=None,
                 time_to_teleport=600, deadlock_waiting_time=3600, sumo_home=None):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.gui = gui
        self.time_to_teleport = time_to_teleport
        self.deadlock_waiting_time = deadlock_waiting_time
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.sumo_home = sumo_home or os.getenv("SUMO_HOME")

        # Validate files before starting SUMO
        self._validate_files()
        
        self._start_sumo()

        """
        Just this single line can get all junctions with traffic lights in the network, the way it works is when we run traci.start(sumo_cmd)
        it launches a sumo C++ executable in the background, now when this line is read, python sends a Traci command to the local network sockets running sumo
        sumo looks up it's data strctures and returns a list of junctions.
        """
        self.junction_ids = traci.trafficlight.getIDList()
        print("Detected Junctions:", self.junction_ids)
        if not self.junction_ids:
            raise ValueError("No junctions (traffic lights) found in the SUMO network!")

        """
        Even though we have full network file, we still want to build_neighbor_map() since it allows us build our map in our caches
        and establish those relations between junction->edge->lane->connection->lane->egde->junction, this gives us a fast O(1) lookups 
        """
        self.neighbor_map = self._build_neighbor_map()
        self.max_neighbors = max(len(n) for n in self.neighbor_map.values())

        # Per-agent observation: [local_state (4), neighbor_states (4 * max_neighbors), previous_action (1)]
        """
        As you can see we add a custom observation space for each junction, we have 4 
        """

        self.obs_dim_per_agent = 4 + (4 * self.max_neighbors) + 1
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.obs_dim_per_agent,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self.previous_phases = {jid: None for jid in self.junction_ids}
        self.previous_actions = {jid: 0 for jid in self.junction_ids}

        self._reset_metrics()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _validate_files(self):
        """Validate that net_file and route_file exist before starting SUMO."""
        net_file = Path(self.net_file).resolve()
        route_file = Path(self.route_file).resolve()

        if not net_file.is_file():
            raise FileNotFoundError(f"Network file not found:\n{net_file}")
        if not route_file.is_file():
            raise FileNotFoundError(f"Route file not found:\n{route_file}")

    def _get_sumo_binary(self):
        """
        Resolve the SUMO binary path explicitly on Windows.
        This avoids PATH lookup issues that can occur with winget installations.
        """
        if not self.sumo_home:
            raise EnvironmentError(
                "SUMO_HOME not set. Please set the environment variable SUMO_HOME "
                "to your SUMO installation directory (e.g., C:\\Program Files (x86)\\Eclipse\\Sumo)"
            )

        sumo_home = Path(self.sumo_home)
        if not sumo_home.is_dir():
            raise FileNotFoundError(f"SUMO_HOME directory not found:\n{sumo_home}")

        binary_name = "sumo-gui.exe" if self.gui else "sumo.exe"
        sumo_binary = sumo_home / "bin" / binary_name

        if not sumo_binary.is_file():
            raise FileNotFoundError(
                f"SUMO executable not found:\n{sumo_binary}\n"
                f"Check your SUMO_HOME setting: {sumo_home}"
            )

        return str(sumo_binary)

    def _build_neighbor_map(self):
        """Build adjacency map of neighboring junctions connected by edges."""
        neighbor_map = {jid: set() for jid in self.junction_ids}
        edges = traci.edge.getIDList()
        for edge in edges:
            from_junction = traci.edge.getFromJunction(edge)
            to_junction = traci.edge.getToJunction(edge)
            if from_junction in self.junction_ids and to_junction in self.junction_ids:
                neighbor_map[from_junction].add(to_junction)
                neighbor_map[to_junction].add(from_junction)
        return {jid: list(neighbors) for jid, neighbors in neighbor_map.items()}

    def _start_sumo(self):
        """Start SUMO with explicit binary path resolution."""
        sumo_binary = self._get_sumo_binary()
        
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--time-to-teleport", str(self.time_to_teleport),
        ]
        
        print(f"Starting SUMO with command:")
        print(" ".join(sumo_cmd))
        
        try:
            traci.start(sumo_cmd)
        except Exception as e:
            raise RuntimeError(
                f"Failed to start SUMO. Command was:\n{' '.join(sumo_cmd)}\n"
                f"Error: {e}"
            ) from e

    def _reset_metrics(self):
        self._metrics = {
            "total_arrived": 0,
            "step_count": 0,
            "sum_avg_waiting_time": 0.0,
            "sum_avg_queue_length": 0.0,
        }

    # ------------------------------------------------------------------ #
    # Gym-style API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        traci.close()
        self._start_sumo()
        self.step_count = 0
        self.previous_phases = {jid: None for jid in self.junction_ids}
        self.previous_actions = {jid: 0 for jid in self.junction_ids}
        self._reset_metrics()

        #This is what allows us to set up our observation state, since a traffic network is always changing
        snapshot = self._get_network_snapshot()
        global_obs, per_agent_obs = self._get_observation(snapshot)
        return global_obs, per_agent_obs

    def step(self, action):
        tls_ids = self.junction_ids
        if isinstance(action, (list, np.ndarray)):
            action = [int(a) for a in action]
        else:
            action = [int(action)]

        for tls_id, act in zip(tls_ids, action):
            traci.trafficlight.setPhase(tls_id, act)

        traci.simulationStep()
        self.step_count += 1

        snapshot = self._get_network_snapshot()
        global_obs, per_agent_obs = self._get_observation(snapshot)
        total_reward, per_agent_rewards = self._compute_reward(action, snapshot)
        self._update_metrics(snapshot)

        for i, jid in enumerate(tls_ids):
            self.previous_actions[jid] = action[i]

        no_vehicles_left = traci.simulation.getMinExpectedNumber() == 0
        max_waiting_time = max(snapshot["vehicle_waiting_time"].values(), default=0)
        deadlock_detected = max_waiting_time > self.deadlock_waiting_time

        terminated = bool(no_vehicles_left or deadlock_detected)
        truncated = bool(
            self.max_episode_steps is not None and self.step_count >= self.max_episode_steps
        )

        if deadlock_detected:
            print(f"Deadlock detected: max waiting time {max_waiting_time}s. Ending episode.")

        info = {"per_agent_rewards": per_agent_rewards}
        return (global_obs, per_agent_obs), total_reward, terminated, truncated, info

    def close(self):
        traci.close()

    # ------------------------------------------------------------------ #
    # Observation / reward / metrics
    # ------------------------------------------------------------------ #
    def _get_network_snapshot(self):
        """One pass over TraCI per simulation step, reused by all downstream calls."""
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_speeds = {v: traci.vehicle.getSpeed(v) for v in vehicle_ids}
        vehicle_lanes = {v: traci.vehicle.getLaneID(v) for v in vehicle_ids}
        vehicle_waiting_time = {v: traci.vehicle.getWaitingTime(v) for v in vehicle_ids}

        lane_ids = traci.lane.getIDList()
        lane_waiting_time = {l: traci.lane.getWaitingTime(l) for l in lane_ids}
        lane_halting = {l: traci.lane.getLastStepHaltingNumber(l) for l in lane_ids}

        return {
            "vehicle_ids": vehicle_ids,
            "vehicle_speeds": vehicle_speeds,
            "vehicle_lanes": vehicle_lanes,
            "vehicle_waiting_time": vehicle_waiting_time,
            "lane_ids": lane_ids,
            "lane_waiting_time": lane_waiting_time,
            "lane_halting": lane_halting,
            "arrived_this_step": traci.simulation.getArrivedNumber(),
        }

    def _lane_stats(self, lanes, snapshot):
        """Queue length, waiting time, moving-vehicle count, avg speed for a set of lanes."""
        queue_length = sum(snapshot["lane_halting"].get(l, 0) for l in lanes)
        waiting_time = sum(snapshot["lane_waiting_time"].get(l, 0) for l in lanes)
        lanes_set = set(lanes)
        moving, speeds = 0, []
        for v in snapshot["vehicle_ids"]:
            if snapshot["vehicle_lanes"].get(v) in lanes_set:
                speed = snapshot["vehicle_speeds"][v]
                speeds.append(speed)
                if speed > 0.1:
                    moving += 1
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        return queue_length, waiting_time, moving, avg_speed

    def _get_observation(self, snapshot):
        global_obs = []
        for junction in self.junction_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            queue_length, waiting_time, moving, avg_speed = self._lane_stats(controlled_lanes, snapshot)
            n_lanes = max(len(controlled_lanes), 1)
            global_obs.extend([queue_length / n_lanes, waiting_time / n_lanes, moving, avg_speed])
        global_state = np.array(global_obs, dtype=np.float32)

        per_agent_states = {}
        for idx, junction in enumerate(self.junction_ids):
            local_state = global_state[idx * 4:(idx + 1) * 4]

            neighbor_states = []
            for neighbor in self.neighbor_map[junction]:
                neighbor_idx = self.junction_ids.index(neighbor)
                neighbor_states.extend(global_state[neighbor_idx * 4:(neighbor_idx + 1) * 4])
            while len(neighbor_states) < self.max_neighbors * 4:
                neighbor_states.append(0.0)
            neighbor_states = np.array(neighbor_states, dtype=np.float32)

            prev_action = np.array([self.previous_actions[junction]], dtype=np.float32)
            per_agent_states[junction] = np.concatenate([local_state, neighbor_states, prev_action])

        return global_state, per_agent_states

    def _compute_reward(self, action, snapshot):
        global_waiting_time = sum(snapshot["lane_waiting_time"].values())
        global_arrivals = snapshot["arrived_this_step"]
        global_reward = (global_arrivals * 2.0) - (global_waiting_time / 3600.0)

        per_agent_rewards = []
        total_reward = 0.0
        for idx, junction in enumerate(self.junction_ids):
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            queue_length, waiting_time, moving, avg_speed = self._lane_stats(controlled_lanes, snapshot)

            norm_queue = queue_length / 50.0
            norm_waiting = waiting_time / 3600.0

            local_reward = -(norm_queue * 0.5) - (norm_waiting * 0.5)
            reward = 0.3 * local_reward + 0.7 * global_reward

            current_phase = action[idx]
            prev_phase = self.previous_phases[junction]
            if prev_phase is not None and current_phase != prev_phase:
                reward -= 0.2
            self.previous_phases[junction] = current_phase

            per_agent_rewards.append(reward)
            total_reward += reward

        return total_reward, per_agent_rewards

    def _update_metrics(self, snapshot):
        self._metrics["total_arrived"] += snapshot["arrived_this_step"]
        self._metrics["step_count"] += 1
        if snapshot["lane_ids"]:
            self._metrics["sum_avg_waiting_time"] += float(np.mean(list(snapshot["lane_waiting_time"].values())))
            self._metrics["sum_avg_queue_length"] += float(np.mean(list(snapshot["lane_halting"].values())))

    def get_metrics(self):
        """Episode-level metrics accumulated across every step() call since reset()."""
        steps = max(self._metrics["step_count"], 1)
        return {
            "avg_waiting_time": self._metrics["sum_avg_waiting_time"] / steps,
            "avg_queue_length": self._metrics["sum_avg_queue_length"] / steps,
            "throughput": self._metrics["total_arrived"],
        }


# ========================================================================= #
# Test run
# ========================================================================= #

if __name__ == "__main__":
    BASE_DIR = r"D:\Users\NK\Resume\Projects_to_Github\Multi-Agent-SUMO_DQN"
    MAPS_DIR = os.path.join(BASE_DIR, "Maps")

    NET_FILE = os.path.join(MAPS_DIR, "network2.net.xml")
    ROUTE_FILE = os.path.join(MAPS_DIR, "routes2.rou.xml")

    try:
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            gui=True,
            max_episode_steps=1000,
            sumo_home=r"C:\Program Files (x86)\Eclipse\Sumo",  # Explicit path
        )

        print(f"✓ Environment initialized successfully")
        print(f"✓ Number of junctions: {len(env.junction_ids)}")
        print(f"✓ Observation space shape: {env.observation_space.shape}")
        print(f"✓ Action space: {env.action_space}")
        print()

        global_obs, per_agent_obs = env.reset()
        print(f"✓ Reset successful")
        print(f"  Global observation shape: {global_obs.shape}")
        print(f"  Per-agent observations: {len(per_agent_obs)} junctions")
        print()

        print("Running 100 simulation steps...")
        step_count = 0
        done = False
        while not done and step_count < 100:
            action = [env.action_space.sample() for _ in env.junction_ids]
            (global_obs, per_agent_obs), reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if step_count % 20 == 0:
                print(f"  Step {step_count}: total_reward={reward:.2f}, terminated={terminated}")

            done = terminated or truncated

        metrics = env.get_metrics()
        print()
        print(f"Episode finished after {step_count} steps")
        print(f"Metrics:")
        print(f"  Throughput: {metrics['throughput']:.0f} vehicles")
        print(f"  Avg queue length: {metrics['avg_queue_length']:.2f}")
        print(f"  Avg waiting time: {metrics['avg_waiting_time']:.2f}s")

        env.close()
        print()
        print("✓ Simulation completed successfully!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
