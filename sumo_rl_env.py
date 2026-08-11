"""
Builds a shared-parameter-DQN-ready environment from sumo-rl, replacing the
custom SumoEnvironment/SumoVecEnv pair entirely.

Pipeline: sumo_rl.parallel_env (PettingZoo Parallel API, one SUMO simulation,
N traffic-light agents) -> supersuit.pettingzoo_env_to_vec_env_v1 (presents
each agent as one "sub-env," exactly what SumoVecEnv did by hand before,
now library code) -> supersuit.concat_vec_envs_v1(base_class="stable_baselines3")
(SB3-compatible VecEnv) -> VecMonitor (episode logging).

Defaults to the 4x4 grid bundled with sumo-rl itself ("4x4-Lucas" network) so
this runs with zero dependency on your own net/route files -- validate your
DQN setup here first, then point net_file/route_file at your own
Multi-Agent-SUMO_DQN network.

Why time_to_teleport=-1 (disables teleportation) and reward_fn="diff-waiting-time"
matter for the exact problem you hit: teleportation was how your old net's
waiting-time metric could be reduced by deadlocking (SUMO forcibly removes
stuck vehicles, silently erasing their accumulated waiting time from the
running stats). With teleport disabled, a stuck vehicle's waiting time keeps
counting against you for as long as it's stuck -- there's no way to make
gridlock look good on this metric anymore. "diff-waiting-time" (sumo-rl's
default reward) measures the DECREASE in total accumulated waiting time
since the last decision, not a raw penalty sum -- since sumo-rl episodes run
for a fixed `num_seconds` of simulated time with no "success" terminal state
to reach, there's nothing analogous to your +100/-100/-10 terminal bonuses
to replicate here; that whole reward-hacking failure mode doesn't have
anywhere to attach itself.
"""

import os
import sumo_rl
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper

_SUMO_RL_NETS_DIR = os.path.join(os.path.dirname(sumo_rl.__file__), "nets")
DEFAULT_NET_FILE = os.path.join(_SUMO_RL_NETS_DIR, "4x4-Lucas", "4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(_SUMO_RL_NETS_DIR, "4x4-Lucas", "4x4c1.rou.xml")

# These are cumulative-since-episode-start counters in sumo-rl's info dict,
# so reading them at the episode-end step (what VecMonitor's info_keywords
# does) gives a genuine whole-episode total. The instantaneous gauges
# (system_mean_waiting_time, system_total_stopped, system_mean_speed) are
# NOT included here on purpose -- see traffic_metrics_callback.py, which
# averages those properly across the whole episode instead of reading a
# single final-step snapshot.
MONITOR_INFO_KEYWORDS = ("system_total_arrived", "system_total_departed", "system_total_teleported")


class _FixResetClobberedInfo(VecEnvWrapper):
    """
    Works around a real bug in supersuit.vector.markov_vector_wrapper.MarkovVectorEnv.step():
    on episode end, it does `{**terminal_info, **reset_info}` to merge in
    `terminal_observation`, but this also silently overwrites any OTHER
    custom info keys the environment sets -- including sumo-rl's cumulative
    system_total_arrived/departed/teleported counters, which get stomped
    with the *freshly-reset* value (0) instead of the true episode-final
    value. Verified directly: one step before `done`, system_total_arrived
    was 429; on the `done` step itself, it reads back as 0.

    This wrapper caches each sub-env's last pre-reset info and patches the
    known-clobbered keys back in in on the boundary step, so both
    VecMonitor's CSV (info_keywords) and TrafficMetricsCallback see the real
    numbers without needing to know about this quirk themselves.
    """

    def __init__(self, venv):
        #since it is a vectorized env, we need to use self.venv to access the underlying env
        super().__init__(venv)
        self._last_infos = [{} for _ in range(venv.num_envs)]

    def reset(self):
        obs = self.venv.reset()
        self._last_infos = [{} for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        #fixed infos saves the custom infos keys (MONITOR_INFO_KEYWORDS) that are reset the moment the step ends this is a workaround so we can save the custom info keys
        fixed_infos = []
        for i, info in enumerate(infos):
            if dones[i]:
                patched = dict(info)
                for key in MONITOR_INFO_KEYWORDS:
                    if key in self._last_infos[i]:
                        patched[key] = self._last_infos[i][key]
                fixed_infos.append(patched)
                self._last_infos[i] = {}  # next step starts the new episode fresh
            else:
                fixed_infos.append(info)
                self._last_infos[i] = info
        return obs, rewards, dones, fixed_infos


def build_env(
    net_file=DEFAULT_NET_FILE,
    route_file=DEFAULT_ROUTE_FILE,
    gui=False,
    num_seconds=3600,
    #Amount of time agent needs to wait before switching phase
    delta_time=5,
    yellow_time=2,
    min_green=5,
    max_green=50,
    time_to_teleport=-1,
    #Measures decrese in accumlated waiting time
    reward_fn="diff-waiting-time",
    #Wrapping into one vectorized enviornment for SB3, with VecMonitor logging to CSV
    num_vec_envs=1,
    monitor_dir=None,
    sumo_seed="random",
):
    parallel_env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        max_green=max_green,
        time_to_teleport=time_to_teleport,
        reward_fn=reward_fn,
        sumo_seed=sumo_seed,
        sumo_warnings=False,
    )

    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, num_vec_envs=num_vec_envs, num_cpus=0, base_class="stable_baselines3"
    )
    vec_env = _FixResetClobberedInfo(vec_env)
    vec_env = VecMonitor(vec_env, filename=monitor_dir, info_keywords=MONITOR_INFO_KEYWORDS)
    return vec_env
