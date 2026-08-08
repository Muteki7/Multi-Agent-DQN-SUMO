"""
Logs traffic metrics to TensorBoard using sumo-rl's built-in info dict --
no custom TraCI snapshot code needed anymore, sumo-rl computes these already.

Two different kinds of values in that info dict, handled differently:
- system_total_arrived / departed / teleported: CUMULATIVE counters, reset
  on env.reset(). Valid to read directly at episode end -- as long as
  sumo_rl_env.py's _FixResetClobberedInfo wrapper is in the env stack (it
  is, by default in build_env()). Without it, SuperSuit's MarkovVectorEnv
  silently overwrites these with post-reset zeros on the exact boundary
  step; see that wrapper's docstring for the full story.
  
- system_mean_waiting_time / system_total_stopped / system_mean_speed:
  INSTANTANEOUS snapshots, recomputed fresh every step. Reading these only
  at episode end would give you just that one instant, not how the episode
  behaved overall -- this callback accumulates a running mean across every
  non-boundary step instead, same idea as the custom env's get_metrics().
"""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class TrafficMetricsCallback(BaseCallback):
    def __init__(self, window=10, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.recent = {
            "avg_waiting_time": deque(maxlen=window),
            "avg_stopped": deque(maxlen=window),
            "throughput": deque(maxlen=window),
            "teleports": deque(maxlen=window),
        }
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._wait_sum = 0.0
        self._stopped_sum = 0.0
        self._n = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        # Every agent's info dict carries an identical copy of the system_*
        # keys (sumo-rl copies them to every agent each step -- verified
        # against the library source), so infos[0] represents the whole
        # simulation at this step; no need to loop over all agents.
        info0 = infos[0]
        episode_info = info0.get("episode")  # set by VecMonitor at episode boundaries

        if episode_info:
            # Note: we do NOT fold this boundary step's own system_mean_*
            # values into the running average -- they reflect the freshly
            # auto-reset environment, not the episode that just ended.
            avg_wait = self._wait_sum / max(self._n, 1)
            avg_stopped = self._stopped_sum / max(self._n, 1)
            throughput = info0.get("system_total_arrived", 0)
            teleports = info0.get("system_total_teleported", 0)

            self.logger.record("traffic/avg_waiting_time", avg_wait)
            self.logger.record("traffic/avg_stopped", avg_stopped)
            self.logger.record("traffic/throughput", throughput)
            self.logger.record("traffic/teleports", teleports)
            self.logger.record("traffic/reward_per_step", episode_info["r"] / max(episode_info["l"], 1))

            self.recent["avg_waiting_time"].append(avg_wait)
            self.recent["avg_stopped"].append(avg_stopped)
            self.recent["throughput"].append(throughput)
            self.recent["teleports"].append(teleports)

            self._reset_accumulators()
        else:
            self._wait_sum += info0.get("system_mean_waiting_time", 0.0)
            self._stopped_sum += info0.get("system_total_stopped", 0.0)
            self._n += 1
        return True

    def get_recent_mean(self, key):
        values = self.recent.get(key)
        if not values:
            return None
        return sum(values) / len(values)
