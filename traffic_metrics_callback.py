"""
Custom SB3 callback that logs avg_waiting_time / avg_queue_length /
throughput to TensorBoard every time a SUMO episode ends -- these appear
alongside the built-in rollout/train charts under a `traffic/` tab, so you
can compare "did the proxy reward go up" against "did the actual traffic
outcome improve" side by side.

This is the direct answer to "can we use VecMonitor for this": VecMonitor
CAN log these (see the `metric_*` keys added in SumoVecEnv + the
`info_keywords` argument in train.py) but it writes to a CSV file, not
TensorBoard, and only supports flat scalar columns. Use VecMonitor's CSV for
a clean, dedicated post-hoc plot of just these 3 metrics; use this callback
for live, side-by-side comparison against ep_rew_mean/loss/etc. during
training. Both read from the same underlying `episode_metrics` data -- pick
whichever viewing surface suits the question you're asking.
"""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class TrafficMetricsCallback(BaseCallback):
    """
    Also exposes get_recent_mean(key), a rolling window average that the
    Optuna pruning callback reuses instead of running a separate, expensive
    evaluation episode.
    """

    def __init__(self, window=10, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.recent = {
            "avg_waiting_time": deque(maxlen=window),
            "avg_queue_length": deque(maxlen=window),
            "throughput": deque(maxlen=window),
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and "episode_metrics" in infos[0]:
            metrics = infos[0]["episode_metrics"]
            for key, value in metrics.items():
                self.logger.record(f"traffic/{key}", value)
                if key in self.recent:
                    self.recent[key].append(value)

            # Operationalizes the ep_len/ep_rew intuition from our
            # conversation: total episode reward conflates "how good was
            # each decision" with "how many decisions did we get to make."
            # Dividing by episode length disentangles them. VecMonitor
            # attaches this "episode" key to the same info dict.
            episode_info = infos[0].get("episode")
            if episode_info and episode_info.get("l", 0) > 0:
                self.logger.record(
                    "traffic/reward_per_step", episode_info["r"] / episode_info["l"]
                )
        return True

    def get_recent_mean(self, key):
        values = self.recent.get(key)
        if not values:
            return None
        return sum(values) / len(values)
