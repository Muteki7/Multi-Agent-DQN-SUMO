"""
Reports a rolling-window average of avg_waiting_time to Optuna periodically
and raises optuna.TrialPruned() when MedianPruner decides the trial isn't
worth continuing. Unchanged from the custom-env version -- this callback
only depends on TrafficMetricsCallback's rolling window, not on which
environment produced it.

Design note: the "correct" way to do Optuna pruning is usually a separate
held-out evaluation run, not training-time rollout metrics. We reuse
TrafficMetricsCallback's rolling window instead because a SUMO episode is
expensive enough that running a second one per report would roughly double
search time. Worth knowing if search results ever look inconsistent with a
full training run later.
"""

import optuna
from stable_baselines3.common.callbacks import BaseCallback


class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial, metrics_callback, metric_key="avg_waiting_time",
                 eval_every_episodes=3, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.metrics_callback = metrics_callback
        self.metric_key = metric_key
        self.eval_every_episodes = eval_every_episodes
        self._episode_count = 0
        self._last_reported_count = -1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            self._episode_count += 1
            due = self._episode_count % self.eval_every_episodes == 0
            if due and self._episode_count != self._last_reported_count:
                self._last_reported_count = self._episode_count
                mean_value = self.metrics_callback.get_recent_mean(self.metric_key)
                if mean_value is not None:
                    self.trial.report(mean_value, self._episode_count)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
        return True
