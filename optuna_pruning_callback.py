"""
Reports a rolling-window average of avg_waiting_time to Optuna periodically
and raises optuna.TrialPruned() when MedianPruner decides the trial isn't
worth continuing.

Design note: the "correct" way to do Optuna pruning is usually to run a
separate held-out evaluation episode periodically (e.g. via EvalCallback) so
the pruning signal isn't just noisy training-time behavior. We're not doing
that here because a SUMO episode is expensive -- running a full extra
evaluation episode every N steps would roughly double total tuning time for
every trial. Instead we reuse TrafficMetricsCallback's rolling window over
recent *training* episodes as the pruning signal. This is a reasonable
practical compromise, not a rigorous held-out evaluation -- worth knowing
the difference if your search results ever look inconsistent with a later
full training run.
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
        if infos and "episode_metrics" in infos[0]:
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
