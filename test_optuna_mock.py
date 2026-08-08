"""
Validates the exact objective()/pruning pattern used in tune_hyperparams.py
against MockSumoEnv, so we can confirm: bad trials actually get pruned,
vec_env.close() runs even when TrialPruned is raised mid-training, and
user_attrs/best_value come back sensible -- without waiting on real SUMO.
"""
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList

from test_vecenv_mock import MockSumoEnv
from sumo_vec_env import SumoVecEnv
from traffic_metrics_callback import TrafficMetricsCallback
from optuna_pruning_callback import OptunaPruningCallback

closed_envs = []  # track whether close() actually gets called on every trial


class TrackedMockSumoEnv(MockSumoEnv):
    def close(self):
        closed_envs.append(id(self))


def objective(trial):
    # Deliberately bias one hyperparameter to produce obviously worse
    # trials, so we can confirm MedianPruner actually prunes some of them.
    bad_run = trial.suggest_categorical("bad_run", [0, 1])
    episode_len = trial.suggest_int("episode_len", 5, 10)

    mock = TrackedMockSumoEnv(episode_len=episode_len)
    vec_env = SumoVecEnv(mock)
    vec_env = VecMonitor(vec_env)
    try:
        model = DQN("MlpPolicy", vec_env, learning_starts=5, buffer_size=200,
                    batch_size=8, train_freq=1, verbose=0)

        metrics_cb = TrafficMetricsCallback(window=5)
        pruning_cb = OptunaPruningCallback(trial, metrics_cb, metric_key="avg_waiting_time",
                                            eval_every_episodes=2)

        # Simulate "bad" trials having worse (higher) waiting time by
        # monkeypatching get_metrics on this instance.
        if bad_run:
            original_get_metrics = mock.get_metrics
            mock.get_metrics = lambda: {**original_get_metrics(), "avg_waiting_time": 999.0}

        model.learn(total_timesteps=200, callback=CallbackList([metrics_cb, pruning_cb]))

        final_wait = metrics_cb.get_recent_mean("avg_waiting_time")
        trial.set_user_attr("avg_queue_length", metrics_cb.get_recent_mean("avg_queue_length"))
        trial.set_user_attr("throughput", metrics_cb.get_recent_mean("throughput"))
        if final_wait is None:
            raise optuna.TrialPruned("no episodes completed")
        return final_wait
    finally:
        vec_env.close()


def main():
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=15)

    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{len(completed)} completed, {len(pruned)} pruned, out of {len(study.trials)} total trials")
    assert len(pruned) > 0, "expected at least one 'bad_run' trial to be pruned by MedianPruner"

    print(f"closed_envs count: {len(closed_envs)} (should equal total trials: {len(study.trials)})")
    assert len(closed_envs) == len(study.trials), "vec_env.close() was not called for every trial!"

    print("\nBest trial:", study.best_trial.params, "value:", study.best_value)
    print("user_attrs:", study.best_trial.user_attrs)
    print("\nALL OPTUNA INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
