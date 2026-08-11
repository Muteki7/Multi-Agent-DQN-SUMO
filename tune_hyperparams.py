"""
Hyperparameter search for the shared DQN traffic-signal policy, using Optuna
with a MedianPruner -- rewritten to use sumo_rl_env.build_env() instead of
the custom SumoEnvironment/SumoVecEnv pair. Everything else about the search
design carries over unchanged from the custom-env version.

Single objective (avg_waiting_time), not three: MedianPruner needs one
scalar per trial. avg_queue_length-equivalent (avg_stopped) and throughput
are recorded as trial user_attrs so you can still compare them across
trials without inventing an arbitrary weighted combination of differently-
scaled quantities.

Every trial closes its SUMO subprocess in a `finally` block -- not
optional. Each trial spawns a fresh `sumo` subprocess via TraCI; without
this, a pruned or failed trial leaks a SUMO process, and after 20-30 trials
you'd have that many zombie processes running.
"""

import optuna
from optuna.pruners import MedianPruner

from sumo_rl_env import build_env, DEFAULT_NET_FILE, DEFAULT_ROUTE_FILE
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from traffic_metrics_callback import TrafficMetricsCallback
from optuna_pruning_callback import OptunaPruningCallback

# Shorter than a full training run on purpose -- search needs many trials,
# and each trial is a full SUMO run. Once you've found good hyperparameters,
# run train.py with the full num_seconds/desired_rounds using the winner.
SEARCH_NUM_SECONDS = 900
SEARCH_ROUNDS = 3_000

NET_FILE = DEFAULT_NET_FILE
ROUTE_FILE = DEFAULT_ROUTE_FILE


def objective(trial: optuna.Trial) -> float:
    hyperparams = dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        buffer_size=trial.suggest_categorical("buffer_size", [10_000, 20_000, 50_000]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        gamma=trial.suggest_float("gamma", 0.90, 0.999),
        train_freq=trial.suggest_categorical("train_freq", [1, 4, 8]),
        target_update_interval=trial.suggest_categorical("target_update_interval", [500, 1_000, 5_000]),
        exploration_fraction=trial.suggest_float("exploration_fraction", 0.1, 0.5),
        exploration_final_eps=trial.suggest_float("exploration_final_eps", 0.01, 0.1),
    )

    vec_env = build_env(net_file=NET_FILE, route_file=ROUTE_FILE, gui=False, num_seconds=SEARCH_NUM_SECONDS)
    try:
        total_timesteps = SEARCH_ROUNDS * vec_env.num_envs

        model = DQN("MlpPolicy", vec_env, tensorboard_log=None, verbose=0, **hyperparams)

        metrics_cb = TrafficMetricsCallback(window=10)
        pruning_cb = OptunaPruningCallback(
            trial, metrics_cb, metric_key="avg_waiting_time", eval_every_episodes=3
        )
        model.learn(total_timesteps=total_timesteps, callback=CallbackList([metrics_cb, pruning_cb]))

        final_wait = metrics_cb.get_recent_mean("avg_waiting_time")
        trial.set_user_attr("avg_stopped", metrics_cb.get_recent_mean("avg_stopped"))
        trial.set_user_attr("throughput", metrics_cb.get_recent_mean("throughput"))
        trial.set_user_attr("teleports", metrics_cb.get_recent_mean("teleports"))

        if final_wait is None:
            raise optuna.TrialPruned("No episodes completed within the trial budget.")

        return final_wait
    finally:
        vec_env.close()


def run_study(n_trials=30, study_name="dqn_traffic_tuning", storage=None):
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def inspect_results(study: optuna.Study):
    print(f"\nCompleted trials: {len(study.trials)}")
    print(f"Best avg_waiting_time: {study.best_value:.3f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Other metrics for the best trial (not optimized directly, for reference):")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)
    print("\nFull trial history written to optuna_results.csv")

    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html("optuna_history.html")
        vis.plot_param_importances(study).write_html("optuna_param_importances.html")
        print("Wrote optuna_history.html and optuna_param_importances.html")
    except ImportError:
        print("Install `plotly` to also get optuna_history.html / optuna_param_importances.html")


if __name__ == "__main__":
    study = run_study(n_trials=30, storage="sqlite:///optuna_Petting_zoo_traffic.db")
    inspect_results(study)
