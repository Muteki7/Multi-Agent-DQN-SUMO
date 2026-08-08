"""
Hyperparameter search for the shared DQN traffic-signal policy, using Optuna
with a MedianPruner to cut off unpromising trials early.

Why avg_waiting_time as the single objective: Optuna's pruners (MedianPruner
included) work against a single scalar per trial. You care about three
things (waiting time down, queue down, throughput up); rather than inventing
an arbitrary weighted combination of three differently-scaled quantities,
this optimizes avg_waiting_time directly (the metric you emphasized most)
and records avg_queue_length / throughput as trial user_attrs so you can
still inspect and compare them across trials afterward -- see
`inspect_results()` at the bottom. If you later want true multi-objective
search (a Pareto front across all three), look at Optuna's native
multi-objective studies (`optuna.create_study(directions=[...])`) -- note
pruning support there is more limited than in single-objective studies,
which is part of why this script sticks to single-objective + pruning.

MedianPruner intuition: at each reported step, a trial is pruned if its
value so far is worse than the MEDIAN of all other trials' values at that
same step -- but only after `n_startup_trials` trials have completed
(so there's a real median to compare against) and after `n_warmup_steps`
reports within the current trial (so a trial isn't killed before its metric
has had a chance to stabilize).

Practical note: every trial spawns a fresh `sumo` subprocess via TraCI. The
`finally: vec_env.close()` block below is not optional -- without it, a
pruned or failed trial leaks a SUMO process, and after 20-30 trials you'll
have that many zombie SUMO processes running.
"""

import optuna
from optuna.pruners import MedianPruner

from train import build_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from traffic_metrics_callback import TrafficMetricsCallback
from optuna_pruning_callback import OptunaPruningCallback

# Shorter than a full training run on purpose -- hyperparameter search needs
# many trials, and each trial is a full SUMO run, so this trades per-trial
# training quality for being able to run enough trials at all. Once you've
# narrowed in on good hyperparameters, run train.py with the full
# desired_sim_steps using the winning config.
SEARCH_SIM_STEPS = 3_000
SEARCH_MAX_EPISODE_STEPS = 500


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

    vec_env = build_env(gui=False, max_episode_steps=SEARCH_MAX_EPISODE_STEPS)
    try:
        total_timesteps = SEARCH_SIM_STEPS * vec_env.num_envs

        model = DQN("MlpPolicy", vec_env, tensorboard_log=None, verbose=0, **hyperparams)

        metrics_cb = TrafficMetricsCallback(window=10)
        pruning_cb = OptunaPruningCallback(
            trial, metrics_cb, metric_key="avg_waiting_time", eval_every_episodes=3
        )
        model.learn(total_timesteps=total_timesteps, callback=CallbackList([metrics_cb, pruning_cb]))

        final_wait = metrics_cb.get_recent_mean("avg_waiting_time")
        trial.set_user_attr("avg_queue_length", metrics_cb.get_recent_mean("avg_queue_length"))
        trial.set_user_attr("throughput", metrics_cb.get_recent_mean("throughput"))

        if final_wait is None:
            # No episode completed at all in this trial's budget -- treat as
            # a degenerate trial rather than reporting a misleading None/0.
            raise optuna.TrialPruned("No episodes completed within the trial budget.")

        return final_wait
    finally:
        vec_env.close()


def run_study(n_trials=30, study_name="dqn_traffic_tuning", storage="sqlite:///optuna_traffic.db"):
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # minimizing avg_waiting_time
        pruner=pruner,
        storage=storage,       # e.g. "sqlite:///optuna_traffic.db" to persist/resume
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
    study = run_study(n_trials=30, storage="sqlite:///optuna_traffic.db")
    inspect_results(study)
