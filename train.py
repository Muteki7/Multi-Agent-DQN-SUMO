"""
Trains ONE shared DQN policy across all traffic-light junctions using
sumo-rl's PettingZoo env + SuperSuit for parameter sharing -- replaces the
custom SumoEnvironment/SumoVecEnv pair entirely.

Defaults to the 4x4 grid bundled with sumo-rl (see sumo_rl_env.py) so this
runs immediately with no dependency on your own net/route files. Once this
is working well, switch NET_FILE/ROUTE_FILE to your own
Multi-Agent-SUMO_DQN network -- everything else (training loop, metrics,
Optuna tuning, evaluation) stays the same, since sumo-rl reads any
standard SUMO net.xml/rou.xml pair.

IMPORTANT semantics carried over from the custom env: SB3's total_timesteps
counts one increment per agent per decision round (num_envs = number of
traffic-light junctions), not real simulation seconds. total_timesteps here
is computed from how many real decision ROUNDS you want, so this stays
intuitive.
"""

import os
from sumo_rl_env import build_env, DEFAULT_NET_FILE, DEFAULT_ROUTE_FILE
from traffic_metrics_callback import TrafficMetricsCallback
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

BASE_DIR = r"D:\Users\NK\Resume\Projects_to_Github\Multi-Agent-SUMO_DQN"
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Swap these to your own Multi-Agent-SUMO_DQN network once the pipeline is
# validated on the bundled 4x4 grid.
NET_FILE = DEFAULT_NET_FILE
ROUTE_FILE = DEFAULT_ROUTE_FILE


def train(desired_rounds=9_500, num_seconds=3600):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    vec_env = build_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        gui=False,
        num_seconds=num_seconds,
        monitor_dir=os.path.join(LOG_DIR, "monitor"),
    )

    total_timesteps = desired_rounds * vec_env.num_envs
    print(
        f"{vec_env.num_envs} traffic-light agents detected -> training for "
        f"{total_timesteps} SB3 timesteps ({desired_rounds} real decision "
        f"rounds, since every round produces one transition per agent under "
        f"parameter sharing)."
    )

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    metrics_callback = TrafficMetricsCallback(window=10)
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)

    final_model_path = os.path.join(MODEL_DIR, "shared_dqn_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}.zip")
    print(f"Load it later with: DQN.load('{final_model_path}')")

    vec_env.close()
    plot_training_curve()
    return final_model_path


def plot_training_curve():
    df = load_results(LOG_DIR)
    if df.empty:
        print("No completed episodes were logged -- nothing to plot.")
        return

    x, y = ts2xy(df, "timesteps")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x, y)
    axes[0].set_title("Episode reward (diff-waiting-time)")
    axes[0].set_xlabel("Timesteps")
    axes[0].grid(True)

    if "system_total_arrived" in df.columns:
        axes[1].plot(x, df["system_total_arrived"])
        axes[1].set_title("Throughput (vehicles arrived per episode)")
        axes[1].set_xlabel("Timesteps")
        axes[1].grid(True)
    else:
        axes[1].set_visible(False)

    fig.suptitle("Reward vs. throughput -- for avg_waiting_time/queue, see the TensorBoard traffic/ tab")
    fig.tight_layout()
    out_path = os.path.join(LOG_DIR, "training_curve.png")
    plt.savefig(out_path)
    print(f"Training curve saved to {out_path}")


if __name__ == "__main__":
    train(desired_rounds=9_500, num_seconds=3600)
