"""
Trains ONE shared DQN policy across all traffic-light junctions in the SUMO
network (parameter sharing), using SumoVecEnv to present each junction as a
parallel env to Stable-Baselines3.

IMPORTANT semantics: SB3's `total_timesteps` counts VecEnv steps, and each
VecEnv step here advances the ONE underlying SUMO simulation once, for
`num_envs` (= number of junctions) agents at once. So `total_timesteps` in
SB3's own bookkeeping is `real_sumo_steps * num_junctions`, not
`real_sumo_steps`. We compute total_timesteps from a "how many real
simulation steps do I want" number so this stays intuitive -- see
`desired_sim_steps` below.
"""

import os
from sumo_env import SumoEnvironment
from sumo_vec_env import SumoVecEnv
from traffic_metrics_callback import TrafficMetricsCallback
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

# --- Paths: update these for your machine/environment. ---
# The original hardcoded an absolute Windows path; this defaults to a
# `maps/` folder next to this script instead, which works regardless of OS
# or whether you're running this natively or under WSL2.
BASE_DIR = r"D:\Users\NK\Resume\Projects_to_Github\Multi-Agent-SUMO_DQN"
MAPS_DIR = os.path.join(BASE_DIR, "Maps")
SUMO_NET_PATH = os.path.join(MAPS_DIR, "network2.net.xml")
SUMO_ROUTE_PATH = os.path.join(MAPS_DIR, "routes2.rou.xml")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")


def build_env(gui=False, max_episode_steps=950):
    """
    Builds SumoEnvironment -> SumoVecEnv -> VecMonitor.
    VecMonitor logs per-episode reward/length to a CSV in LOG_DIR so we can
    plot the training curve afterward without hand-rolling reward tracking.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    sumo_env = SumoEnvironment(
        SUMO_NET_PATH, SUMO_ROUTE_PATH, gui=gui, max_episode_steps=max_episode_steps
    )
    vec_env = SumoVecEnv(sumo_env)
    vec_env = VecMonitor(
        vec_env,
        filename=os.path.join(LOG_DIR, "monitor"),
        info_keywords=("metric_avg_waiting_time", "metric_avg_queue_length", "metric_throughput"),
    )
    return vec_env


def train(desired_sim_steps=9_500, max_episode_steps=2000):
    os.makedirs(MODEL_DIR, exist_ok=True)
    vec_env = build_env(gui=False, max_episode_steps=max_episode_steps)

    total_timesteps = desired_sim_steps * vec_env.num_envs
    print(
        f"{vec_env.num_envs} junctions detected -> training for "
        f"{total_timesteps} SB3 timesteps ({desired_sim_steps} real SUMO "
        f"simulation steps, since every sim step produces one transition "
        f"per junction under parameter sharing)."
    )

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=4.6354919948565836e-4,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=32,
        gamma=0.9255333029299476,
        train_freq=8,              # learn after every real sim step
        target_update_interval=500,
        exploration_fraction=0.49477201339838284,  # epsilon anneal over the first 30% of training
        exploration_final_eps=0.08957867548503272,
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(x, y)
    axes[0, 0].set_title("Episode reward (proxy objective)")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].grid(True)

    metric_panels = [
        ("metric_avg_waiting_time", axes[0, 1], "Avg waiting time (lower is better)"),
        ("metric_avg_queue_length", axes[1, 0], "Avg queue length (lower is better)"),
        ("metric_throughput", axes[1, 1], "Throughput (higher is better)"),
    ]
    for col, ax, title in metric_panels:
        if col in df.columns:
            ax.plot(x, df[col])
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.grid(True)
        else:
            ax.set_visible(False)

    fig.suptitle("Proxy reward vs. actual task metrics -- these should trend together")
    fig.tight_layout()
    out_path = os.path.join(LOG_DIR, "training_curve.png")
    plt.savefig(out_path)
    print(f"Training curve saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    train(desired_sim_steps=9_500, max_episode_steps=2000)
