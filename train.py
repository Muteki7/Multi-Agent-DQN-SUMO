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
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

# --- Paths: update these for your machine/environment. ---
# The original hardcoded an absolute Windows path; this defaults to a
# `maps/` folder next to this script instead, which works regardless of OS
# or whether you're running this natively or under WSL2.

#We are linking all the paths, sumo.exe to run the gui, maps to find the network and the route files, we also have model_dir and log_dir both which we create in our code
#The reason we are mentioning this now itself is since once it's created it needs to know where to be stored, os.join stores it in the main file
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
    #As we build our enviornment we create log_dir if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    sumo_env = SumoEnvironment(
        SUMO_NET_PATH, SUMO_ROUTE_PATH, gui=gui, max_episode_steps=max_episode_steps, sumo_home=r"C:\Program Files (x86)\Eclipse\Sumo"
    )

    #We wrap our sumo enviornmet in a vectorized enviorment
    vec_env = SumoVecEnv(sumo_env)
    vec_env = VecMonitor(vec_env, filename=os.path.join(LOG_DIR, "monitor"))
    return vec_env


def train(desired_sim_steps=9_500, max_episode_steps=950):
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
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,              # learn after every real sim step
        target_update_interval=1_000,
        exploration_fraction=0.3,  # epsilon anneal over the first 30% of training
        exploration_final_eps=0.02,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

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
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (summed per-agent reward per episode)")
    plt.title("Shared DQN Training Reward")
    plt.grid(True)
    out_path = os.path.join(LOG_DIR, "training_curve.png")
    plt.savefig(out_path)
    print(f"Training curve saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    train(desired_sim_steps=9_500, max_episode_steps=950)
