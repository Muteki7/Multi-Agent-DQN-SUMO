"""
Loads a trained shared-DQN model and runs one evaluation episode
(deterministic actions), reporting the traffic metrics for that episode.
Run after train.py has produced models/shared_dqn_final.zip.
"""

import os
from stable_baselines3 import DQN
from sumo_rl_env import build_env, DEFAULT_NET_FILE, DEFAULT_ROUTE_FILE

BASE_DIR = r"D:\Users\NK\Resume\Projects_to_Github\Multi-Agent-SUMO_DQN"
MODEL_DIR = os.path.join(BASE_DIR, "models")

NET_FILE = DEFAULT_NET_FILE
ROUTE_FILE = DEFAULT_ROUTE_FILE


def evaluate(model_path=None, gui=True, num_seconds=3600):
    model_path = model_path or os.path.join(MODEL_DIR, "shared_dqn_final")
    model = DQN.load(model_path)

    vec_env = build_env(net_file=NET_FILE, route_file=ROUTE_FILE, gui=gui, num_seconds=num_seconds)
    obs = vec_env.reset()

    done = False
    while not done:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(actions)
        # All agents share one SUMO simulation, so every entry in `dones` is
        # identical -- checking dones[0] is enough.
        done = bool(dones[0])

    # Thanks to _FixResetClobberedInfo in sumo_rl_env.py, infos[0] at this
    # final step correctly holds the just-ended episode's true totals, not
    # a post-reset zeroed snapshot.
    final_info = infos[0]
    print("\nEvaluation metrics for this episode:")
    for key in ["system_total_arrived", "system_total_departed", "system_total_teleported"]:
        print(f"  {key}: {final_info.get(key)}")

    vec_env.close()
    return final_info


if __name__ == "__main__":
    evaluate(gui=True)
