"""
Loads the final trained shared-DQN model and runs one evaluation episode
(deterministic actions, no exploration), reporting the traffic metrics for
that episode. Run this after train.py has produced models/shared_dqn_final.zip.
"""

import os
from stable_baselines3 import DQN
from train import build_env, MODEL_DIR


def evaluate(model_path=None, gui=True, max_episode_steps=950):
    model_path = model_path or os.path.join(MODEL_DIR, "shared_dqn_final")
    model = DQN.load(model_path)

    vec_env = build_env(gui=gui, max_episode_steps=max_episode_steps)
    obs = vec_env.reset()

    done = False
    episode_metrics = None
    while not done:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(actions)
        # All junctions share one SUMO simulation, so every entry in `dones`
        # is identical -- checking dones[0] is enough.
        done = bool(dones[0])
        if done:
            episode_metrics = infos[0]["episode_metrics"]

    print("\nEvaluation metrics for this episode:")
    for key, value in episode_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    vec_env.close()
    return episode_metrics


if __name__ == "__main__":
    evaluate(gui=True)
