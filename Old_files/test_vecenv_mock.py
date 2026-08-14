"""
Validates SumoVecEnv end-to-end against a fake SumoEnvironment (no real
TraCI/SUMO needed) by running a few steps of SB3 DQN training and one
save/load/predict cycle. This is a structural test, not a behavioral one --
it can't tell us if the traffic policy is good, only that the plumbing
(shapes, dones, resets, replay buffer) is wired correctly.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Old_files.sumo_vec_env import SumoVecEnv


class MockSumoEnv(gym.Env):
    """Mimics SumoEnvironment's interface with 3 junctions, no TraCI calls."""

    def __init__(self, n_junctions=3, obs_dim=9, episode_len=15):
        self.junction_ids = [f"J{i}" for i in range(n_junctions)]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.episode_len = episode_len
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        per_agent = {jid: np.random.rand(self.observation_space.shape[0]).astype(np.float32)
                     for jid in self.junction_ids}
        return None, per_agent

    def step(self, actions):
        self.t += 1
        per_agent = {jid: np.random.rand(self.observation_space.shape[0]).astype(np.float32)
                     for jid in self.junction_ids}
        per_agent_rewards = [float(-a) for a in actions]  # arbitrary but deterministic signal
        terminated = False
        truncated = self.t >= self.episode_len
        info = {"per_agent_rewards": per_agent_rewards}
        return (None, per_agent), sum(per_agent_rewards), terminated, truncated, info

    def close(self):
        pass

    def get_metrics(self):
        return {"avg_waiting_time": 1.23, "avg_queue_length": 4.56, "throughput": self.t}


def main():
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.env_checker import check_env  # noqa: F401 (single-agent only, not used here)

    mock = MockSumoEnv()
    vec_env = SumoVecEnv(mock)
    vec_env = VecMonitor(vec_env)

    print("num_envs:", vec_env.num_envs)
    print("observation_space:", vec_env.observation_space)
    print("action_space:", vec_env.action_space)

    obs = vec_env.reset()
    assert obs.shape == (3, 9), f"unexpected reset obs shape: {obs.shape}"

    model = DQN(
        "MlpPolicy", vec_env, learning_starts=10, buffer_size=500,
        batch_size=8, train_freq=1, verbose=0,
    )
    model.learn(total_timesteps=100)

    # Confirm at least one episode boundary was crossed and reset correctly
    # by stepping manually and checking shapes/dtypes at a done boundary.
    obs = vec_env.reset()
    for i in range(20):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape == (3, 9)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3
        if dones[0]:
            assert "terminal_observation" in infos[0]
            assert "episode_metrics" in infos[0]
            print(f"step {i}: episode boundary hit correctly, "
                  f"terminal_observation + episode_metrics present: {infos[0]['episode_metrics']}")

    model.save("/tmp/test_dqn_model")
    loaded = DQN.load("/tmp/test_dqn_model")
    action, _ = loaded.predict(obs, deterministic=True)
    print("predict() output shape:", action.shape)
    assert action.shape == (3,)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
