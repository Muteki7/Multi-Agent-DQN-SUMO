"""
Adapter that makes `SumoEnvironment` look like N parallel Gymnasium envs to
Stable-Baselines3, where N = number of traffic-light junctions.

Why this exists: SB3 has no built-in multi-agent support, but it IS built
around training one policy against a VecEnv of many parallel, structurally
identical environments. Since every junction in this project has the same
observation shape and the same Discrete(4) action space, we can present them
to SB3 *as if* they were independent parallel envs. SB3 then trains one
shared network on all of their experience -- this is parameter sharing
across homogeneous cooperative agents, without needing PettingZoo or RLlib.

The key thing that makes this different from a normal VecEnv (e.g.
DummyVecEnv running 4 separate env instances): there is only ONE underlying
SUMO simulation. Every step_wait() call advances that single simulation once,
with one action per junction, then reshapes the per-junction results into the
(num_envs, ...) arrays SB3 expects. Because of this, `done` is always
identical across all "envs" -- the whole SUMO episode starts and ends
together.
"""

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sumo_env import SumoEnvironment


class SumoVecEnv(VecEnv):
    def __init__(self, sumo_env: SumoEnvironment):
        self.sumo_env = sumo_env
        self._actions = None
        super().__init__(
            #Number of envs = number of junctions in sumo network
            num_envs=len(sumo_env.junction_ids),
            observation_space=sumo_env.observation_space,
            action_space=sumo_env.action_space,
        )

    # ------------------------------------------------------------------ #
    # Core VecEnv API
    # ------------------------------------------------------------------ #
    def reset(self):
        _, per_agent_obs = self.sumo_env.reset()
        return self._stack_obs(per_agent_obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_async() must be called before step_wait().")

        (_, per_agent_obs), _total_reward, terminated, truncated, info = self.sumo_env.step(self._actions)
        obs = self._stack_obs(per_agent_obs)
        rewards = np.array(info["per_agent_rewards"], dtype=np.float32)

        done = bool(terminated or truncated)
        dones = np.full(self.num_envs, done, dtype=bool)
        # SB3's replay buffer uses "TimeLimit.truncated" to avoid bootstrapping
        # incorrectly through artificial step-limit cutoffs (see
        # handle_timeout_termination in ReplayBuffer). Since our episode ends
        # for all agents simultaneously, this is the same for every agent.
        infos = [{"TimeLimit.truncated": bool(truncated and not terminated)} for _ in range(self.num_envs)]

        if done:
            # get_metrics() must be read BEFORE reset(), since reset() zeroes
            # the accumulator for the next episode. Stashing it in info means
            # both training code and evaluate.py can retrieve the metrics for
            # the episode that just ended, at the exact step where it ended.
            episode_metrics = self.sumo_env.get_metrics()
            # Match the autoreset convention used by SB3's own DummyVecEnv:
            # stash the real terminal observation, then return the *new*
            # episode's reset observation so training can continue seamlessly.
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["episode_metrics"] = episode_metrics
            _, per_agent_obs = self.sumo_env.reset()
            obs = self._stack_obs(per_agent_obs)

        self._actions = None
        return obs, rewards, dones, infos

    def close(self):
        self.sumo_env.close()

    # ------------------------------------------------------------------ #
    # Bookkeeping methods required by the VecEnv ABC. There's only one real
    # underlying env (`self.sumo_env`), so these just broadcast to/from it.
    # ------------------------------------------------------------------ #
    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        value = getattr(self.sumo_env, attr_name)
        return [value for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.sumo_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = self._get_indices(indices)
        method = getattr(self.sumo_env, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = self._get_indices(indices)
        return [False for _ in indices]

    # ------------------------------------------------------------------ #
    def _stack_obs(self, per_agent_obs):
        return np.stack(
            [per_agent_obs[jid] for jid in self.sumo_env.junction_ids]
        ).astype(np.float32)
