import random

import gymnasium as gym
import numpy as np
import tqdm
from gymnasium.wrappers import TimeLimit

from xqc.envs.wrapper import RepeatAction
from xqc.envs.subproc_worker import SubprocEnvPool
from xqc.utils import is_slurm_job

# One OS process per env replica whenever more than one is requested (see
# xqc/envs/subproc_worker.py): the MuJoCo physics, the IK and the rendering
# then run concurrently across CPU cores.
_SUBPROC_BENCHMARKS = {"pick_place"}

_BENCHMARK_REGISTRY = [
    ("pick_place", "xqc.envs.pick_place", "PICK_PLACE_ALL"),
]


def resolve_env_benchmark(env_name: str) -> str:
    import importlib

    for benchmark, module_path, list_attr in _BENCHMARK_REGISTRY:
        if env_name in getattr(importlib.import_module(module_path), list_attr):
            return benchmark

    raise ValueError(f"Unknown env: {env_name}")


def make_env(
    env_name: str, seed: int = 0, action_repeat=1, render: bool = False, camera_names: list = None,
    horizon: int = None, dataset_path: str = None, abs_action: bool = False, render_resolution: int = None,
) -> gym.Env:
    benchmark = resolve_env_benchmark(env_name)

    if benchmark == "pick_place":
        from xqc.envs.pick_place import make_pick_place_env

        # abs_action must be True: the env only takes absolute end-effector
        # targets (see xqc/envs/pick_place.py).
        assert abs_action, "PickPlaceBottle takes absolute end-effector actions: set agent abs_action=true"
        env = make_pick_place_env(
            env_name, dataset_path=dataset_path, render=render, camera_names=camera_names, horizon=horizon,
        )

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    return env


class ParallelEnv:
    """
    Parallel env adapted from https://github.com/naumix/BiggerRegularizedCategorical/blob/main/jaxrl/envs.py
    """

    def __init__(
        self, env_names: list, seed: int = 0, action_repeat=1, render: bool = False, camera_names: list = None,
        horizon: int = None, dataset_path: str = None, abs_action: bool = False, render_resolution: int = None,
    ):
        np.random.seed(seed)
        random.seed(seed)

        benchmark = resolve_env_benchmark(env_names[0])
        use_subproc = benchmark in _SUBPROC_BENCHMARKS

        if use_subproc:
            self.envs = None
            self._pool = SubprocEnvPool(
                env_names, seed, action_repeat, render, camera_names, horizon, dataset_path, abs_action,
            )
            obs_dims = np.array([shape[0] for shape in self._pool.obs_shapes], dtype=np.int32)
            act_dims = np.array([shape[0] for shape in self._pool.act_shapes], dtype=np.int32)
            dtype = self._pool.obs_dtypes[0]
            action_dtype = self._pool.act_dtypes[0]
            self._horizon = self._pool.horizon
        else:
            self._pool = None
            self._horizon = None
            envs = []
            obs_dims = np.zeros(len(env_names), dtype=np.int32)
            act_dims = np.zeros(len(env_names), dtype=np.int32)
            for i, env_name in enumerate(env_names):
                envs.append(
                    make_env(
                        env_name, seed=seed, action_repeat=action_repeat, render=render, camera_names=camera_names,
                        horizon=horizon, dataset_path=dataset_path, abs_action=abs_action,
                        render_resolution=render_resolution,
                    )
                )
                obs_dims[i] = envs[-1].observation_space.shape[0]
                act_dims[i] = envs[-1].action_space.shape[0]
            self.envs = envs
            dtype = envs[-1].observation_space.dtype
            action_dtype = envs[-1].action_space.dtype

        max_state_dim = int(np.max(obs_dims))
        max_action_dim = int(np.max(act_dims))
        state_dim_differences = max_state_dim - obs_dims

        observation_space = gym.spaces.Box(
            low=(np.ones(max_state_dim, dtype=dtype)[None, :] - np.inf).repeat(
                len(env_names), axis=0
            ),
            high=(np.ones(max_state_dim, dtype=dtype)[None, :] + np.inf).repeat(
                len(env_names), axis=0
            ),
            shape=(len(env_names), max_state_dim),
            dtype=dtype,
        )

        action_space = gym.spaces.Box(
            low=(np.ones(max_action_dim, dtype=action_dtype)[None, :] * -1).repeat(
                len(env_names), axis=0
            ),
            high=(np.ones(max_action_dim, dtype=action_dtype)[None, :]).repeat(
                len(env_names), axis=0
            ),
            shape=(len(env_names), max_action_dim),
            dtype=action_dtype,
        )

        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.state_dim_differences = state_dim_differences
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_tasks = len(env_names)

    @property
    def max_episode_steps(self):
        if self._pool is not None:
            return self._horizon
        if self.envs is None:
            return None

        env = self.envs[0]
        while True:
            if isinstance(env, TimeLimit):
                return env._max_episode_steps
            # Robomimic / Robosuite / Mimicgen
            if hasattr(env, "horizon"):
                return env.horizon
            # If it has an underlying env, keep unwrapping; otherwise, we're done
            if hasattr(env, "env"):
                env = env.env
            else:
                break
        return None

    def _reset_idx(self, idx: int):
        seed = np.random.randint(0, 1e8)
        if self._pool is not None:
            state = self._pool.reset_idx(idx, seed)
        else:
            state, _ = self.envs[idx].reset(seed=seed)
        state = np.concatenate(
            (state, np.zeros(self.state_dim_differences[idx], dtype=np.float32)), axis=0
        )
        return state

    def generate_masks(self, terminals: np.ndarray, truncates: np.ndarray):
        masks = 1 - (terminals * (1 - truncates))
        return masks

    def reset_where_done(
        self, states: np.ndarray, terminals: np.ndarray, truncates: np.ndarray
    ):
        resets = np.zeros((terminals.shape))
        for j, (terminal, truncate) in enumerate(zip(terminals, truncates)):
            if terminal or truncate:
                states[j], terminals[j], truncates[j] = self._reset_idx(j), False, False
                resets[j] = 1
        return states, terminals, truncates, resets

    def reset(self):
        states = []
        for i in range(self.num_tasks):
            states.append(self._reset_idx(i))
        return np.stack(states)

    def _get_goal(self, info: dict):
        if "success" in info:
            goal = info["success"]
        elif "is_success" in info:
            goal = info["is_success"]
        elif "solved" in info:
            goal = info["solved"]
        else:
            goal = 0
        return goal

    def step(self, actions: np.ndarray):
        if self._pool is not None:
            raw_states, rewards, terminals, truncates, goals = self._pool.step(
                [action[: self.act_dims[i]] for i, action in enumerate(actions)]
            )
            states = [
                np.concatenate((state, np.zeros(self.state_dim_differences[i], dtype=np.float32)), axis=0)
                for i, state in enumerate(raw_states)
            ]
            return np.stack(states), rewards, terminals, truncates, goals

        states, rewards, terminals, truncates, goals = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminal, truncate, info = env.step(
                action[: self.act_dims[i]]
            )
            state = np.concatenate(
                (state, np.zeros(self.state_dim_differences[i], dtype=np.float32)),
                axis=0,
            )
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            truncates.append(truncate)
            goals.append(self._get_goal(info))
        return (
            np.stack(states),
            np.stack(rewards),
            np.stack(terminals),
            np.stack(truncates),
            np.stack(goals),
        )

    def evaluate(
        self,
        agent,
        num_episodes,
        temperature=0.0,
        render=False,
        max_render_steps=1000,
        render_frameskip=4,
        render_num_envs=1,
        render_episodes=1,
        verbose=False,
    ):
        n_rollouts = np.zeros(self.num_tasks)
        returns = np.zeros(self.num_tasks)
        goals = np.zeros(self.num_tasks)
        mask = np.ones(self.num_tasks)
        mask_goals = np.ones(self.num_tasks)
        timesteps = np.zeros(self.num_tasks)
        observations = self.reset()
        # Opt-in only: agents without this method (e.g. XQCLearner) are unaffected.
        if hasattr(agent, "reset_policy"):
            agent.reset_policy()
        if render:
            render_num_envs = min(render_num_envs, self.num_tasks)
            max_frames = max_render_steps // max(render_frameskip, 1)
            frame_shape = self.render(num_envs=render_num_envs).shape
            renders = np.empty((max_frames, *frame_shape), dtype=np.uint8)
            render_idx = 0
        i = 0
        # Opt-in only (hasattr): agents without a round_trip_error method
        # (everything except MFInvertibleLearner) are unaffected. Gated to
        # i % execution_horizon == 0 - the same cadence sample_actions
        # itself calls the network at (it reuses a buffered action chunk in
        # between - xqc/agents/mf_invertible/mf_learner.py's sample_actions)
        # - so this adds roughly one extra network call per real chunk
        # inference, not per env step. Reads self._obs_history/
        # self._image_history as sample_actions itself just left them (see
        # MFInvertibleLearner.round_trip_error's docstring on why this is
        # safe to interleave rather than disturbing the rollout).
        round_trip_errors = []
        # Progress = the slowest replica's completed-episode count (n_rollouts.min()),
        # since that's what the loop below actually waits on to break - faster
        # replicas finishing more episodes in the meantime wouldn't reflect how
        # much longer the call is going to take.
        pbar = tqdm.tqdm(total=num_episodes, desc="eval", leave=False, disable=is_slurm_job())
        prev_done = 0
        while True:
            if render:
                render_active = n_rollouts[:render_num_envs].min() < render_episodes
                if (
                    render_active
                    and i % render_frameskip == 0
                    and render_idx < max_frames
                ):
                    renders[render_idx] = self.render(num_envs=render_num_envs)
                    render_idx += 1
            # Opt-in only: agents without a camera_keys attribute (e.g. XQCLearner) are unaffected.
            if getattr(agent, "camera_keys", None):
                actions = agent.sample_actions(
                    observations, temperature=temperature, images=self.get_images(agent.camera_keys)
                )
            else:
                actions = agent.sample_actions(observations, temperature=temperature)
            if hasattr(agent, "round_trip_error") and i % getattr(agent, "execution_horizon", 1) == 0:
                round_trip_errors.append(np.asarray(agent.round_trip_error()))
            next_observations, rewards, terms, truncs, success = self.step(actions)
            # Opt-in only: agents without this method (e.g. XQCLearner) are unaffected.
            if hasattr(agent, "reset_policy_idx"):
                agent.reset_policy_idx(np.logical_or(terms, truncs))
            returns += rewards * mask
            # Gated by `mask` too (not just `mask_goals`, which only guards against
            # double-counting success within one episode): once a replica has
            # completed its quota of `num_episodes`, it must stop contributing to
            # `goals` the same way `returns`/`timesteps` already do. Without this,
            # a replica that keeps succeeding while OTHER parallel replicas
            # (num_seeds>1) are still catching up to `num_episodes` keeps adding to
            # `goals` unbounded, since the loop only breaks once the slowest
            # replica reaches num_episodes - producing avg_goal = goals/num_episodes
            # values above 1.0 (a "success rate" > 100%, which isn't meaningful).
            goals += success * mask_goals * mask
            mask_goals = np.where(success, 0, mask_goals)
            # Captured here (after the success-based update, before the
            # terminal-episode reset below) so it reflects "did this episode
            # ever succeed" for the episode that's about to end, not just
            # whether `success` happens to be True on this exact final step.
            episode_succeeded = mask_goals == 0
            still_counted = mask.copy()
            mask_goals = np.where(np.logical_or(terms, truncs), 1, mask_goals)
            timesteps += mask
            observations = next_observations
            just_finished = np.logical_or(terms, truncs)
            n_rollouts += just_finished
            observations, terms, truncs, _ = self.reset_where_done(
                observations, terms, truncs
            )
            mask = np.where(n_rollouts >= num_episodes, 0, 1)
            if verbose:
                # Only for replicas whose episode actually counted toward
                # num_episodes (still_counted) - a fast-finishing replica in
                # a num_seeds>1 eval keeps looping past its own quota while
                # slower replicas catch up, and those extra episodes aren't
                # part of the reported average, so printing them would be
                # misleading noise.
                for task_idx in np.nonzero(just_finished & (still_counted > 0))[0]:
                    ep_num = int(n_rollouts[task_idx])
                    outcome = "SUCCESS" if episode_succeeded[task_idx] else "FAIL"
                    print(
                        f"[eval] task {task_idx}: episode {ep_num}/{num_episodes} "
                        f"{outcome} | cumulative goals={int(goals[task_idx])}/{ep_num} "
                        f"({goals[task_idx] / ep_num:.3f})",
                        flush=True,
                    )
            i += 1
            done = int(n_rollouts.min())
            if done > prev_done:
                pbar.update(done - prev_done)
                prev_done = done
            if done == num_episodes:
                break
        pbar.close()

        avg_goal = goals / num_episodes
        avg_return = returns / num_episodes
        avg_time = timesteps / num_episodes
        avg_completion_speed = (
            (self.max_episode_steps - avg_time) / self.max_episode_steps * 100.0
        )

        stats = {
            "goal": avg_goal,
            "return": avg_return,
            "completion_speed": avg_completion_speed,
        }
        # Per-replica mean across however many chunk-inference steps fired
        # the hook above - shape (num_tasks,), same as goal/return/
        # completion_speed, so log_infos_to_wandb's per-seed-array
        # expectation still holds.
        if round_trip_errors:
            stats["round_trip_error"] = np.mean(round_trip_errors, axis=0)
        if render:
            renders = renders[:render_idx]
            renders = np.transpose(renders, (1, 0, 4, 2, 3))
            stats["renders"] = renders
        return stats

    def render(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_tasks
        if self._pool is not None:
            return self._pool.render(num_envs)
        renders = []
        for i in range(num_envs):
            render = self.envs[i].render()
            renders.append(render)
        renders = np.stack(renders)
        return renders

    def get_images(self, camera_keys) -> dict:
        """Batched raw camera frames, one per key: {key: (num_tasks, H, W, 3) uint8}.
        Requires the envs were created with render=True (use_camera_obs=True)."""
        if self._pool is not None:
            return self._pool.get_images(camera_keys)
        per_env = [env.get_images(camera_keys) for env in self.envs]
        return {key: np.stack([obs[key] for obs in per_env]) for key in camera_keys}

    def close(self):
        """Tears down subprocess workers, if any (no-op for the in-process
        backend, which has nothing to clean up beyond normal GC)."""
        if self._pool is not None:
            self._pool.close()

    def __str__(self):
        if self._pool is not None:
            return f"ParallelEnv<subprocess pool, {self.num_tasks} workers, horizon={self._horizon}>"
        s = "ParallelEnv<\n"
        for env in self.envs:
            s += f"  {env}"
            e = env
            while hasattr(e, "env"):
                if isinstance(e, TimeLimit):
                    s += f"  max_t={e._max_episode_steps}"
                    break
                e = e.env
            s += "\n"
        return s + ">"
