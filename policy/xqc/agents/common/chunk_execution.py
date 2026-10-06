"""Receding-horizon action-chunk execution bookkeeping, factored out of
xqc/agents/chunking/chunked_learner.py (see conversation history): this is
pure imitation-learning/actor plumbing (observation/image history buffers,
a per-env chunk-execution cursor) with no equivalent on the critic side, so
keeping it inline in ChunkedPolicyLearner's own class body would only get
more cramped once that class also hosts a critic - exactly the "actor vs
critic living in one class" concern this split addresses.

Deliberately has zero jax/flax/Model dependency - it never runs the network
itself, only decides WHEN a fresh chunk is needed and HOW to slice/format
one once produced. The actual network inference is supplied as a plain
callback (`infer_chunk_fn`) by the caller (ChunkedPolicyLearner.sample_actions),
mirroring how xqc_learner.py's own sample_actions is a thin delegator to
xqc/networks/policies.py's stateless sample_actions_with_log_probs - the
difference here is that this bookkeeping is genuinely stateful (buffers
accumulate across calls, a per-env cursor persists), so it's a small class
owning that state rather than a bare function.
"""
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from xqc.agents.common.rotation_transformer import rotation_6d_to_axis_angle


class ChunkExecutor:
    """Owns the receding-horizon action-chunk execution state for one
    ChunkedPolicyLearner instance. `infer_chunk_fn(obs_history, image_history)
    -> action_chunk` (passed fresh to sample_actions each call, since it
    closes over the learner's own live params/rng) is the only place any
    actual network math happens - everything here is pure bookkeeping."""

    def __init__(
        self,
        To: int,
        Ta: int,
        execution_horizon: int,
        low_dim_obs_dim: int,
        camera_keys: Sequence[str],
        raw_image_shape: Dict[str, Tuple[int, int]],
        robot_action_dim: int,
        act_dim: int,
        abs_action: bool,
        num_robots: int,
    ):
        self.To = To
        self.Ta = Ta
        self.execution_horizon = execution_horizon
        self.low_dim_obs_dim = low_dim_obs_dim
        self.camera_keys = tuple(camera_keys)
        self.raw_image_shape = raw_image_shape
        self.robot_action_dim = robot_action_dim
        self.act_dim = act_dim
        self.abs_action = abs_action
        self.num_robots = num_robots

        self._needs_init: Optional[np.ndarray] = None
        self._obs_history: Optional[np.ndarray] = None
        self._image_history: Optional[Dict[str, np.ndarray]] = None
        self._action_buffer: Optional[np.ndarray] = None
        # Per-env chunk-execution cursor (array, NOT a shared scalar) - see
        # reset_idx's own comment for why a shared scalar was a real bug
        # for eval_episodes>1.
        self._step_cnt: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Full reset of chunk-execution state. Call once at the start of a
        rollout (ParallelEnv.evaluate() calls this right after env.reset())."""
        self._needs_init = None
        self._obs_history = None
        self._image_history = None
        self._action_buffer = None
        self._step_cnt = None

    def reset_idx(self, done: np.ndarray) -> None:
        """Per-env reset for envs whose episode just ended mid-rollout.

        Also resets _step_cnt[done] to 0, forcing those envs to get a freshly
        inferred chunk (from their real post-reset observation) on the very
        next sample_actions call, instead of continuing to read whatever was
        left in _action_buffer[done] (zeroed just below - i.e. an all-zero
        action, not a no-op for abs_action agents) for however many steps
        remained in the OTHER, still-in-progress envs' shared chunk cycle.
        Previously _step_cnt was a single scalar shared across every env, so
        an env finishing mid-chunk (the common case - episode length has no
        reason to align with execution_horizon) would execute stale/zero
        actions until the whole batch's cycle happened to wrap back to 0.
        Invisible for eval_episodes=1 (a replica's mask zeroes out the
        instant its one episode ends, so there's no next episode for those
        wasted steps to hurt) but a real, systematic per-episode penalty for
        eval_episodes>1 (up to execution_horizon-1 wasted steps at the start
        of every 2nd-5th episode) - see conversation history for how this was
        found (a consistent ~10-point eval-score gap between differently-
        parallelized but otherwise-identical eval configs)."""
        if self._needs_init is None:
            return
        done = np.asarray(done, dtype=bool)
        self._needs_init[done] = True
        if self._action_buffer is not None:
            self._action_buffer[done] = 0.0
        if self._step_cnt is not None:
            self._step_cnt[done] = 0

    def sample_actions(
        self,
        observations: np.ndarray,
        images: Optional[Dict[str, np.ndarray]],
        infer_chunk_fn: Callable[[np.ndarray, Dict[str, np.ndarray]], np.ndarray],
    ) -> np.ndarray:
        """Denoising/flow-map solve, executed as a receding-horizon action
        chunk. `infer_chunk_fn` is called with (obs_history, image_history)
        - both (num_envs, To, ...) - and must return a (num_envs, Ta,
        act_dim) chunk; it's the caller's job to run the actual network."""
        observations = np.asarray(observations)
        num_envs = observations.shape[0]

        if self._needs_init is None:
            self._needs_init = np.ones(num_envs, dtype=bool)
            self._obs_history = np.zeros((num_envs, self.To, self.low_dim_obs_dim), dtype=np.float32)
            self._image_history = {
                key: np.zeros((num_envs, self.To) + self.raw_image_shape[key] + (3,), dtype=np.float32)
                for key in self.camera_keys
            }
            self._action_buffer = np.zeros((num_envs, self.execution_horizon, self.robot_action_dim), dtype=np.float32)
            self._step_cnt = np.zeros(num_envs, dtype=np.int64)

        images_norm = {key: np.asarray(arr, dtype=np.float32) / 255.0 * 2.0 - 1.0 for key, arr in (images or {}).items()}

        for i in range(num_envs):
            if self._needs_init[i]:
                self._obs_history[i] = np.repeat(observations[i][None, :], self.To, axis=0)
                for key in self.camera_keys:
                    self._image_history[key][i] = np.repeat(images_norm[key][i][None, :], self.To, axis=0)
            else:
                self._obs_history[i] = np.concatenate(
                    [self._obs_history[i, 1:], observations[i][None, :]], axis=0
                )
                for key in self.camera_keys:
                    self._image_history[key][i] = np.concatenate(
                        [self._image_history[key][i, 1:], images_norm[key][i][None, :]], axis=0
                    )
        self._needs_init[:] = False

        # "Due" envs are the ones that need a freshly inferred chunk this
        # call - either because they naturally wrapped back to step_cnt==0,
        # or because reset_idx just forced it to 0 (episode just ended).
        # Inference is one batched call across every env regardless
        # (network's own batch dimension, cheap to compute for envs whose
        # result gets discarded), but only `due` envs actually ADOPT the
        # fresh chunk into _action_buffer - other envs keep whatever they
        # already had, preserving their own in-progress execution_horizon
        # cadence unchanged. This is what makes a just-reset env always get
        # a real chunk from its own current (post-reset) observation on its
        # very next step, instead of ever reading the zeroed dummy value
        # reset_idx left in its buffer slot (the bug this replaced - see
        # reset_idx's own comment for the full explanation).
        due = self._step_cnt == 0
        if np.any(due):
            action_chunk = np.asarray(infer_chunk_fn(self._obs_history, self._image_history))
            # The predicted Ta-length chunk starts AT "now" (t) - see
            # conversation history: the reference's own (To-1)-steps-before-
            # "now" alignment was dropped when the IL/RL datasets were
            # unified (xqc/replay_buffer/chunked_qc_buffer.py's own
            # actions[t:t+h] convention, adopted for IL too) - so the
            # executed slice is just the first execution_horizon steps.
            executed_chunk = action_chunk[:, :self.execution_horizon]
            if self.abs_action:
                # Network output is [pos(3), rotation_6d(6), gripper(1)] -
                # convert back to the [pos(3), axis-angle rot(3), gripper(1)]
                # the env's own absolute-pose controller actually accepts
                # (matches the reference's own undo_transform_action, see
                # eval_old_stack.py) - inverse of the forward conversion
                # applied once at dataset load time (xqc/agents/common/
                # dataset.py's own abs_action handling). PER ROBOT (10-dim
                # [pos3,rotation_6d6,gripper1] block each, self.num_robots of
                # them - a no-op loop of 1 robot's worth for every existing
                # single-arm task).
                stacked = executed_chunk.reshape(*executed_chunk.shape[:-1], self.num_robots, 10)
                pos = stacked[..., :3]
                rot6d = stacked[..., 3:9]
                gripper = stacked[..., 9:]
                rot = np.asarray(rotation_6d_to_axis_angle(rot6d))
                executed_chunk = np.concatenate([pos, rot, gripper], axis=-1).reshape(
                    *executed_chunk.shape[:-1], self.num_robots * 7
                )
            self._action_buffer[due] = np.asarray(executed_chunk)[due]

        action = self._action_buffer[np.arange(num_envs), self._step_cnt]
        self._step_cnt = (self._step_cnt + 1) % self.execution_horizon
        return action
