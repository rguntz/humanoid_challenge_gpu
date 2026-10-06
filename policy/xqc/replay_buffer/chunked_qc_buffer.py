"""Chunked replay buffer(s) - offline (Dataset) and online (ReplayBuffer) -
ported from qc/utils/datasets.py (Q-chunking, Li et al., arXiv:2507.07969,
https://colinqiyangli.github.io/qc/), adapted for this repo's To-step
backward observation history and image (camera_keys) support - neither of
which qc's own repo does anywhere (their online policy only ever conditions
on the CURRENT frame, To=1 always; their robomimic loader is state-only, no
images at all - see conversation history).

Replaces xqc/replay_buffer/chunked_parallel_replay_buffer.py (online) and
xqc/replay_buffer/chunked_offline_replay_buffer.py (offline/critic-SARSA
diagnostic), unifying both under ONE shared Dataset/ReplayBuffer(Dataset)
pair - EXACTLY the two classes qc's own repo has (confirmed directly:
main_online.py imports `Dataset, ReplayBuffer` and uses `Dataset` for the
offline train/val data, `ReplayBuffer(Dataset)` for the live online buffer -
no third class on top of either). No config-holding wrapper class sits on
top of Dataset here either, for the same reason - callers (
train_parallel_chunking.py, train_critic_sarsa.py) hold To/h/discount/
camera_keys themselves and pass them explicitly at each call, the same way
qc/main.py itself calls sample_sequence with explicit args rather than
through a config object.

======================================================================
WHAT IS PORTED FAITHFULLY (same name, same logic as qc/utils/datasets.py,
kept present and callable EVEN WHERE UNUSED by our own pipeline, so every
function here can be checked directly against the original file - explicit
requirement, see conversation history):

    get_size, random_crop, batched_random_crop, Dataset.create,
    Dataset.__init__ (incl. frame_stack/p_aug/return_next_actions/
    terminal_locs/initial_locs), Dataset.get_random_idxs, Dataset.get_subset,
    Dataset.augment, ReplayBuffer.create, ReplayBuffer.create_from_initial_dataset,
    ReplayBuffer.__init__, ReplayBuffer.add_transition, ReplayBuffer.clear.

None of the above are called anywhere by our own training scripts, but each
one IS genuinely exercised somewhere in qc's own repo (get_subset via their
own Dataset.sample, etc.) - so they're kept, unused, purely so this file is
a checkable, function-by-function replica of qc/utils/datasets.py.

Module-level add_history is the one exception - deliberately NOT ported at
all (not even as an unused faithful copy), because it has zero call sites
anywhere in qc's own repo either (confirmed via grep) - unlike everything
else above, there's no real usage in the original to be a replica OF. See
the comment where it would have been, in this file, for the full note.

WHAT IS MODIFIED (qc's own function/method, same name/slot, changed in
place rather than kept as a separate unused copy alongside new same-purpose
functions - see conversation history: you're allowed to change these two
specifically):

    Dataset.sample_sequence - same TECHNIQUE as qc's own version (walk from
    a position, check self['terminals'] LIVE at each step, no cached list -
    see conversation history on why add_history's cached terminal_locs/
    initial_locs go stale against a live buffer and this live re-read
    doesn't), kept as ONE self-contained method with everything inlined
    (no separate windowing helper methods - qc doesn't have any either),
    but:
      (a) extended to also walk BACKWARD (To-step observation history -
          doesn't exist anywhere in qc's own repo at all, online or
          offline, since their own policy is To=1),
      (b) takes idxs explicitly (not drawn internally) and returns a
          ChunkedBatch instead of qc's own dict shape,
      (c) the FORWARD direction (actions/rewards/masks/next_anchor/
          next_actions) is boundary-aware and near-pads (repeats the last
          REAL action, truncates the reward sum to `valid_len = min(h, L -
          t)`) at a genuine episode boundary - see xqc/replay_buffer/
          chunked_qc_buffer_qc_reference.py for what qc's OWN literal
          version does instead (blind-clamp past the boundary, then flag
          via `valid=0` and discard the whole sample). See conversation
          history for the full story: an earlier draft of this file kept
          qc's own boundary-unaware convention faithfully (matching a
          direct request to port sample_sequence as close to their literal
          code as possible), a real wandb comparison against this repo's
          OLD (pre-qc-port) chunked_offline_replay_buffer.py then showed
          measurable divergence (train/weight_norm_ratio, train/q_values,
          val/target_q_max, val/td_error_max), traced to qc's convention
          silently discarding real, informative near-terminal training
          data (proportionally large and reward-concentrated on this
          repo's own robomimic tasks - ~18% of every episode for h=8 on
          the lift task, carrying nearly all of the real reward signal),
          and an isolated real training run confirmed that switching just
          this one piece back to this repo's own OLD near-pad/truncate
          convention recovers the OLD baseline's curves. `valid` is kept
          in ChunkedBatch's schema (always 1 now - the reward sum can no
          longer be contaminated by construction) purely so the 3
          critic-loss call sites don't need their own signature to change;
          it's the same field qc's own agents/acfql.py-style code
          multiplies the loss by, now permanently a no-op weight of 1 here.
          Correctness verified via an exhaustive independent cross-check
          against this repo's OLD literal formula, not just hand-picked
          cases - see scripts/test_sample_sequence_boundary_fix.py.
    Note: this does NOT close the one small risk qc's own forward code
    already accepts (a sampled window landing exactly on the ring-buffer's
    current write-pointer seam, where neither neighboring slot is itself
    marked terminal) - see conversation history for the full analysis. That
    risk is inherent to using ONE SHARED flat array at all (qc's own
    structure, adopted here over this repo's earlier per-episode-list
    design), not something either output convention (theirs or ours) can
    detect.

    Dataset.sample - same slot/name as qc's own (their online loop's
    replay_buffer never actually uses this one, only sample_sequence - but
    the offline train_dataset.sample(()) call in their own main.py/
    main_online.py does exercise it) - now our own epoch-shuffled offline
    entry point (train_critic_sarsa.py), calling the modified
    sample_sequence above instead of building a raw dict via get_subset/
    frame_stack/p_aug.

NOTE - deliberately NOT a method here: drawing `num_batches` (qc's own
utd_ratio) worth of random indices from ReplayBuffer.size and reshaping
sample_sequence's flat result into (num_batches, batch_size, ...) for the
online loop. qc's own main_online.py does exactly this same two-step
pattern (sample_sequence with one big flat batch_size, then
`.reshape((utd_ratio, batch_size) + ...)`) directly inline in the training
script itself, not as a buffer method - see main_online.py lines ~241-248.
train_parallel_chunking.py's own online loop does the identical thing,
matching that structure (see conversation history - an earlier version of
this file had a ReplayBuffer.sample_chunked_multibatch wrapping this;
removed once the qc precedent was found).

WHAT IS NEW / OURS (our own pipeline calls these; not a faithful port and
not a modification of an existing qc function/method, since qc has nothing
playing this role at all):

    Dataset.get_val_batch - offline held-out entry point (train_critic_sarsa.py).
    _convert_abs_action - unchanged from chunked_parallel_replay_buffer.py.
    load_robomimic_flat - our own hdf5-to-flat-array loader, modeled on
    qc's own envs/robomimic_utils.py:get_dataset (a DIFFERENT qc file,
    envs/robomimic_utils.py not utils/datasets.py - not held to the
    faithful-port naming convention either, since there's no
    utils/datasets.py analog to check it against) - same "loop over
    episodes, concatenate into one flat array" pattern, but built on THIS
    repo's own per-episode reading logic (ROBOSUITE_OBS_KEYS,
    L=argmax(dones)+1, abs_action conversion via axis_angle_to_rotation_6d)
    since qc's own loader has none of that, and extended to also
    concatenate each camera_keys image stream (qc's loader never touches
    images at all). Returns plain Dataset objects (train, val) - see this
    module's own opening paragraph on why there's no wrapper class here.
    compute_chunked_stats - standalone function (not a method - same
    reasoning), reuses xqc/agents/common/dataset.py's own already-verified
    compute_stats formula directly (same technique
    chunked_offline_replay_buffer.py already used), not reimplemented.

Storage: ONE SHARED flat array across all episodes (qc's own structure) -
not this repo's earlier per-episode-list design. Accepts the same small,
rare wraparound-seam risk qc's own published, working implementation
accepts, as the deliberate cost of the simpler, unified structure - see
conversation history for the full risk analysis and the explicit decision
to accept it.
"""
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from xqc.agents.common.rotation_transformer import axis_angle_to_rotation_6d
from xqc.replay_buffer.offline_replay_buffer import ROBOSUITE_OBS_KEYS, convert_abs_action_batch


def _peek_robot_layout(dataset_path: str) -> Tuple[int, int]:
    """Cheaply determines (num_robots, gripper_qpos_dim) by reading just the
    first demo's raw action/obs dims - num_robots is 1 for every existing
    single-arm task, 2 for TwoArmTransport (7 raw action dims per robot:
    pos3+axis-angle-rot3+grip1); gripper_qpos_dim is 2 for every robosuite
    task actually used here (Panda gripper) but read from the data rather
    than hardcoded. Avoids loading the whole dataset just to know the
    per-robot obs block width. Relocated from the former
    xqc/agents/common/dataset.py (see conversation history: that file is
    retired as part of unifying the IL and RL datasets)."""
    with h5py.File(dataset_path, "r") as hf:
        first_demo = next(iter(hf["data"].values()))
        num_robots = first_demo["actions"].shape[-1] // 7
        gripper_dim = first_demo["obs"]["robot0_gripper_qpos"].shape[-1]
        return num_robots, gripper_dim


def _held_out_episode_indices(n_episodes: int, val_ratio: float, val_seed: int) -> np.ndarray:
    """Replicates diffusion_policy/common/sampler.py's get_val_mask exactly.
    Relocated from the former xqc/agents/common/dataset.py."""
    if val_ratio <= 0:
        return np.zeros(0, dtype=np.int64)
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=val_seed)
    return rng.choice(n_episodes, size=n_val, replace=False)


class ChunkedNormalizationStats(NamedTuple):
    obs_offset: np.ndarray
    obs_scale: np.ndarray
    action_offset: np.ndarray
    action_scale: np.ndarray


class ChunkedBatch(NamedTuple):
    observations: np.ndarray       # (B, To, obs_dim) - ends at "now" (t)
    actions: np.ndarray            # (B, h, act_dim)  - actions[t : t+h]
    next_observations: np.ndarray  # (B, To, obs_dim) - ends at t+h
    next_actions: np.ndarray       # (B, h, act_dim)  - actions[t+h : t+2h], the SARSA bootstrap chunk
    rewards: np.ndarray            # (B,) - sum_{k=0}^{valid_len-1} gamma^k * r_{t+k}, valid_len=min(h, L-t) - TRUNCATED at the episode's own terminal (see sample_sequence), never contaminated by a reward from the next episode
    masks: np.ndarray              # (B,) - 0 once a terminal occurred within the h-step window (no valid next state to bootstrap from) - multiplies the bootstrap TARGET term (Q(next_obs, next_actions)), same as qc's own critic loss
    discount: np.ndarray           # (B,) - gamma**h (constant across the batch)
    valid: np.ndarray              # (B,) - always 1 now that `rewards` is truncated at the boundary by construction (see sample_sequence) - kept in the schema only so the 3 critic-loss call sites (which multiply the loss by it, matching qc's own critic_loss * batch['valid'] pattern) don't need their own signature to change; see xqc/replay_buffer/chunked_qc_buffer_qc_reference.py for qc's own literal version, where this flag does real work (0 whenever the window overran a boundary, discarding that sample)
    in_reference_region: np.ndarray  # (B,) - 1.0 if this sample's Ta-length action window would have satisfied the reference Diffusion Policy's own SequenceSampler cutoff (at most execution_horizon-1 padded slots in the SPECULATIVE tail beyond execution_horizon - see sample_sequence's own docstring), 0.0 otherwise. PURELY INFORMATIONAL - never affects sampling (every position is still sampled/conditioned-on regardless, see conversation history: excluding samples on this basis was the actual actor-critic coverage bug this repo had). Exists so the actor's own BC loss can OPTIONALLY mask by it - an ablation knob, not a default-on behavior - to reproduce the reference's own stricter loss-target-quality regime without losing the sampling fix. Irrelevant to the critic (RL fields are computed over `sequence_length`, not `Ta`, and never touch the speculative tail this flag is about).
    images: Dict[str, np.ndarray] = {}       # {camera_key: (B, To, H, W, 3)} - empty dict when camera_keys=() (state-only)
    next_images: Dict[str, np.ndarray] = {}  # {camera_key: (B, To, H, W, 3)} - ends at t+h, same convention as next_observations


def _convert_abs_action(action: np.ndarray, num_robots: int) -> np.ndarray:
    """pos(3) + axis-angle-rot(3) + gripper(1), per robot -> pos(3) +
    rotation_6d(6) + gripper(1), per robot - unchanged from
    chunked_parallel_replay_buffer.py's own helper."""
    stacked = action.reshape(num_robots, 7)
    pos, rot, gripper = stacked[:, :3], stacked[:, 3:6], stacked[:, 6:]
    rot6d = np.asarray(axis_angle_to_rotation_6d(jnp.asarray(rot)))
    return np.concatenate([pos, rot6d, gripper], axis=-1).reshape(num_robots * 10).astype(np.float32)


# ======================================================================
# FAITHFUL PORT of qc/utils/datasets.py - same names, same logic, kept
# present (even where unused by our own pipeline) so every function can be
# checked directly against the original - see module docstring.
# ======================================================================

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image."""
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Faithful port of qc/utils/datasets.py's Dataset - see module
    docstring. NONE of this class's own methods (sample/sample_sequence/
    get_subset/augment) are called by our own pipeline - our own logic
    lives in the separately-named methods further below."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields."""
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(
        self, batch_size: int, To: int, h: int, gamma: float, camera_keys: Sequence[str] = (),
        rng: Optional[np.random.Generator] = None, Ta: Optional[int] = None,
    ) -> ChunkedBatch:
        """MODIFIED from qc/utils/datasets.py's own Dataset.sample (see
        conversation history - same "you're allowed to change this" basis
        as sample_sequence below) - this IS our own offline entry point
        (train_critic_sarsa.py), reusing qc's own name/slot instead of a
        separately-named sample_chunked.

        Shuffled-without-replacement epochs, matching the old
        ChunkedOfflineReplayBuffer's own sample() cadence - qc's own
        sample() instead draws WITH replacement every call
        (get_random_idxs). `rng`: an explicit seeded generator the caller
        owns (see train_critic_sarsa.py's own shuffle_rng, seeded from
        cfg.critic.seed) - falls back to the global RNG when not given.

        Drops the frame_stack/p_aug/get_subset branches qc's own version
        had - not applicable anymore now that this returns a ChunkedBatch
        (via sample_sequence) instead of a raw dict; Dataset.__init__
        still sets those attributes faithfully, they're just no longer
        read by anything.

        Samples from EVERY position uniformly (no near-boundary exclusion) -
        same rule for every caller of this method, actor or critic alike
        (see conversation history: an earlier version gave the actor its
        own stricter "exclude near-tail starts" convention, matching the
        diffusion_policy reference; that turned out to be a real bug in an
        actor-critic setting, not just a stylistic difference - the actor
        never trained on states near an episode boundary, exactly the
        states a critic's bootstrap target most needs it to be competent
        at, and exactly where task success is usually decided. Dropped
        entirely - qc's own reference has no such distinction either)."""
        if not hasattr(self, "_epoch_order"):
            self._epoch_order = np.zeros(0, dtype=np.int64)
            self._epoch_pos = 0
        if self._epoch_pos + batch_size > len(self._epoch_order):
            self._epoch_order = (rng.permutation(self.size) if rng is not None else np.random.permutation(self.size))
            self._epoch_pos = 0
        idxs = self._epoch_order[self._epoch_pos : self._epoch_pos + batch_size]
        self._epoch_pos += batch_size
        return self.sample_sequence(idxs, sequence_length=h, discount=gamma, To=To, camera_keys=camera_keys, Ta=Ta)

    def sample_parallel_multibatch(
        self, batch_size: int, num_batches: int, To: int, h: int, gamma: float, camera_keys: Sequence[str] = (),
        rng: Optional[np.random.Generator] = None, Ta: Optional[int] = None,
    ) -> ChunkedBatch:
        """num_batches independent calls to sample() above (this class's
        own shuffled-epoch draw), stacked along a leading (num_batches,
        batch_size, ...) axis - the shape ChunkedPolicyLearner.update()
        expects (one full batch per update iteration). Named to mirror
        ParallelReplayBuffer's own sample_parallel_multibatch
        (train_parallel.py's non-chunked path) even though the underlying
        sampling mechanics differ (per-camera images, windowed To/h
        sequences, no num_seeds axis) - purely so the two driver scripts'
        call sites read the same way side by side. Formalizes what was
        previously inlined as a bare list-comprehension-then-stack at each
        call site (train_parallel_chunking.py's own offline critic batch,
        and its online offline/online batch mixing) - see conversation
        history."""
        batches = [
            self.sample(batch_size, To=To, h=h, gamma=gamma, camera_keys=camera_keys, rng=rng, Ta=Ta)
            for _ in range(num_batches)
        ]
        return jax.tree.map(lambda *xs: np.stack(xs), *batches)

    def sample_sequence(self, idxs, sequence_length, discount, To, camera_keys=(), Ta=None, mask_ta=None):
        """MODIFIED from qc/utils/datasets.py's own sample_sequence, kept as
        close to their own code/style as possible (see conversation
        history), retargeted to what our own pipeline actually needs:
          - observations: the To-step BACKWARD window ending at "now" (idxs)
          - actions: the chunk [t : t+sequence_length)
          - next_observations: the To-step BACKWARD window ending at the
            NEXT chunk's own start (t+sequence_length)
          - next_actions: the next chunk [t+sequence_length : t+2*sequence_length)

        `idxs` is a parameter (not drawn internally like qc's own version)
        since the three callers of this method (train_critic_sarsa.py's
        epoch-shuffled sample, get_val_batch, train_parallel_chunking.py's
        online loop) each need a different way of picking positions - see
        conversation history for the full "what is idxs" discussion.

        `Ta` (defaults to `sequence_length`): widens ONLY the returned
        `actions` window to `Ta` steps (near-padded the same way past
        `sequence_length`) - the IL actor's own optional training-time
        lookahead beyond what actually gets executed (see conversation
        history: `execution_horizon`==`sequence_length` is the one shared
        quantity between the actor's real execution and the critic's own
        `h`; `Ta` exists only to serve the actor's loss, so `next_actions`
        - purely an RL bootstrap field IL never reads - and
        rewards/masks/valid/next_anchor all stay anchored at
        `sequence_length` regardless of `Ta`).

        `mask_ta` (defaults to `Ta`, i.e. today's exact behavior when
        omitted): the window width used ONLY for in_reference_region's own
        padding check below - decoupled from `Ta` so a caller can get a
        NARROW `actions` window (e.g. execution_horizon, what the online
        critic needs - see xqc/agents/critics/*/critic.py, which read
        batch.actions directly with no slicing, so a Ta-widened actions
        array would corrupt the critic's own fixed input width) while
        still computing in_reference_region as if checking a WIDER
        lookahead (matching the offline BC call's own Ta, for the online
        actor's mask_reference_region loss weighting - see
        xqc/agents/chunking/actor.py's own update_actor). Existing callers
        (BC, get_val_batch, critic-only calls) never pass this, so
        mask_ta=Ta preserves their exact current behavior unchanged - see
        conversation history for the full "why not just widen Ta" tradeoff
        (widening Ta for the online call directly fed Ta-wide actions to
        the critic and broke its fixed input shape).

        Forward-looking pieces (actions/next_actions/rewards/masks) are
        boundary-AWARE and near-pad at a genuine episode boundary (repeat
        the last real action, truncate the reward sum to `valid_len =
        min(h, L - t)`) - NOT qc's own literal blind-clamp-then-discard-via-
        valid convention (see xqc/replay_buffer/chunked_qc_buffer_qc_reference.py
        for that literal version, and this module's own docstring for why:
        a real training-curve comparison showed qc's convention silently
        throwing away a large, reward-concentrated fraction of near-terminal
        training data on this repo's own robomimic tasks). `valid` is
        always 1 here (the reward sum can no longer be contaminated by
        construction) - kept in the schema only so the 3 critic-loss call
        sites don't need to change.

        The two BACKWARD windows (observations/next_observations, plus
        images) use the same near-pad technique (repeats the earliest REAL
        frame reached), checked live against self['terminals'] at each
        step. Written twice (once per anchor) rather than factored into a
        helper method, matching qc's own flat, non-factored style - the
        forward walk below follows the same convention, also written twice
        (once for the current chunk, once for next_actions).

        Returns a ChunkedBatch (this repo's own existing contract, unchanged
        except for the new `valid` field - see that class's own docstring)
        rather than qc's own dict shape: rewards/masks/valid are collapsed
        to their LAST position only (`[:, -1]`) before returning - the same
        single values qc's own agent code itself extracts from the full
        per-step arrays (see conversation history) - since nothing
        downstream ever needs the earlier, intermediate steps, only the
        final one. `terminals` is dropped entirely (computed only to derive
        `valid`, never read by anything on its own)."""
        # Forward walk, boundary-aware and near-padded, reproducing the OLD
        # chunked_offline_replay_buffer.py's `valid_len = min(h, L - t)`
        # truncation + repeat-last-real-action padding - via the same
        # live-terminals-check walk already used below for the backward
        # windows, instead of qc's own blind np.minimum(..., size-1) clamp.
        #
        # Worked example (episode A = A1,A2,A3 terminal at A3, episode B =
        # B1,B2; h=3, idxs=position of A3): old code produced
        # actions=[A3,A3,A3] (repeat the last real action), reward=r3 only
        # (no fake contribution from the 2 repeated slots), masks=0 (next
        # chunk anchor A3+3 doesn't exist within episode A) - see
        # conversation history for the full derivation. `is_real[:, k]`
        # below tracks exactly this: True while the walk is still reading
        # genuine, un-repeated steps of THIS SAME episode; once it goes
        # terminal, every later slot in the chunk repeats that same
        # terminal position (near-pad) and stops counting toward the
        # reward sum.
        terminals_flat = self["terminals"]

        # action_window_len: the walk itself always covers at least
        # sequence_length (execution_horizon) steps - Ta only widens the
        # ACTIONS returned, everything RL-relevant (reward/mask/next_anchor)
        # stays anchored at sequence_length regardless (see docstring).
        #
        # returned_action_len (what actions/batch_actions actually get
        # sliced to below) depends ONLY on Ta, exactly like before mask_ta
        # existed - mask_ta must NOT widen the returned actions window
        # (that's the entire point of the two being separate params - see
        # mask_ta's own docstring above). action_window_len (what the WALK
        # itself covers, i.e. fwd_idx/is_real's own width) additionally
        # covers mask_ta, purely so in_reference_region's own slice below
        # (is_real[:, :mask_ta]) has real data to read - actions/
        # batch_actions get sliced back down to returned_action_len right
        # after the walk, discarding the extra mask_ta-only columns.
        _mask_ta = mask_ta if mask_ta is not None else Ta
        returned_action_len = max(sequence_length, Ta) if Ta is not None else sequence_length
        action_window_len = max(returned_action_len, _mask_ta or 0)

        fwd_idx = np.empty((idxs.shape[0], action_window_len), dtype=np.int64)
        is_real = np.ones((idxs.shape[0], action_window_len), dtype=bool)
        fwd_idx[:, 0] = idxs
        cur = idxs.copy()
        for k in range(1, action_window_len):
            at_terminal = terminals_flat[cur] > 0
            is_real[:, k] = is_real[:, k - 1] & (~at_terminal)
            candidate = np.minimum(cur + 1, self.size - 1)
            cur = np.where(at_terminal, cur, candidate)
            fwd_idx[:, k] = cur

        batch_actions = self['actions'][fwd_idx[:, :returned_action_len].flatten()].reshape(idxs.shape[0], returned_action_len, *self['actions'].shape[1:])
        actions = batch_actions  # (batch_size, returned_action_len, action_dim) - near-padded, repeats last real action past the episode boundary

        # in_reference_region: replicates the reference Diffusion Policy's
        # own SequenceSampler cutoff (`t_stop`) as a PURE INFORMATIONAL flag
        # (never excludes anything from being sampled - see conversation
        # history and ChunkedBatch's own docstring for the full "why").
        # Their own rule, re-derived algebraically from t_stop = L - Ta +
        # execution_horizon (see conversation history for the full
        # derivation): the TOTAL padded-slot count across the WHOLE
        # mask_ta-length window (not just the speculative tail past
        # execution_horizon - an earlier version of this formula only
        # checked the speculative tail, which is wrong whenever the
        # EXECUTED portion itself also needs padding, e.g. an episode
        # shorter than execution_horizon - caught by
        # scripts/test_sample_sequence_forward_boundary.py's own
        # independently re-derived cross-check) must be at most
        # execution_horizon-1. Uses mask_ta, not Ta, so a caller can widen
        # just the padding check without widening the returned actions
        # window itself (see mask_ta's own docstring above - the online
        # actor's mask_reference_region needs exactly this: a critic-safe
        # narrow actions window, paired with a BC-equivalent wide reference
        # check). Only meaningful when mask_ta widens past sequence_length
        # at all - with no widening (mask_ta<=sequence_length, e.g. the
        # critic's own calls, which never pass Ta/mask_ta), every sample is
        # trivially "in region" (t_stop >= L always in that case).
        if _mask_ta is not None and _mask_ta > sequence_length:
            total_padded_count = _mask_ta - is_real[:, :_mask_ta].sum(axis=1)
            in_reference_region = (total_padded_count <= sequence_length - 1).astype(np.float32)
        else:
            in_reference_region = np.ones(idxs.shape[0], dtype=np.float32)

        batch_rewards_raw = self['rewards'][fwd_idx[:, :sequence_length].flatten()].reshape(idxs.shape[0], sequence_length)
        effective_rewards = np.where(is_real[:, :sequence_length], batch_rewards_raw, 0.0)  # padded slots contribute nothing - matches old valid_len truncation
        discount_powers = discount ** np.arange(sequence_length)
        rewards = np.cumsum(effective_rewards * discount_powers[None, :], axis=1)  # (batch_size, sequence_length) - only the LAST column is used below

        # masks: kept as qc's OWN original cumulative-min loop, unchanged -
        # not re-derived - just fed batch_masks read at the corrected
        # (near-padded) fwd_idx instead of qc's naive clamped all_idxs. This
        # works without any extra "is_real" logic (unlike the reward sum
        # above) because re-reading the same terminal position's mask=0
        # repeatedly and taking a running min is already idempotent - no
        # double-counting risk the way summing a repeated reward has. See
        # conversation history: this was over-rewritten in an earlier draft
        # via a separate is_real-derived formula; simplified back to qc's
        # own structure once confirmed equivalent.
        batch_masks = self['masks'][fwd_idx[:, :sequence_length].flatten()].reshape(idxs.shape[0], sequence_length)
        masks = np.empty((idxs.shape[0], sequence_length), dtype=float)
        masks[:, 0] = batch_masks[:, 0]
        for k in range(1, sequence_length):
            masks[:, k] = np.minimum(masks[:, k - 1], batch_masks[:, k])
        valid = np.ones((idxs.shape[0], sequence_length), dtype=float)  # reward sum can no longer be contaminated by construction - see module docstring

        # Next chunk starts one more step past the BASE (sequence_length,
        # not Ta-widened) walk - anchored at execution_horizon regardless of
        # Ta, same as reward/mask above (near-padded the same way: stay if
        # the last chunk slot is itself terminal).
        last_walk_idx = fwd_idx[:, sequence_length - 1]
        last_at_terminal = terminals_flat[last_walk_idx] > 0
        next_anchor = np.where(last_at_terminal, last_walk_idx, np.minimum(last_walk_idx + 1, self.size - 1))

        # Second forward walk, starting at next_anchor instead of idxs -
        # same technique, written out again rather than factored into a
        # helper (matching this file's own existing convention for the
        # backward windows just below).
        next_fwd_idx = np.empty((idxs.shape[0], sequence_length), dtype=np.int64)
        next_fwd_idx[:, 0] = next_anchor
        cur = next_anchor.copy()
        for k in range(1, sequence_length):
            at_terminal = terminals_flat[cur] > 0
            candidate = np.minimum(cur + 1, self.size - 1)
            cur = np.where(at_terminal, cur, candidate)
            next_fwd_idx[:, k] = cur
        batch_next_actions = self['actions'][next_fwd_idx.flatten()].reshape(idxs.shape[0], sequence_length, *self['actions'].shape[1:])
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)

        # --- Backward To-step window ending at "now" (idxs) - observations/images.
        back_idx = np.empty((idxs.shape[0], To), dtype=np.int64)
        back_idx[:, To - 1] = idxs
        cur = idxs.copy()
        blocked = np.zeros(idxs.shape[0], dtype=bool)
        for j in range(1, To):
            candidate = cur - 1
            newly_blocked = (candidate < 0) | (terminals_flat[np.clip(candidate, 0, None)] > 0)
            blocked = blocked | newly_blocked
            cur = np.where(blocked, cur, candidate)
            back_idx[:, To - 1 - j] = cur
        observations = self["observations"][back_idx]
        images = {cam: self[cam][back_idx] for cam in camera_keys}

        # --- Backward To-step window ending at next_anchor (t+sequence_length)
        # - next_observations/next_images. Same technique as above, just
        # centered on next_anchor instead of idxs.
        next_back_idx = np.empty((idxs.shape[0], To), dtype=np.int64)
        next_back_idx[:, To - 1] = next_anchor
        cur = next_anchor.copy()
        blocked = np.zeros(idxs.shape[0], dtype=bool)
        for j in range(1, To):
            candidate = cur - 1
            newly_blocked = (candidate < 0) | (terminals_flat[np.clip(candidate, 0, None)] > 0)
            blocked = blocked | newly_blocked
            cur = np.where(blocked, cur, candidate)
            next_back_idx[:, To - 1 - j] = cur
        next_observations = self["observations"][next_back_idx]
        next_images = {cam: self[cam][next_back_idx] for cam in camera_keys}

        # Collapse to the single final value per sample - see docstring.
        # discount is a plain constant (gamma**sequence_length), same for
        # every sample given one fixed sequence_length/discount per call -
        # broadcast to (B,) to match ChunkedBatch's own existing convention.
        discount_h = np.full(idxs.shape, np.float32(discount) ** sequence_length, dtype=np.float32)
        return ChunkedBatch(
            observations=observations.astype(np.float32),
            actions=actions.astype(np.float32),
            next_observations=next_observations.astype(np.float32),
            next_actions=next_actions.astype(np.float32),
            rewards=rewards[:, -1].astype(np.float32),
            masks=masks[:, -1].astype(np.float32),
            discount=discount_h,
            valid=valid[:, -1].astype(np.float32),
            in_reference_region=in_reference_region,
            images=images,
            next_images=next_images,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self._dict["actions"][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    # ==================================================================
    # NEW / OURS - see module docstring. Our own pipeline calls these.
    # ==================================================================

    def get_val_batch(
        self, To: int, h: int, gamma: float, camera_keys: Sequence[str] = (), max_size: Optional[int] = None,
        Ta: Optional[int] = None,
    ) -> ChunkedBatch:
        """The full dataset as one ChunkedBatch (no shuffling) - for a
        periodic no-gradient held-out check, same as the old
        ChunkedOfflineReplayBuffer.get_val_batch. Call this on a Dataset
        built from only the held-out episodes (see load_robomimic_flat).
        Every position, same as sample() - see its own docstring on why
        there's no near-boundary exclusion here."""
        n = self.size if max_size is None else min(max_size, self.size)
        idx = np.arange(n)
        return self.sample_sequence(idx, sequence_length=h, discount=gamma, To=To, camera_keys=camera_keys, Ta=Ta)


# NOTE: qc/utils/datasets.py's module-level add_history is deliberately NOT
# ported here - confirmed via grep that it has zero call sites anywhere in
# qc's own repo (not even from their own main.py/main_online.py), unlike
# every other faithfully-ported function/method above, which - even where
# unused by OUR pipeline - is at least genuinely exercised by theirs. Its
# job (backward observation-history windowing) is what
# Dataset._backward_window above does instead.


class ReplayBuffer(Dataset):
    """Faithful port of qc/utils/datasets.py's ReplayBuffer - see module
    docstring. Online sampling goes through the inherited sample_sequence
    directly (train_parallel_chunking.py draws indices and reshapes inline,
    matching qc's own main_online.py - see module docstring's NOTE)."""

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset. Ported, present
        for future use, currently unused by our own scripts (they keep two
        separate buffer instances) - see conversation history."""

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

    # ==================================================================
    # NEW / OURS - see module docstring.
    # ==================================================================

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        truncs: bool,
        next_observation: np.ndarray,
        images: Optional[Dict[str, np.ndarray]] = None,
        abs_action: bool = False,
        num_robots: int = 1,
    ) -> None:
        """Drop-in replacement for ChunkedParallelReplayBuffer.insert - same
        argument list/semantics (num_seeds batch-dim squeeze, abs_action
        conversion) - see that class's own docstring, unchanged here.
        next_observation is accepted but unused (same reason as before: the
        "next" window is read directly off the same flat `observations`
        array at a computed offset, never stored separately)."""
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if observation.ndim == 2:
            assert observation.shape[0] == 1, (
                f"ReplayBuffer.insert only supports num_seeds=1 (ChunkedPolicyLearner has no "
                f"multi-seed/vmap support), got a batch of {observation.shape[0]}"
            )
            observation, action = observation[0], action[0]
            reward, mask, truncs = np.asarray(reward)[0], np.asarray(mask)[0], np.asarray(truncs)[0]
            if images is not None:
                images = {cam: np.asarray(img)[0] for cam, img in images.items()}

        if abs_action:
            action = _convert_abs_action(action, num_robots)

        transition = dict(
            observations=observation,
            actions=action,
            rewards=np.float32(reward),
            terminals=np.float32(0.0 if mask else 1.0),
            masks=np.float32(mask),
        )
        if images is not None:
            for cam, img in images.items():
                transition[cam] = np.asarray(img, dtype=np.uint8)
        self.add_transition(transition)


# ======================================================================
# Offline (robomimic hdf5) loading - NEW / OURS, see module docstring.
# ======================================================================

def load_robomimic_flat(
    dataset_path: str,
    n_demos: int,
    abs_action: bool,
    camera_keys: Sequence[str],
    val_ratio: float,
    val_seed: int,
) -> Tuple[Dataset, Dataset, np.ndarray, np.ndarray, int, int]:
    """Loop-over-episodes-and-concatenate loader, modeled directly on qc's
    own envs/robomimic_utils.py:get_dataset, but built on this repo's own
    per-episode reading logic (ROBOSUITE_OBS_KEYS, L=argmax(dones)+1,
    abs_action conversion) since qc's own loader is state-only with no
    abs_action/image support at all - see module docstring. New/ours, not a
    faithful port (get_dataset lives in a different qc file entirely,
    envs/robomimic_utils.py, not utils/datasets.py - see conversation
    history), so it isn't held to that naming convention either.

    Returns (train_dataset, val_dataset, raw_obs_all_episodes,
    raw_actions_all_episodes, num_robots, gripper_dim) - a plain tuple, not
    a wrapper object (see conversation history: qc's own repo has exactly
    two classes, Dataset and ReplayBuffer(Dataset) - no third
    config-holding class on top of a Dataset - so callers here hold
    To/h/discount/camera_keys themselves and pass them explicitly at each
    call site, same as qc/main.py itself does). raw_obs_all_episodes/
    raw_actions_all_episodes (train+val concatenated) are what
    compute_chunked_stats needs, matching ChunkedOfflineReplayBuffer's own
    documented reasoning (windowing can only ever duplicate already-
    observed values, never introduce new extremes, so held-out episodes
    must still contribute to the stats). num_robots/gripper_dim (via
    _peek_robot_layout) are what compute_chunked_stats's optional
    per_dim_range normalizer branch needs."""
    num_robots, gripper_dim = _peek_robot_layout(dataset_path)
    held_out = set(_held_out_episode_indices(n_demos, val_ratio, val_seed).tolist())

    # "object" (privileged ground-truth object pose, not observable from
    # pixels) must NEVER leak into a vision agent's observation - same
    # exclusion xqc/agents/common/dataset.py's ChunkedOfflineImageDataset
    # applies (see its own "Privileged state" module-docstring note). Only
    # relevant once camera_keys is non-empty; state-only agents keep it.
    obs_keys = [k for k in ROBOSUITE_OBS_KEYS if k != "object"] if camera_keys else ROBOSUITE_OBS_KEYS

    fields = {"train": {"observations": [], "actions": [], "rewards": [], "terminals": [], "masks": []},
              "val": {"observations": [], "actions": [], "rewards": [], "terminals": [], "masks": []}}
    for cam in camera_keys:
        fields["train"][cam] = []
        fields["val"][cam] = []

    raw_obs_all: List[np.ndarray] = []
    raw_actions_all: List[np.ndarray] = []

    with h5py.File(dataset_path, "r") as hf:
        for k, (_, d) in enumerate(hf["data"].items()):
            if k == n_demos:
                break
            split = "val" if k in held_out else "train"
            dst = fields[split]

            # L = number of real timesteps in this episode - same
            # convention as OfflineReplayBuffer/ChunkedOfflineReplayBuffer.
            L = int(np.argmax(d["dones"])) + 1
            obs = np.concatenate([d["obs"][key][:L] for key in obs_keys], axis=-1).astype(np.float32)
            actions = np.array(d["actions"][:L], dtype=np.float32)
            if abs_action:
                # Once-per-episode conversion, shared with dataset.py's own
                # IL loader - see convert_abs_action_batch's own docstring.
                # Uses the outer num_robots (from _peek_robot_layout above,
                # same for every episode in this dataset) rather than
                # re-deriving a same-named local one.
                actions = convert_abs_action_batch(actions, num_robots)
            rewards = np.array(d["rewards"][:L], dtype=np.float32)
            terminals = np.zeros(L, dtype=np.float32)
            terminals[-1] = 1.0
            masks = 1.0 - terminals

            raw_obs_all.append(obs)
            raw_actions_all.append(actions)
            dst["observations"].append(obs)
            dst["actions"].append(actions)
            dst["rewards"].append(rewards)
            dst["terminals"].append(terminals)
            dst["masks"].append(masks)
            for cam in camera_keys:
                dst[cam].append(np.array(d["obs"][cam][:L], dtype=np.uint8))

    def _finalize(dst: Dict[str, list]) -> Dataset:
        merged = {key: (np.concatenate(arrs, axis=0) if arrs else np.zeros((0,), dtype=np.float32))
                  for key, arrs in dst.items()}
        return Dataset.create(**merged)

    train_dataset = _finalize(fields["train"])
    val_dataset = _finalize(fields["val"])
    raw_obs_all_episodes = np.concatenate(raw_obs_all, axis=0) if raw_obs_all else np.zeros((0, 0), dtype=np.float32)
    raw_actions_all_episodes = np.concatenate(raw_actions_all, axis=0) if raw_actions_all else np.zeros((0, 0), dtype=np.float32)
    return train_dataset, val_dataset, raw_obs_all_episodes, raw_actions_all_episodes, num_robots, gripper_dim


def _compute_stats_image(dataset: Dataset) -> ChunkedNormalizationStats:
    """Vision-observation normalizer - mirrors RobomimicReplayImageDataset's
    own get_normalizer: per-dimension range-to-[-1,1] for the eef_pos/
    gripper_qpos fields, identity for the eef_quat slice and for actions
    (delta-action, already ~[-1,1]), including its near-constant-dimension
    fallback (range < 1e-7 -> scale=1, offset=input_min, not a clipped-
    small-range spread). Relocated verbatim from the former
    xqc/agents/common/dataset.py's ChunkedOfflineImageDataset.compute_stats
    (see conversation history: that file is retired as part of unifying
    the IL and RL datasets) - reads the same attributes
    compute_chunked_stats below sets on `dataset` directly (a plain Dataset
    instance supports extra attributes fine, same trick as frame_stack/
    p_aug), just `dataset.X` instead of the former `self.X`."""
    obs = dataset._raw_obs_all_episodes
    act_dim = dataset.actions.shape[-1]
    quat_start, quat_end = dataset.eef_pos_dim, dataset.eef_pos_dim + dataset.eef_quat_dim

    obs_min = np.min(obs, axis=0)
    obs_max = np.max(obs, axis=0)
    obs_range = obs_max - obs_min
    ignore = obs_range < dataset.range_eps
    obs_scale = np.where(ignore, 1.0, obs_range / 2.0).astype(np.float32)
    obs_offset = np.where(ignore, obs_min, (obs_min + obs_max) / 2.0).astype(np.float32)
    # Quaternion slice: identity (no normalization), preserves unit-norm structure.
    obs_offset[quat_start:quat_end] = 0.0
    obs_scale[quat_start:quat_end] = 1.0

    if dataset.abs_action:
        # Same robomimic_abs_action_only_normalizer_from_stat scheme as
        # _compute_stats_lowdim's own abs_action branch (see its comment
        # for the full derivation) - EACH robot's own pos (first 3 of its
        # own 10-dim [pos3,rotation_6d6,gripper1] block) gets an
        # independent per-dim range-to-[-1,1]; rotation_6d(6)+gripper(1)
        # stay identity. num_robots derived from act_dim itself (10 dims
        # per robot post-conversion) rather than a stored dataset.num_robots
        # - vision datasets don't track that separately (single-arm only
        # for now) - act_dim//10 is a no-op (1) for every existing
        # single-arm task.
        num_robots = act_dim // 10
        actions = dataset._raw_actions_all_episodes
        action_offset = np.zeros(act_dim, dtype=np.float32)
        action_scale = np.ones(act_dim, dtype=np.float32)
        for i in range(num_robots):
            pos = actions[:, i * 10 : i * 10 + 3]
            pos_min = np.min(pos, axis=0)
            pos_max = np.max(pos, axis=0)
            pos_range = pos_max - pos_min
            pos_ignore = pos_range < 1e-7
            pos_scale = np.where(pos_ignore, 1.0, pos_range / 2.0).astype(np.float32)
            pos_offset = np.where(pos_ignore, pos_min, (pos_min + pos_max) / 2.0).astype(np.float32)
            action_offset[i * 10 : i * 10 + 3] = pos_offset
            action_scale[i * 10 : i * 10 + 3] = pos_scale
    else:
        # Identity - matches the reference's own get_identity_normalizer_
        # from_stat for the delta-action case (already-normalized deltas).
        action_offset = np.zeros(act_dim, dtype=np.float32)
        action_scale = np.ones(act_dim, dtype=np.float32)

    return ChunkedNormalizationStats(
        obs_offset=obs_offset,
        obs_scale=obs_scale,
        action_offset=action_offset,
        action_scale=action_scale,
    )


def _compute_stats_lowdim(dataset: Dataset) -> ChunkedNormalizationStats:
    """State-observation normalizer - mirrors RobomimicReplayLowdimDataset's
    own get_normalizer/normalizer_from_stat: default ("global") is a SINGLE
    global scalar scale (1 / max absolute value, computed over the ENTIRE
    concatenated obs vector - no per-dimension distinction, no centering,
    no quaternion exception, "object" included) - deliberately different
    from _compute_stats_image's own per-key range normalizer above (the
    reference genuinely does something different between its image and
    lowdim task configs here, not an oversight).
    normalizer="per_dim_range" is an alternate, NOT reference-matching
    scheme kept only as a debugging lever - see conversation history.
    Relocated verbatim from the former xqc/agents/common/dataset.py's
    ChunkedOfflineLowdimDataset.compute_stats - see _compute_stats_image's
    own docstring for why (dataset.X instead of the former self.X)."""
    obs = dataset._raw_obs_all_episodes
    act_dim = dataset.actions.shape[-1]

    # Obs normalization first.
    if dataset.normalizer == "global":
        # np.abs(obs).max() == max(colmax.max(), abs(colmin).max()) (the
        # reference's own normalizer_from_stat formula).
        obs_abs_max = np.abs(obs).max()
        obs_scale = np.full(obs.shape[-1], obs_abs_max, dtype=np.float32)
        obs_offset = np.zeros(obs.shape[-1], dtype=np.float32)
    elif dataset.normalizer == "per_dim_range":
        # Pre-refactor xqcfd formula (xqc/agents/common/normalization.py's
        # old compute_stats_from_dataset, eps=1e-3) - see this function's
        # own docstring for why this exists. Per-robot quat block (robot
        # i's own eef_quat sits at offset i*robot_obs_width+eef_pos_dim,
        # robot_obs_width = eef_pos_dim+eef_quat_dim+gripper_dim - matches
        # robosuite_obs_keys_for_num_robots's own per-robot
        # [pos,quat,gripper] concatenation order) - a no-op loop of 1 for
        # every existing single-arm task, num_robots=2 for TwoArmTransport.
        robot_obs_width = dataset.eef_pos_dim + dataset.eef_quat_dim + dataset._gripper_dim
        obs_min = np.min(obs, axis=0)
        obs_max = np.max(obs, axis=0)
        obs_offset = ((obs_min + obs_max) / 2).astype(np.float32)
        obs_scale = np.clip((obs_max - obs_min) / 2, a_min=1e-3, a_max=None).astype(np.float32)
        for i in range(dataset.num_robots):
            quat_start = i * robot_obs_width + dataset.eef_pos_dim
            quat_end = quat_start + dataset.eef_quat_dim
            obs_offset[quat_start:quat_end] = 0.0
            obs_scale[quat_start:quat_end] = 1.0
    elif dataset.normalizer == "identity":
        # No observation normalization at all (explicit request - see
        # conversation history): matches xqc/agents/xqc/xqc_learner.py's own
        # reference pipeline exactly, which has no obs_offset/obs_scale
        # concept anywhere - raw observations go straight into the network.
        # NOT the default (see conf/actor/training/state.yaml's own
        # lowdim_normalizer field) - "global" stays the default, reproducing
        # today's behavior for every existing agent unless a config
        # explicitly opts into this.
        obs_offset = np.zeros(obs.shape[-1], dtype=np.float32)
        obs_scale = np.ones(obs.shape[-1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown normalizer {dataset.normalizer!r}, expected 'global', 'per_dim_range', or 'identity'")

    # Action normalization.
    if dataset.abs_action:
        # Matches the reference's own robomimic_abs_action_only_normalizer_
        # from_stat (single-arm) / robomimic_abs_action_only_dual_arm_
        # normalizer_from_stat (num_robots==2) exactly (diffusion_policy/
        # common/normalize_util.py): EACH robot's own pos (first 3 of its
        # own 10-dim [pos3,rotation_6d6,gripper1] block) gets an
        # independent per-dim range-to-[-1,1] (same get_range_normalizer_
        # from_stat formula as the "per_dim_range" obs branch above,
        # range_eps=1e-7 - reference's own default for this specific
        # normalizer); everything else per robot (rotation_6d(6) +
        # gripper(1)) is identity - the reference leaves those dims
        # untouched (get_other_param_info: scale=1, offset=0). A no-op loop
        # of 1 robot's worth for every existing single-arm task.
        actions = dataset._raw_actions_all_episodes
        action_offset = np.zeros(act_dim, dtype=np.float32)
        action_scale = np.ones(act_dim, dtype=np.float32)
        for i in range(dataset.num_robots):
            pos = actions[:, i * 10 : i * 10 + 3]
            pos_min = np.min(pos, axis=0)
            pos_max = np.max(pos, axis=0)
            pos_range = pos_max - pos_min
            pos_ignore = pos_range < 1e-7
            pos_scale = np.where(pos_ignore, 1.0, pos_range / 2.0).astype(np.float32)
            pos_offset = np.where(pos_ignore, pos_min, (pos_min + pos_max) / 2.0).astype(np.float32)
            action_offset[i * 10 : i * 10 + 3] = pos_offset
            action_scale[i * 10 : i * 10 + 3] = pos_scale
    else:
        # Identity - matches the reference's own get_identity_normalizer_
        # from_stat for the delta-action (abs_action=False) case (already-
        # normalized deltas).
        action_offset = np.zeros(act_dim, dtype=np.float32)
        action_scale = np.ones(act_dim, dtype=np.float32)

    return ChunkedNormalizationStats(
        obs_offset=obs_offset,
        obs_scale=obs_scale,
        action_offset=action_offset,
        action_scale=action_scale,
    )


def compute_chunked_stats(
    dataset: Dataset,
    raw_obs_all_episodes: np.ndarray,
    raw_actions_all_episodes: np.ndarray,
    camera_keys: Sequence[str],
    abs_action: bool,
    normalizer: str = "global",
    eef_pos_dim: int = 3,
    eef_quat_dim: int = 4,
    range_eps: float = 1e-7,
    gripper_dim: int = 0,
    num_robots: int = 1,
) -> ChunkedNormalizationStats:
    """Uses this module's own already-verified _compute_stats_image/
    _compute_stats_lowdim formulas (see conversation history: formerly
    lived as ChunkedOfflineImageDataset.compute_stats/
    ChunkedOfflineLowdimDataset.compute_stats in xqc/agents/common/dataset.py,
    relocated here once that file's only other job - IL windowing - moved
    onto this module's own Dataset too) - a standalone function, not a
    method on a wrapper class (see load_robomimic_flat's own docstring:
    qc's repo has exactly two classes, no third config-holding one).

    Sets the attributes _compute_stats_image/_compute_stats_lowdim expect
    directly on `dataset` - a plain Dataset instance supports extra
    attributes fine (same as frame_stack/p_aug already do)."""
    dataset._raw_obs_all_episodes = raw_obs_all_episodes
    dataset._raw_actions_all_episodes = raw_actions_all_episodes
    dataset.actions = raw_actions_all_episodes
    dataset.normalizer = normalizer
    dataset.eef_pos_dim = eef_pos_dim
    dataset.eef_quat_dim = eef_quat_dim
    dataset.range_eps = range_eps
    dataset.abs_action = abs_action
    dataset._gripper_dim = gripper_dim
    dataset.num_robots = num_robots
    if camera_keys:
        return _compute_stats_image(dataset)
    return _compute_stats_lowdim(dataset)


