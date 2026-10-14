"""MIP (Minimum Iterative Policy): a deterministic, 2-step flow-map BC
policy over action chunks. Ported from xqc/networks/mip/losses.py's
mip_loss + xqc/networks/mip/samplers.py's mip_sampler.

Wraps xqc/agents/common/network.py's ChiUNet with the uniform
`.loss(act, cond, rng) -> (loss, infos)` / `.sample(rng, shape, cond) ->
action` interface xqc/agents/chunking/chunked_learner.py's ChunkedPolicyLearner
expects (matching DiffusionPolicy/FlowMatching/MeanFlow) - this used to be a
free-function pair (mip_loss/mip_sampler) calling a bare ChiUNet directly,
with t_two_step/norm_type/loss_scale/target_mode threaded as extra call
args; they're Flax dataclass fields here instead, so mip_loss/mip_sampler's
old per-call arguments collapse into ordinary jit-static config baked into
`network_apply_fn` before jit, exactly like DiffusionPolicy's own
num_train_timesteps/beta_schedule/etc. `.sample` accepts (and ignores) `rng`
purely for call-site uniformity - MIP's own inference is fully
deterministic, no randomness anywhere."""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from xqc.agents.common.loss_masking import masked_mean
from xqc.agents.common.network import ChiUNet
from xqc.agents.common.squashing import safe_arctanh


def _get_norm(x: jnp.ndarray, norm_type: str) -> jnp.ndarray:
    if norm_type == "l2":
        return jnp.sum(x * x, axis=-1)
    elif norm_type == "l1":
        return jnp.sum(jnp.abs(x), axis=-1)
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


class MIP(nn.Module):
    """Deterministic 2-step flow-map policy over action chunks, ChiUNet-
    backed. act_dim, Ta, cond_dim, To: same meaning/shapes as
    DiffusionPolicy's own fields - condition is (B, To, cond_dim), and
    actions are (B, Ta, act_dim)."""

    act_dim: int
    Ta: int
    cond_dim: int
    To: int
    model_dim: int = 256
    emb_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    dim_mult: Sequence[int] = (1, 2, 2)
    n_groups: int = 8
    t_two_step: float = 0.9
    norm_type: str = "l2"
    loss_scale: float = 100.0
    # "approx" (default): act_pred_0 (queried at the fixed noise point
    # act_0=0) regresses directly to `act`, a simplification that's only
    # exact in the t_two_step->1 limit - the true intermediate target is
    # t_two_step*act (see the mse_mid_point diagnostic in loss(), which
    # already reads out this bias regardless of target_mode). "exact":
    # act_pred_0 regresses to t_two_step*act instead - see loss()'s own
    # docstring for the full derivation/tradeoff.
    target_mode: str = "approx"
    # squash_actions (same convention as MeanFlow, xqc/agents/actors/mf/
    # mean_flow.py): False (default) - MIP trains on and outputs the
    # normalized action directly (unbounded output). True - loss() trains on
    # the pre-tanh coordinate (safe_arctanh(act) - squash_offset) /
    # squash_scale, and sample() maps the network output back with
    # tanh(y * squash_scale + squash_offset), so actions are hard-bounded to
    # (-1, 1) and the online Q-gradient through sample() saturates near the
    # bounds. squash_scale/squash_offset: per-action-dim tuples computed from
    # the dataset by MIPLearner (its squash_scale arg) - None = 1 / 0.
    squash_actions: bool = False
    squash_scale: Optional[Tuple[float, ...]] = None
    squash_offset: Optional[Tuple[float, ...]] = None

    def setup(self):
        self.net = ChiUNet(
            act_dim=self.act_dim,
            Ta=self.Ta,
            obs_dim=self.cond_dim,
            To=self.To,
            model_dim=self.model_dim,
            emb_dim=self.emb_dim,
            kernel_size=self.kernel_size,
            cond_predict_scale=self.cond_predict_scale,
            n_groups=self.n_groups,
            obs_as_global_cond=True,
            dim_mult=self.dim_mult,
            # get_velocity below always calls the network with s=t - the
            # dual s/t embedding split is pure redundancy here, not two
            # independent signals - see xqc/agents/actors/mf/mean_flow.py's own
            # (genuinely two-time-scale) use_full_width_t_embedding=False
            # for the contrast.
            use_full_width_t_embedding=True,
            # Repo-wide "shared prediction backbone" decision - see
            # xqc/agents/common/obs_encoder.py's IdentityObsEncoder
            # docstring.
            zero_init_final_conv=False,
            project_condition=False,
        )

    def get_velocity(self, t: jnp.ndarray, xs: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """Direct network prediction at a single time value (s = t),
        matching xqc/networks/mip/flow_map.py:FlowMap.get_velocity."""
        velocity, _ = self.net(xs, t, t, cond)
        return velocity

    def loss(
        self, act: jnp.ndarray, cond: jnp.ndarray, rng: jax.random.PRNGKey, mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, dict]:
        """act: (B, Ta, act_dim) demonstration actions, normalized to
        (-1, 1). cond: (B, To, cond_dim). mask: optional (B,) per-sample
        weight (see xqc/agents/common/loss_masking.py), applied after
        averaging loss0+loss1 over the Ta axis - None (default) reproduces
        plain-mean behavior exactly. Returns (loss, infos).

        target_mode: "approx" (default) or "exact" (both act_pred_0's
        target AND act_pred_1's training-input act_t are corrected to be
        centered at the mathematically correct intermediate point
        t_two_step*act instead of `act` - keeps act_pred_1 trained on
        inputs centered at whatever point act_pred_0 is pushed toward,
        avoiding a train/inference mismatch either way).

        squash_actions: act is mapped to the pre-tanh training coordinate
        first (see the field comment), so the losses and the mse_*
        diagnostics are all in that coordinate."""
        if self.squash_actions:
            act = safe_arctanh(act)
            if self.squash_offset is not None:
                act = act - jnp.asarray(self.squash_offset, dtype=act.dtype)
            if self.squash_scale is not None:
                act = act / jnp.asarray(self.squash_scale, dtype=act.dtype)
        bs = act.shape[0]
        s = jnp.zeros((bs,))
        t = jnp.full((bs,), self.t_two_step)
        act_0 = jnp.zeros_like(act)
        noise = jax.random.normal(rng, act.shape)

        if self.target_mode == "approx":
            target_0 = act
            act_t = act + (1 - self.t_two_step) * noise
        elif self.target_mode == "exact":
            target_0 = self.t_two_step * act
            act_t = self.t_two_step * act + (1 - self.t_two_step) * noise
        else:
            raise ValueError(f"Unknown target_mode {self.target_mode!r}, expected 'approx' or 'exact'")

        act_pred_0 = self.get_velocity(s, act_0, cond)
        act_pred_1 = self.get_velocity(t, act_t, cond)

        loss0 = _get_norm((act_pred_0 - target_0) / self.t_two_step, self.norm_type)
        loss1 = _get_norm((act_pred_1 - act) / (1 - self.t_two_step), self.norm_type)
        # loss0/loss1: (B, Ta). Mean over Ta first, THEN (masked-)mean over
        # B - equals the plain flat mean over (B, Ta) when mask is None
        # (mean-of-means over equal-sized groups), so this is a pure
        # generalization, not a behavior change.
        per_sample_loss = jnp.mean(loss0 + loss1, axis=1)
        loss = self.loss_scale * masked_mean(per_sample_loss, mask)

        # Diagnostic-only MSE (not part of the optimized loss, no gradient):
        # act_pred_0 is queried from the fixed noise point act_0=0 at
        # old-time 0, so the point it should land on (if the flow-map were
        # exact) is the true linear-path point at old-time t_two_step, i.e.
        # t_two_step*act - not `act` itself. act_pred_1 is queried at
        # old-time t_two_step (close to 1, the data side), so its
        # comparison point is `act` directly.
        mid_point_true = self.t_two_step * act
        mse_mid_point = jnp.mean((jax.lax.stop_gradient(act_pred_0) - mid_point_true) ** 2)
        mse_final_point = jnp.mean((jax.lax.stop_gradient(act_pred_1) - act) ** 2)

        return loss, {
            "mip_loss": loss,
            "mip_loss_step0": jnp.mean(loss0),
            "mip_loss_step1": jnp.mean(loss1),
            "mse_mid_point": mse_mid_point,
            "mse_final_point": mse_final_point,
        }

    def sample(self, rng: jax.random.PRNGKey, shape: Sequence[int], cond: jnp.ndarray) -> jnp.ndarray:
        """shape: (B, Ta, act_dim). cond: (B, To, cond_dim). Two
        deterministic forward passes through the flow-map network - no
        randomness, no iterative integration. `rng` is unused (accepted
        only for call-site uniformity - see module docstring)."""
        del rng
        bs = shape[0]
        s = jnp.zeros((bs,))
        t = jnp.full((bs,), self.t_two_step)

        act_0 = jnp.zeros(shape)
        act_pred_0 = self.get_velocity(s, act_0, cond)
        act_pred_1 = self.get_velocity(t, act_pred_0, cond)
        if not self.squash_actions:
            return act_pred_1
        y = act_pred_1
        if self.squash_scale is not None:
            y = y * jnp.asarray(self.squash_scale, dtype=y.dtype)
        if self.squash_offset is not None:
            y = y + jnp.asarray(self.squash_offset, dtype=y.dtype)
        return jnp.tanh(y)
