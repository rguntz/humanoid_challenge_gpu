"""Conditional flow-matching (rectified-flow-style) policy over action
chunks: a multi-step ODE-integration generative model, trained via the
standard conditional flow-matching identity. Ported from
xqc/networks/mip/losses.py's flow_loss + xqc/networks/mip/samplers.py's
ode_sampler (the "loss_type='flow'" option in that reference repo), using
the linear interpolant only (I(t,x0,x1)=(1-t)*x0+t*x1, constant velocity
x1-x0 - the standard rectified-flow straight-line path; the reference's
"trig" interpolant option is not ported).

Two departures from that reference:
- The DiT/MLP backbone is replaced by xqc/agents/common/network.py's
  ChiUNet, shared with MIP/NF/MF - get_velocity(t, x, cond) calls
  net(x, t, t, cond) (both of ChiUNet's time args set to the same value),
  exactly matching the reference's FlowMap.get_velocity(t, xs, label) ->
  net(xs, t, t, label).
- The reference's `sample_mode` toggle (letting inference start from a fixed
  zero instead of Gaussian noise) is dropped entirely. This isn't a
  simplification for its own sake: the reference's own shipped default
  config (examples/configs/optimization/default.yaml) sets
  `loss_type: flow` + `sample_mode: zero` + `num_steps: 1` together - a real
  train/inference mismatch, since flow_loss trains unconditionally with
  act_0 ~ N(0,1) but that config's ode_sampler would start from a fixed
  zero point the model was never trained to be accurate at, taking a single
  Euler step over the entire [0,1] interval with no training objective that
  targets that regime (unlike MIP's own mip_loss, which explicitly trains
  for a one-shot jump from zero). This port always samples act_0 ~ N(0,1) at
  both training and inference, and defaults to a 10-step Euler schedule
  (irrelevant at training time; a deliberate speed/accuracy choice at
  inference time), so that mismatch can't happen.

Like MIP and MF, this is not a bijective flow (unlike NF's RealNVP) - there
is no log_prob, only a generated action.
"""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from xqc.agents.common.loss_masking import masked_mean
from xqc.agents.common.network import ChiUNet


class FlowMatching(nn.Module):
    """Conditional flow-matching policy over action chunks, ChiUNet-backed
    velocity net. act_dim, Ta, cond_dim, To: same meaning/shapes as
    MeanFlow's own fields - condition is (B, To, cond_dim), the
    obs_encoder's output, and actions/velocities are (B, Ta, act_dim)."""

    act_dim: int
    Ta: int
    cond_dim: int
    To: int
    num_steps: int = 10
    model_dim: int = 256
    emb_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    dim_mult: Sequence[int] = (1, 2, 2)
    n_groups: int = 8

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
            # get_velocity above always calls net with s=t (see its own
            # docstring) - the dual s/t embedding split is pure redundancy
            # here, not two independent signals - see
            # xqc/agents/actors/mip/mip_learner.py's matching comment for the full
            # derivation (identical pattern). Was False - see git history.
            use_full_width_t_embedding=True,
            # Repo-wide "shared prediction backbone" decision - see
            # xqc/agents/common/obs_encoder.py's IdentityObsEncoder
            # docstring and xqc/agents/actors/mip/mip_learner.py's matching
            # comments. Both were True - see git history.
            zero_init_final_conv=False,
            project_condition=False,
        )

    def get_velocity(self, t: jnp.ndarray, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """t: (B,) or (B,1,1). x: (B, Ta, act_dim). cond: (B, To, cond_dim).
        Returns the instantaneous velocity field, (B, Ta, act_dim). Matches
        the reference's FlowMap.get_velocity: both of ChiUNet's time args
        get the same value t."""
        bz = x.shape[0]
        t_flat = t.reshape(bz)
        velocity, _ = self.net(x, t_flat, t_flat, cond)
        return velocity

    def loss(self, act: jnp.ndarray, cond: jnp.ndarray, rng: jax.random.PRNGKey, mask: Optional[jnp.ndarray] = None):
        """act: (B, Ta, act_dim) demonstration actions, normalized to
        (-1, 1). cond: (B, To, cond_dim). mask: optional (B,) per-sample
        weight (see xqc/agents/common/loss_masking.py) - None (default)
        reproduces plain-mean behavior exactly. Returns (loss, infos)."""
        bz = act.shape[0]
        rng_t, rng_noise = jax.random.split(rng)
        t = jax.random.uniform(rng_t, (bz, 1, 1))
        act_0 = jax.random.normal(rng_noise, act.shape)
        act_1 = act

        act_t = (1.0 - t) * act_0 + t * act_1
        act_t_dot = act_1 - act_0  # exact d/dt of the linear interpolant, constant in t

        b_t = self.get_velocity(t, act_t, cond)

        sq_err = (b_t - act_t_dot) ** 2
        # mean (not sum) over the (Ta, act_dim) axes - matches DP's own fix
        # for the identical bug (xqc/agents/actors/dp/diffusion_policy.py's loss()):
        # summing here inflates the logged/trained loss by a constant
        # Ta*act_dim factor for no reason - was jnp.sum, see git history.
        per_sample_loss = jnp.mean(sq_err, axis=(1, 2))
        loss = masked_mean(per_sample_loss, mask)
        return loss, {"fm_loss": loss}

    def sample(self, rng: jax.random.PRNGKey, shape: Sequence[int], cond: jnp.ndarray) -> jnp.ndarray:
        """shape: (B, Ta, act_dim). cond: (B, To, cond_dim). Returns the
        generated action, normalized to (-1, 1). Forward-Euler integration
        of the learned ODE from t=0 (act_0 ~ N(0,1)) to t=1, over
        `num_steps` uniformly-spaced steps - matches the reference's
        ode_sampler exactly, modulo always starting from Gaussian noise
        (see module docstring)."""
        bz = shape[0]
        t_schedule = jnp.linspace(0.0, 1.0, self.num_steps + 1)
        x = jax.random.normal(rng, shape)

        # Plain Python loop, not jax.lax.fori_loop: num_steps is a static
        # field (unrolled at trace time), and Flax submodule calls (self.net,
        # via get_velocity) can't be traced through raw JAX control-flow
        # primitives - see xqc/agents/actors/mf/mean_flow.py's sample() for the
        # same issue hit and fixed there.
        for i in range(self.num_steps):
            s = jnp.full((bz,), t_schedule[i])
            t = jnp.full((bz,), t_schedule[i + 1])
            b_s = self.get_velocity(s, x, cond)
            x = x + b_s * (t - s).reshape(bz, 1, 1)
        return x
