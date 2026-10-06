"""Diffusion Policy (Chi et al.) over action chunks: standard DDPM training
(predict the injected noise, uniform random training timestep) with a choice
of DDPM (full-chain, stochastic) or DDIM (few-step, deterministic) inference,
ported from the reference implementation at /home/rguntz/workspace/diffusion_policy
(diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py and
diffusion_policy/model/diffusion/conditional_unet1d.py).

Like xqc/agents/actors/fm/flow_matching.py, the only real port is the loss/sampler -
the reference's own ConditionalUnet1D backbone is replaced by
xqc/agents/common/network.py's ChiUNet, shared with MIP/MF/FM. ChiUNet takes
two time-conditioning scalars (s, t); the reference's UNet only takes one
(the diffusion timestep). Following mip/mip_flow's own pattern of holding one
slot fixed when a method only needs a single time value, `s` is pinned to 0
and the diffusion timestep goes entirely into `t`.

Reference defaults ported as this module's defaults (image_pusht config):
beta_schedule="squaredcos_cap_v2" (cosine, Nichol & Dhariwal), beta_start=1e-4,
beta_end=0.02, num_train_timesteps=100, clip_sample=True, prediction_type
fixed to "epsilon" (the reference's default and this port's only supported
mode - it also supports "sample" but nothing in this codebase's other agents
needs that variant, so it isn't ported). variance_type is fixed to
"fixed_small" (the reference's own default), i.e. the DDPM reverse step uses
the posterior variance beta_tilde_t = beta_t * (1-alpha_bar_{t-1}) /
(1-alpha_bar_t) - no learned variance head.

Two inference modes (see `sampler` field):
- "ddpm": the reference's plain DDPMScheduler - runs the full
  `num_train_timesteps`-step ancestral chain (stochastic, one denoising step
  per training timestep, no respacing).
- "ddim": the reference's DDIMScheduler variant - `num_inference_steps`
  evenly-spaced steps (leading spacing, matching diffusers' default), eta=0
  (deterministic), matching the reference's own DDIM config (8 steps by
  default).

Like FM and MF, this is not a bijective flow/exact-density model - there is
no log_prob, only a generated action via iterative denoising from Gaussian
noise.
"""
import math
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from xqc.agents.common.loss_masking import masked_mean
from xqc.agents.common.network import ChiUNet


def _cosine_betas(num_train_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """squaredcos_cap_v2, matching diffusers' betas_for_alpha_bar /
    the reference's default beta_schedule."""
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float64)


def _make_betas(num_train_timesteps: int, beta_schedule: str, beta_start: float, beta_end: float) -> np.ndarray:
    if beta_schedule == "squaredcos_cap_v2":
        return _cosine_betas(num_train_timesteps)
    elif beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
    raise ValueError(f"Unknown beta_schedule {beta_schedule!r}, expected 'squaredcos_cap_v2' or 'linear'")


class DiffusionPolicy(nn.Module):
    """DDPM/DDIM policy over action chunks, ChiUNet-backed noise predictor.
    act_dim, Ta, cond_dim, To: same meaning/shapes as FlowMatching's own
    fields - condition is (B, To, cond_dim), the obs_encoder's output, and
    actions/noise are (B, Ta, act_dim)."""

    act_dim: int
    Ta: int
    cond_dim: int
    To: int
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    clip_sample: bool = True
    sampler: str = "ddpm"  # "ddpm" (full-chain, stochastic) or "ddim" (num_inference_steps, deterministic)
    num_inference_steps: int = 100  # only used when sampler == "ddim" (ddpm always runs the full chain)
    model_dim: int = 256
    emb_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    dim_mult: Sequence[int] = (1, 2, 2)
    n_groups: int = 8

    def setup(self):
        betas = _make_betas(self.num_train_timesteps, self.beta_schedule, self.beta_start, self.beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self._alphas_cumprod = jnp.asarray(alphas_cumprod, dtype=jnp.float32)
        # Plain Python/numpy arrays (not jnp), used only to build the
        # unrolled sampling loop below - concrete at trace time since
        # num_train_timesteps/num_inference_steps are static Flax fields.
        self._betas_np = betas
        self._alphas_cumprod_np = alphas_cumprod

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
            # `s` is pinned to 0 below (predict_noise) and carries no signal
            # for DP - give the diffusion timestep the full embedding width
            # instead of splitting it with an always-constant emb_s, matching
            # the reference's diffusion_step_encoder exactly. See ChiUNet's
            # own field docstring for the full derivation.
            use_full_width_t_embedding=True,
            # The reference's ConditionalUnet1D does not zero-init its
            # final_conv - see ChiUNet's own field docstring for the full
            # derivation. DP-only; MIP/NF/MF keep the default (True).
            zero_init_final_conv=False,
            # The reference has no learned projection between the raw
            # obs-encoder output and the FiLM conditioning - see ChiUNet's
            # own project_condition field docstring for the full derivation.
            # DP-only; MIP/NF/MF keep the default (True).
            project_condition=False,
        )

    def predict_noise(self, x: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """x: (B, Ta, act_dim). t: (B,) integer diffusion timesteps (0 to
        num_train_timesteps-1). cond: (B, To, cond_dim). Returns the
        predicted noise, (B, Ta, act_dim). ChiUNet's `s` slot is unused here
        (pinned to 0) - see module docstring."""
        bz = x.shape[0]
        s0 = jnp.zeros((bz,))
        eps, _ = self.net(x, s0, t.astype(jnp.float32).reshape(bz), cond)
        return eps

    def loss(self, act: jnp.ndarray, cond: jnp.ndarray, rng: jax.random.PRNGKey, mask: Optional[jnp.ndarray] = None):
        """act: (B, Ta, act_dim) demonstration actions, normalized to
        (-1, 1). cond: (B, To, cond_dim). mask: optional (B,) per-sample
        weight (see xqc/agents/common/loss_masking.py) - None (default)
        reproduces plain-mean behavior exactly. Returns (loss, infos).
        Standard DDPM training: uniform random timestep per sample, predict
        the injected noise, plain MSE - matches the reference's
        compute_loss exactly (mask-free here since obs_as_global_cond=True
        means the whole action chunk is diffused/supervised, see module
        docstring)."""
        bz = act.shape[0]
        rng_t, rng_noise = jax.random.split(rng)
        t = jax.random.randint(rng_t, (bz,), 0, self.num_train_timesteps)
        noise = jax.random.normal(rng_noise, act.shape)

        alpha_bar_t = self._alphas_cumprod[t].reshape(bz, 1, 1)
        x_t = jnp.sqrt(alpha_bar_t) * act + jnp.sqrt(1.0 - alpha_bar_t) * noise

        eps_pred = self.predict_noise(x_t, t, cond)

        sq_err = (eps_pred - noise) ** 2
        per_sample_loss = jnp.mean(sq_err, axis=(1, 2))
        loss = masked_mean(per_sample_loss, mask)
        return loss, {"dp_loss": loss}

    def _pred_x0(self, x_t: jnp.ndarray, eps: jnp.ndarray, alpha_bar_t: float) -> jnp.ndarray:
        x0 = (x_t - jnp.sqrt(1.0 - alpha_bar_t) * eps) / jnp.sqrt(alpha_bar_t)
        if self.clip_sample:
            x0 = jnp.clip(x0, -1.0, 1.0)
        return x0

    def _ddpm_sample(self, rng: jax.random.PRNGKey, shape: Sequence[int], cond: jnp.ndarray) -> jnp.ndarray:
        """Full num_train_timesteps-step ancestral DDPM chain, fixed_small
        posterior variance - matches diffusers' DDPMScheduler.step exactly.
        Plain Python loop (not jax.lax.fori_loop): the timestep count is a
        static field and Flax submodule calls (self.net, via predict_noise)
        can't be traced through raw JAX control-flow primitives - same
        constraint noted in xqc/agents/actors/mf/mean_flow.py's sample()."""
        bz = shape[0]
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, shape)

        for t in range(self.num_train_timesteps - 1, -1, -1):
            t_b = jnp.full((bz,), t, dtype=jnp.int32)
            eps = self.predict_noise(x, t_b, cond)

            alpha_bar_t = self._alphas_cumprod_np[t]
            alpha_bar_prev = self._alphas_cumprod_np[t - 1] if t > 0 else 1.0
            beta_t = self._betas_np[t]
            alpha_t = 1.0 - beta_t

            x0 = self._pred_x0(x, eps, alpha_bar_t)
            mean = (
                (math.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)) * x0
                + (math.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * x
            )
            if t > 0:
                variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                rng, step_rng = jax.random.split(rng)
                x = mean + math.sqrt(variance) * jax.random.normal(step_rng, shape)
            else:
                x = mean
        return x

    def _ddim_timesteps(self) -> np.ndarray:
        """Evenly-spaced descending timestep schedule ("leading" spacing),
        matching diffusers' DDIMScheduler.set_timesteps default."""
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round().astype(np.int64)
        return timesteps[::-1].copy()

    def _ddim_sample(self, rng: jax.random.PRNGKey, shape: Sequence[int], cond: jnp.ndarray) -> jnp.ndarray:
        """num_inference_steps-step deterministic DDIM chain (eta=0) -
        matches diffusers' DDIMScheduler.step(eta=0) exactly."""
        bz = shape[0]
        x = jax.random.normal(rng, shape)
        timesteps = self._ddim_timesteps()
        step_ratio = self.num_train_timesteps // self.num_inference_steps

        for t in timesteps:
            t = int(t)
            prev_t = t - step_ratio
            t_b = jnp.full((bz,), t, dtype=jnp.int32)
            eps = self.predict_noise(x, t_b, cond)

            alpha_bar_t = self._alphas_cumprod_np[t]
            alpha_bar_prev = self._alphas_cumprod_np[prev_t] if prev_t >= 0 else 1.0

            x0 = self._pred_x0(x, eps, alpha_bar_t)
            dir_xt = math.sqrt(1.0 - alpha_bar_prev) * eps
            x = math.sqrt(alpha_bar_prev) * x0 + dir_xt
        return x

    def sample(self, rng: jax.random.PRNGKey, shape: Sequence[int], cond: jnp.ndarray) -> jnp.ndarray:
        """shape: (B, Ta, act_dim). cond: (B, To, cond_dim). Returns the
        generated action, normalized to (-1, 1)."""
        if self.sampler == "ddim":
            return self._ddim_sample(rng, shape, cond)
        elif self.sampler == "ddpm":
            return self._ddpm_sample(rng, shape, cond)
        raise ValueError(f"Unknown sampler {self.sampler!r}, expected 'ddpm' or 'ddim'")
