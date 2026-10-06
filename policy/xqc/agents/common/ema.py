"""Shared EMA (exponential moving average) parameter tracking, used by every
ChiUNet-backed learner (dp/fm/mf/mf_2_step/mf_invertible/nf/mip/mip_flow/
mip_flow_injection) for its self.params_ema. Was defined separately (and
only) in dp_learner.py, with every other learner's own EMA hardcoded to a
flat ema_rate multiplier for the whole run - see git history for the parity
gap this closes (repo-wide "benchmark everything against DP on equal
optimizer/EMA footing" decision, not a reference-fidelity port for its own
sake - much-ado-about-noising's own agent.py uses a flat rate too, so this
is a deliberate repo-wide standardization, matching xqc/agents/common/
vision.py's crop_images and xqc/agents/*/*_learner.py's cosine_warmup
lr_schedule in spirit)."""
import jax
import jax.numpy as jnp


def ema_decay(step: int, ema_schedule: str, ema_rate: float, ema_power: float, ema_max_value: float) -> float:
    """"fixed": a constant ema_rate for the whole run. "warmup": the
    reference Diffusion Policy implementation's own diffusers-style
    EMAModel formula (inv_gamma=1.0, update_after_step=0 folded in),
    asymptoting to ema_max_value as step grows - see conf/agent/dp.yaml's
    "ema_schedule" docstring for the full derivation.

    `step` here is `self.state.step` read *after* that update's gradient
    step (dp_learner.py), i.e. 1 on the very first EMA update, 2 on the
    second, etc. The reference's EMAModel.get_decay(optimization_step)
    reads its own counter *before* incrementing it (0 on the first call, 1
    on the second, ...) and additionally forces decay=0.0 whenever
    `optimization_step - update_after_step - 1 <= 0` - with
    update_after_step=0 that's optimization_step in {0, 1}, i.e. the first
    two EMA updates are a hardcoded copy (decay=0.0), not part of the
    1-(1+k)^-power ramp. `k = step - 1` re-derives that same reference
    counter from xqc's 1-indexed `step`, so this reproduces the reference
    bit-for-bit (including that two-update hard-zero grace period), not
    just the same asymptote."""
    if ema_schedule == "fixed":
        return ema_rate
    if ema_schedule == "warmup":
        k = step - 1
        if k <= 1:
            return 0.0
        return float(min(ema_max_value, max(0.0, 1.0 - k ** (-ema_power))))
    raise ValueError(f"Unknown ema_schedule {ema_schedule!r}, expected 'fixed' or 'warmup'")


@jax.jit
def ema_update(params_ema, params, rate):
    return jax.tree_util.tree_map(lambda e, p: rate * e + (1.0 - rate) * p, params_ema, params)
