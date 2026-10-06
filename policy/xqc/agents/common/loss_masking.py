"""Shared masked-mean helper for the 8 IL network `.loss()` methods
(xqc/agents/actors/{dp,fm,mf,mip,mip_flow,nf,bc,af}) - lets each one accept
an optional per-sample `mask: (B,)` and reduce with it instead of a plain
mean, without duplicating the same guard/divide-by-zero logic 8 times.

Built for the `in_reference_region` ablation flag (xqc/replay_buffer/
chunked_qc_buffer.py's ChunkedBatch docstring, see conversation history):
`mask=None` (the default everywhere) reproduces today's behavior exactly;
passing `batch.in_reference_region` zeroes out the loss contribution of
samples whose Ta-length window needed more padding than the reference
Diffusion Policy's own SequenceSampler would have tolerated, as a controlled
ablation of the near-boundary-padding fix rather than a re-exclusion of
those samples from the batch.
"""
from typing import Optional

import jax.numpy as jnp


def masked_mean(per_sample_loss: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
    """per_sample_loss: (B,). mask: (B,) or None. None reduces to
    per_sample_loss.mean() exactly (no numerical difference from the
    unmasked path). A real mask must not be all-zero for a given batch
    (division by zero) - callers combine this with a large enough batch
    size that in_reference_region is essentially never all-zero in
    practice; not defended against here since it would silently mask a
    genuine bug (e.g. a misthreaded flag) instead of surfacing it."""
    if mask is None:
        return per_sample_loss.mean()
    return (per_sample_loss * mask).sum() / mask.sum()
