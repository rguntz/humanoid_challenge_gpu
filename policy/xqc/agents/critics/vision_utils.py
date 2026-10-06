"""Small helpers shared by every chunked_qc_buffer.py-based critic
(xqc/agents/critics/{mlp,xqc_style,transformer}) that supports camera
observations - vision trained from scratch on the critic, independent of
any actor (see conversation history)."""
import jax.numpy as jnp


def rescale_images(images):
    """Raw uint8 [0,255] (chunked_qc_buffer.py's own storage convention -
    see that file's own docstring) -> float32 [-1,1], matching encode_condition's own
    expected input range and xqc/agents/chunking/chunk_execution.py's
    identical rescale for the actor side."""
    return {k: (jnp.asarray(v, dtype=jnp.float32) / 255.0 * 2.0 - 1.0) for k, v in images.items()}
