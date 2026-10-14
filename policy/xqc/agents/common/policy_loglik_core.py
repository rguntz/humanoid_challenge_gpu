"""Core of the policy log-likelihood diagnostic, kept dependency-free (jax
only) so stationarity/compare_stationarity_mf.py can load this exact file
and validate the very lines xqc/agents/chunking/actor.py's policy_loglik
runs in training (the toy has its own environment and can't import the
training stack).

Change of variables for a policy that samples a = tanh(y), y = pre_tanh_fn(e, c),
e ~ N(0, I):
    log p(a | c) = log N(e) - log|det dy/de| - sum log(1 - a^2)
The Jacobian is taken only up to the pre-tanh value y; the tanh term is
added analytically from y in a numerically stable form. Taking the Jacobian
through the tanh instead gives the same value mathematically, but in float32
an action that rounds to exactly +-1 (y above ~9) gets a zero tanh slope and
an infinite log-likelihood - computing log(1 - tanh(y)^2) from y avoids that.
"""
import jax
import jax.numpy as jnp


def log1m_tanh_sq(y: jnp.ndarray) -> jnp.ndarray:
    """log(1 - tanh(y)^2) = 2 * (log 2 - y - softplus(-2y)), computed from y
    (never from the rounded tanh(y)), so it stays finite for large |y|."""
    return 2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y))


def split_tanh_loglik(pre_tanh_fn, e: jnp.ndarray, c: jnp.ndarray, squash: bool = True):
    """e: (N, *event) noise; c: (N, ...) conditioning; pre_tanh_fn(e_i, c_i)
    -> y_i, same shape as e_i. Returns (loglik (N,), sign (N,), logabsdet
    (N,), y (N, *event)). squash=False: the action is y itself (no tanh
    term)."""
    n = e.shape[0]
    d = 1
    for k in e.shape[1:]:
        d *= k
    y = jax.vmap(pre_tanh_fn)(e, c)
    jac = jax.vmap(jax.jacfwd(pre_tanh_fn))(e, c).reshape(n, d, d)
    sign, logabsdet = jnp.linalg.slogdet(jac)
    event_axes = tuple(range(1, e.ndim))
    log_noise = -0.5 * jnp.sum(e ** 2, axis=event_axes) - 0.5 * d * jnp.log(2 * jnp.pi)
    loglik = log_noise - logabsdet
    if squash:
        loglik = loglik - jnp.sum(log1m_tanh_sq(y), axis=event_axes)
    return loglik, sign, logabsdet, y
