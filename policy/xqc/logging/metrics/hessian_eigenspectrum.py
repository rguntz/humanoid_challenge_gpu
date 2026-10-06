# Adapted from https://github.com/google/spectral-density

from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, jvp
from jax.flatten_util import ravel_pytree


def hvp(loss, params, batch, v):
    """Computes the hessian vector product Hv.

    This implementation uses forward-over-reverse mode for computing the hvp.

    Args:
      loss: function computing the loss with signature
        loss(params, batch).
      params: pytree for the parameters of the model.
      batch:  A batch of data. Any format is fine as long as it is a valid input
        to loss(params, batch).
      v: pytree of the same structure as params.

    Returns:
      hvp: array of shape [num_params] equal to Hv where H is the hessian.
    """

    def loss_fn(p):
        return loss(critic_params=p, batch=batch)[0]

    return jvp(grad(loss_fn), [params], [v])[1]


def get_hvp_fn(loss, params, batches, keys):
    """Generates a function mapping (params, v) -> Hv where H is the hessian.

    This function will batch the inputs and targets to be fed into loss. The
    hessian will be computed over all points xs, ys, potentially batching if
    needed. This function is intended to be used in cases where xs, ys are too
    large to run on a single pass. This function should not be jit compiled. If
    the computation is small enough to do all batches in memory then one can just
    do the following:

    @jit
    def jitted_hvp(params, v):
      return hvp(loss, params, all_data, v)

    Args:
      loss: scalar valued loss function with signature loss(params, batch).
        Assumes the loss computes a sum over all data points. If the loss computes
        the mean, results may be slightly off in cases where batch sizes are not
        uniform.
      params: params of the model, these will be flatten and concatentated into a
        single vector. Any pytree is valid
      batches: A generator yielding batches to be fed into loss. Must support the
        API "for b in batches(): ". batches() must yield a single epoch of data,
        it should also yield the same epoch of data everytime it is called.

    Returns:
      hvp: A function mapping (params, v) -> Hv. H is the Hessian of the loss
        with respect to the model parameters (it is a num_params by num_params
        matrix). v will be a flat vector of shape [num_params]. params will be
        the PyTree containing the model parameters (so calling ravel_pytree on
        parameters). The function signature is hvp(params, v).
      unravel: Maps v back to the form reprented as params.
      num_params: Total number of parameters in params (int).
    """

    flat_params, unravel = ravel_pytree(params)  # type: ignore

    @jit
    def hvp_fn(params, v):
        """Maps a vector v to Hv, where H is the hessian.

        Args:
        params: pytree of model parameters.
        v: array of size [num_params]
        Returns:
        hessian_vp_flat: array of size [num_params] equal to Hv.
        """
        v = unravel(v)  # convert v to the param tree structure
        hessian_vp = jax.tree.map(lambda p: jnp.zeros_like(p), params)  # ['params']

        num_batches = batches.next_observations.shape[0]

        def hvp_batch(hessian_vp, batch_and_key):
            batch, key = batch_and_key
            partial_vp = hvp(partial(loss, key=key), params, batch, v)  # ['params']
            hessian_vp = jax.tree.map(lambda x, y: x + y, hessian_vp, partial_vp)
            return hessian_vp, None

        hessian_vp, _ = jax.lax.scan(hvp_batch, hessian_vp, (batches, keys))

        hessian_vp_flat, _ = ravel_pytree(hessian_vp)
        hessian_vp_flat /= num_batches
        return hessian_vp_flat

    return hvp_fn, unravel, flat_params.shape[0]


def lanczos_alg(matrix_vector_product, dim, order, rng_key):
    """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

    This function applies Lanczos algorithm of a given order.  This function
    does full reorthogonalization.

    WARNING: This function may take a long time to jit compile (e.g. ~3min for
    order 90 and dim 1e7).

    Args:
      matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
        Input/Output must be of shape [dim].
      dim: Matrix H is [dim, dim].
      order: An integer corresponding to the number of Lanczos steps to take.
      rng_key: The jax PRNG key.

    Returns:
      tridiag: A tridiagonal matrix of size (order, order).
      vecs: A numpy array of size (order, dim) corresponding to the Lanczos
        vectors.
    """

    tridiag = jnp.zeros((order, order))
    vecs = jnp.zeros((order, dim))

    init_vec = jax.random.normal(rng_key, shape=(dim,))
    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0].set(init_vec)

    beta = 0

    for i in range(order):
        v = vecs[i, :].reshape((dim))
        if i == 0:
            v_old = 0.0
        else:
            v_old = vecs[i - 1, :].reshape((dim))

        w = matrix_vector_product(v)
        assert (
            w.shape[0] == dim and len(w.shape) == 1
        ), "Output of matrix_vector_product(v) must be of shape [dim]."
        w = w - beta * v_old

        alpha = jnp.dot(w, v)  # type: ignore
        tridiag = tridiag.at[i, i].set(alpha)
        w = w - alpha * v

        # Full Reorthogonalization
        for j in range(i):
            tau = vecs[j, :].reshape((dim))
            coeff = jnp.dot(w, tau)  # type: ignore
            w += -coeff * tau

        beta = jnp.linalg.norm(w)

        if i + 1 < order:
            tridiag = tridiag.at[i, i + 1].set(beta)
            tridiag = tridiag.at[i + 1, i].set(beta)
            vecs = vecs.at[i + 1].set(w / beta)
    return (tridiag, vecs)


def tridiag_to_eigv(tridiag_list):
    """Preprocess the tridiagonal matrices for density estimation.

    Args:
      tridiag_list: Array of shape [num_draws, order, order] List of the
        tridiagonal matrices computed from running num_draws independent runs
        of lanczos. The output of this function can be fed directly into
        eigv_to_density.

    Returns:
      eig_vals: Array of shape [num_draws, order]. The eigenvalues of the
        tridiagonal matricies.
      all_weights: Array of shape [num_draws, order]. The weights associated with
        each eigenvalue. These weights are to be used in the kernel density
        estimate.
    """
    # Calculating the node / weights from Jacobi matrices.
    num_draws = len(tridiag_list)
    num_lanczos = tridiag_list[0].shape[0]
    eig_vals = np.zeros((num_draws, num_lanczos))
    all_weights = np.zeros((num_draws, num_lanczos))
    for i in range(num_draws):
        nodes, evecs = np.linalg.eigh(tridiag_list[i])
        index = np.argsort(nodes)
        nodes = nodes[index]
        evecs = evecs[:, index]
        eig_vals[i, :] = nodes
        all_weights[i, :] = evecs[0] ** 2
    return eig_vals, all_weights


def compute_hessian_eigenspectrum(
    agent, replay_buffer, batch_size, num_seeds, n_batches=50, order=90
):
    condition_number_logs = defaultdict(list)

    def model_for_seed(seed, model):
        def params_for_seed(seed, params):
            return jax.tree.map(lambda p: p[seed], params)

        return model.replace(
            params=params_for_seed(seed, model.params),
            batch_stats=(
                params_for_seed(seed, model.batch_stats) if model.batch_stats else None
            ),
        )

    all_batches = replay_buffer.sample_parallel_multibatch(batch_size, n_batches)

    for j in range(num_seeds):
        _rng, loss_key, lanczos_key = jax.random.split(agent.rng[j], 3)
        agent.rng = agent.rng.at[j].set(_rng)
        critic_j = model_for_seed(j, agent.critic)

        loss_fn = partial(
            agent.critic.loss_fn,
            critic_batch_stats=critic_j.batch_stats,
            actor=model_for_seed(j, agent.actor),
            critic=critic_j,
            target_critic=model_for_seed(j, agent.target_critic),
            temperature=model_for_seed(j, agent.temperature),
        )

        batches_j = jax.tree.map(lambda x: x[j], all_batches)
        hvp, unravel, num_params = get_hvp_fn(
            loss_fn, critic_j.params, batches_j, jax.random.split(loss_key, n_batches)
        )

        tridiag, vecs = lanczos_alg(
            lambda v: hvp(critic_j.params, v) / n_batches,
            num_params,
            order,
            lanczos_key,
        )

        [eig_vals], _ = tridiag_to_eigv([tridiag])

        condition_number_logs["critic_eigvals"].append(list(eig_vals))

    return condition_number_logs
