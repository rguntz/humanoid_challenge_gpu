import os
from typing import Any, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import flax.traverse_util
import jax
import jax.numpy as jnp
import optax

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None
    batch_stats: Optional[Params] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    batch_stats: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    init_norm: Optional[dict] = None
    loss_fn: Optional[Any] = flax.struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        loss_fn: Optional[Any] = None,
        method: Optional[str] = None,
    ) -> "Model":
        # method: passed through to .init() - needed for a module whose
        # params are built via a non-__call__ method (e.g. a prediction
        # network initialized through its own `.loss` method - see
        # xqc/agents/chunking/chunked_learner.py's own network construction).
        variables = model_def.init(*inputs, method=method)

        # .get(..., {}) not [...]: a genuinely parameter-free module (e.g.
        # IdentityObsEncoder) makes Flax's own .init() omit the "params" key
        # entirely rather than return an empty dict under it.
        params = variables.get("params", {})

        init_norm_dict = {}
        p_flat = flax.traverse_util.flatten_dict(params)
        for path, param in p_flat.items():
            if path[0] == "log_temp":
                continue
            if path[-1] == "kernel":
                init_norm_dict[path] = jnp.linalg.norm(
                    param, axis=(-1, -2), keepdims=True
                )

        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            batch_stats=variables.get("batch_stats", None),
            tx=tx,
            opt_state=tx.init(params) if tx is not None else None,
            init_norm=init_norm_dict,
            loss_fn=loss_fn,
        )

    def __call__(self, *args, **kwargs):
        return self.apply(self.params, *args, batch_stats=self.batch_stats, **kwargs)

    def apply(self, params, *args, batch_stats=None, **kwargs):
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats
        return self.apply_fn.apply(variables, *args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params, self.batch_stats)
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return (
            self.replace(
                step=self.step + 1, params=new_params, opt_state=new_opt_state
            ),
            grads,
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    SaveState(
                        params=self.params,
                        opt_state=self.opt_state,
                        batch_stats=self.batch_stats,
                    )
                )
            )

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(
                    params=self.params,
                    opt_state=self.opt_state,
                    batch_stats=self.batch_stats,
                ),
                contents,
            )
            return self.replace(
                params=saved_state.params,
                opt_state=saved_state.opt_state,
                batch_stats=saved_state.batch_stats,
            )


def norm_dense_layer(params, path, norm_bias=True):

    kernel = params[path + "/kernel"]
    bias = params.get(path + "/bias", None)

    # if bias is present, normalize kernel and bias together
    if norm_bias and bias is not None:
        w = jnp.concatenate([kernel, jnp.expand_dims(bias, -2)], axis=-2)
    else:
        w = kernel

    norm = jnp.linalg.norm(w, axis=-2, keepdims=True)

    params[path + "/kernel"] = kernel / norm
    if norm_bias and bias is not None:
        params[path + "/bias"] = bias / norm.squeeze(-2)

    return params


def norm_network(
    model: Model,
    normalize_last_layer: bool = True,
):
    params_flat = flax.traverse_util.flatten_dict(model.params, sep="/")

    for path in sorted({"/".join(k.split("/")[:-1]) for k in params_flat}):

        # Do not normalize Hetstat Layers
        if 'HetStat' in path:
            continue

        # Normalize all hidden Dense layers, i.e. under 'MLP_0'
        elif "MLP_0" in path and "Dense" in path:
            params_flat = norm_dense_layer(params_flat, path, norm_bias=True)

        # Normalize last Dense layer, i.e. under 'predictor'
        elif "predictor" in path and normalize_last_layer:
            params_flat = norm_dense_layer(params_flat, path, norm_bias=False)

    return model.replace(params=flax.traverse_util.unflatten_dict(params_flat, sep="/"))
