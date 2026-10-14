"""MIP (Minimum Iterative Policy) learner: a deterministic, 2-step flow-map
BC policy over chunked actions.

Thin subclass of xqc/agents/chunking/chunked_learner.py's ChunkedPolicyLearner
- this __init__ exists only to keep conf/agent/mip.yaml's field names/
defaults unchanged, splitting them into the common kwargs ChunkedPolicyLearner
handles and MIP's own network-specific kwargs. See xqc/agents/actors/mip/mip.py's
module docstring for the port details.

BC pretraining: MIP.loss on demonstration chunks. Online RL fine-tuning
(only with a critic, agent.critic_name): ChunkedPolicyLearner's DDPG-style
update_actor (xqc/agents/chunking/actor.py), backpropagating the critic's Q
through MIP.sample's two deterministic forward passes, optionally plus
wasserstein_regularization * ||a_live - a_prior||^2 toward the frozen
BC-pretrained MIP. MIP provides no log_prob (a deterministic flow-map, not a
distribution), so nothing log-prob-based (KL, entropy) applies here.
"""
from typing import Any, Sequence

import numpy as np

from xqc.agents.chunking.chunked_learner import ChunkedPolicyLearner
from xqc.agents.actors.mip.mip import MIP


class MIPLearner(ChunkedPolicyLearner):
    def __init__(
        self,
        seed: int,
        observations: np.ndarray,
        actions: np.ndarray,
        offline_data_path: str,
        model_dim: int = 512,
        emb_dim: int = 128,
        kernel_size: int = 5,
        cond_predict_scale: bool = True,
        dim_mult: Sequence[int] = (1, 2, 2),
        n_groups: int = 8,
        t_two_step: float = 0.9,
        norm_type: str = "l2",
        loss_scale: float = 100.0,
        target_mode: str = "approx",
        # squash_actions/squash_scale: same as MFLearner (xqc/agents/actors/mf/
        # mf_learner.py) - see MIP.squash_actions. squash_scale (needs
        # squash_actions=True), computed per action dim on
        # z = safe_arctanh(normalized action) over the whole dataset:
        # "none" (s=1, mu=0); "rms": s_i = max(RMS(z_i), 1); "standardize":
        # mu_i = mean(z_i), s_i = max(std(z_i), 1). The floor of 1 leaves dims
        # arctanh doesn't inflate (and near-constant ones) unamplified.
        squash_actions: bool = False,
        squash_scale: str = "none",
        **kwargs: Any,
    ):
        if squash_scale not in ("none", "rms", "standardize"):
            raise ValueError(f"Unknown squash_scale {squash_scale!r}, expected 'none', 'rms' or 'standardize'")
        if squash_scale != "none" and not squash_actions:
            raise ValueError("squash_scale requires squash_actions=True")
        # Read by _data_dependent_network_kwargs, which super().__init__
        # calls once the dataset's action normalization is known.
        self.squash_scale = squash_scale
        super().__init__(
            seed=seed,
            observations=observations,
            actions=actions,
            offline_data_path=offline_data_path,
            network_cls=MIP,
            network_extra_kwargs=dict(
                model_dim=model_dim,
                emb_dim=emb_dim,
                kernel_size=kernel_size,
                cond_predict_scale=cond_predict_scale,
                dim_mult=tuple(dim_mult),
                n_groups=n_groups,
                t_two_step=t_two_step,
                norm_type=norm_type,
                loss_scale=loss_scale,
                target_mode=target_mode,
                squash_actions=squash_actions,
            ),
            **kwargs,
        )

    def _data_dependent_network_kwargs(self, network_extra_kwargs: dict, normalized_actions: np.ndarray) -> dict:
        if self.squash_scale == "none":
            return network_extra_kwargs
        z = np.arctanh(np.clip(normalized_actions, -0.999, 0.999))  # same clip as safe_arctanh
        if self.squash_scale == "rms":
            scale = np.maximum(np.sqrt(np.mean(z ** 2, axis=0)), 1.0)
            print(f"MIPLearner squash_scale='rms': scale={np.round(scale, 3).tolist()}")
            return {**network_extra_kwargs, "squash_scale": tuple(float(s) for s in scale)}
        offset = np.mean(z, axis=0)
        scale = np.maximum(np.std(z, axis=0), 1.0)
        print(f"MIPLearner squash_scale='standardize': offset={np.round(offset, 3).tolist()} "
              f"scale={np.round(scale, 3).tolist()}")
        return {
            **network_extra_kwargs,
            "squash_scale": tuple(float(s) for s in scale),
            "squash_offset": tuple(float(m) for m in offset),
        }
