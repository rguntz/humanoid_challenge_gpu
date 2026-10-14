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
        **kwargs: Any,
    ):
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
            ),
            **kwargs,
        )
