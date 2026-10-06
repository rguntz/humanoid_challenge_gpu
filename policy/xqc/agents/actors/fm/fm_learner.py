"""FM (Flow Matching) learner: a multi-step conditional flow-matching policy
over chunked actions.

Thin subclass of xqc/agents/chunking/chunked_learner.py's ChunkedPolicyLearner
- this __init__ exists only to keep conf/agent/fm.yaml's field names/
defaults unchanged, splitting them into the common kwargs ChunkedPolicyLearner
handles and FlowMatching's own network-specific kwargs. See
xqc/agents/actors/fm/flow_matching.py's module docstring for the port details.

Only BC pretraining (FlowMatching.loss on demonstration chunks) is
implemented. Online RL fine-tuning is deliberately NOT implemented - see
ChunkedPolicyLearner. Like MF/MIP, FlowMatching provides no log_prob at all
(it's a generative ODE solve, not a distribution with a tractable density),
so RL fine-tuning would need a genuinely different technique before it could
use anything log-prob-based here.
"""
from typing import Any, Sequence

import numpy as np

from xqc.agents.chunking.chunked_learner import ChunkedPolicyLearner
from xqc.agents.actors.fm.flow_matching import FlowMatching


class FMLearner(ChunkedPolicyLearner):
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
        num_steps: int = 10,
        **kwargs: Any,
    ):
        super().__init__(
            seed=seed,
            observations=observations,
            actions=actions,
            offline_data_path=offline_data_path,
            network_cls=FlowMatching,
            network_extra_kwargs=dict(
                model_dim=model_dim,
                emb_dim=emb_dim,
                kernel_size=kernel_size,
                cond_predict_scale=cond_predict_scale,
                dim_mult=tuple(dim_mult),
                n_groups=n_groups,
                num_steps=num_steps,
            ),
            **kwargs,
        )
