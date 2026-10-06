"""BC (Behavior Cloning) learner: a plain MLP direct-regression policy over
chunked (or single-step) actions.

Thin subclass of xqc/agents/chunking/chunked_learner.py's ChunkedPolicyLearner
- this __init__ exists only to keep conf/agent/bc.yaml's field names/
defaults unchanged, splitting them into the common kwargs ChunkedPolicyLearner
handles and MLPPolicy's own network-specific kwargs. See
xqc/agents/actors/bc/mlp_policy.py's module docstring for the point of this
deliberately minimal baseline (shares the same data pipeline as DP/FM/MF/
MIP/MIP-Flow, none of their ChiUNet/noise/time-conditioning machinery).

Only BC pretraining (MLPPolicy.loss on demonstration chunks) is implemented.
Online RL fine-tuning is deliberately NOT implemented - see
ChunkedPolicyLearner. No log_prob is provided (a point estimate, not a
distribution).
"""
from typing import Any, Sequence

import numpy as np

from xqc.agents.chunking.chunked_learner import ChunkedPolicyLearner
from xqc.agents.actors.bc.mlp_policy import MLPPolicy


class BCLearner(ChunkedPolicyLearner):
    def __init__(
        self,
        seed: int,
        observations: np.ndarray,
        actions: np.ndarray,
        offline_data_path: str,
        hidden_dims: Sequence[int] = (256, 256),
        **kwargs: Any,
    ):
        super().__init__(
            seed=seed,
            observations=observations,
            actions=actions,
            offline_data_path=offline_data_path,
            network_cls=MLPPolicy,
            network_extra_kwargs=dict(hidden_dims=tuple(hidden_dims)),
            **kwargs,
        )
