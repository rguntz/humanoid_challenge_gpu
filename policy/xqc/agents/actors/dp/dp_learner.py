"""DP (Diffusion Policy) learner: a DDPM/DDIM policy over chunked actions.

Thin subclass of xqc/agents/chunking/chunked_learner.py's ChunkedPolicyLearner
- this __init__ exists only to keep conf/agent/dp.yaml's field names/
defaults unchanged, splitting them into the common kwargs ChunkedPolicyLearner
handles and DiffusionPolicy's own network-specific kwargs. See
xqc/agents/actors/dp/diffusion_policy.py's module docstring for the port details
and the reference implementation this mirrors (DP is the fidelity reference
the whole ChunkedPolicyLearner/dataset unification was built against - see
chunked_learner.py's own module docstring).

Only BC pretraining (DiffusionPolicy.loss on demonstration chunks) is
implemented. Online RL fine-tuning is deliberately NOT implemented -
`update(..., pre_train=False)` and `sample_actions_with_log_probs` both raise
NotImplementedError on purpose (see ChunkedPolicyLearner). DiffusionPolicy
provides no log_prob at all (it's a generative denoising solve, not a
distribution with a tractable density), so RL fine-tuning would need a
genuinely different technique before it could use anything log-prob-based
here.
"""
from typing import Any, Sequence

import numpy as np

from xqc.agents.chunking.chunked_learner import ChunkedPolicyLearner
from xqc.agents.actors.dp.diffusion_policy import DiffusionPolicy


class DPLearner(ChunkedPolicyLearner):
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
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        sampler: str = "ddpm",
        num_inference_steps: int = 100,
        **kwargs: Any,
    ):
        super().__init__(
            seed=seed,
            observations=observations,
            actions=actions,
            offline_data_path=offline_data_path,
            network_cls=DiffusionPolicy,
            network_extra_kwargs=dict(
                model_dim=model_dim,
                emb_dim=emb_dim,
                kernel_size=kernel_size,
                cond_predict_scale=cond_predict_scale,
                dim_mult=tuple(dim_mult),
                n_groups=n_groups,
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                clip_sample=clip_sample,
                sampler=sampler,
                num_inference_steps=num_inference_steps,
            ),
            **kwargs,
        )
