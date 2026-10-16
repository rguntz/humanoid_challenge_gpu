"""SmolVLA (LeRobot's lerobot.common.policies.smolvla) for this repository's training script,
evaluation and residual RL.

LeRobot's SmolVLAPolicy, with five changes:
  - All weights in fp32 (LeRobot keeps the language model and action expert in bf16, see
    __init__).
  - select_action / reset(env_ids) keep one action queue per environment (LeRobot's has one
    queue for the whole batch and resets all of it), for vectorized envs whose episodes end
    at different steps; a new chunk is predicted for the environments whose queue is empty.
    SmolVLA takes one observation frame (n_obs_steps = 1), so there is no history to keep.
  - The task sentence is added when the batch has none (the env's observations): the
    datasets' own task, DEFAULT_TASK.
  - The loss is masked on the chunk's padded actions (past the episode's end): LeRobot reads
    the mask from batch["actions_id_pad"], a key nothing writes (LeRobotDataset writes
    action_is_pad), so it never masks.
  - forward returns the loss and {"loss": float}: LeRobot's loss_dict also holds the
    per-element loss tensors.
load_pretrained_weights initializes the model from a SmolVLA checkpoint (lerobot/smolvla_base):
its weights, not its normalization statistics, which come from the dataset trained on.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import safetensors.torch
import torch
from lerobot.common.constants import OBS_STATE
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLAPolicy
from torch import Tensor

DEFAULT_TASK = "pick up the pill box and place it in the container"
# Matches convert_pick_place_to_lerobot.TASK, baked into the dataset's own per-episode task
# field at conversion time: this policy's own eval-time fallback, used when the env doesn't
# supply a task (see the module docstring), must say the same thing training did. In practice
# train_bc.evaluate() always passes the dataset's own task explicitly (vec_env.py reads it
# from meta/info.json), so this constant is only reached for a dataset built before --task
# existed (no "task" in its meta/info.json).
NORMALIZATION_PREFIXES = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")


class SmolVLAPolicy(LeRobotSmolVLAPolicy):
    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(self, config: SmolVLAConfig, dataset_stats: dict | None = None):
        super().__init__(config, dataset_stats)
        # LeRobot builds the language model and the action expert in bf16 (their config's
        # torch_dtype): AdamW would then update bf16 weights directly, losing every update
        # under ~0.4 % of a weight. fp32 weights; train_bc's --precision bf16 still runs the
        # forward pass in bf16 (autocast). Checkpoints are loaded into these fp32 weights.
        self.float()

    def reset(self, env_ids: list[int] | None = None):
        """Clear the action queues of the given environments (all if None)."""
        if env_ids is None or not hasattr(self, "_env_queues"):
            super().reset()
            self._env_queues = []
            return
        for idx in env_ids:
            if idx < len(self._env_queues):
                self._env_queues[idx].clear()

    @staticmethod
    def _with_task(batch: dict) -> dict:
        return batch if "task" in batch else {**batch, "task": DEFAULT_TASK}

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """One action per environment (see the module docstring)."""
        self.eval()
        batch = self._with_task(batch)
        batch_size = batch[OBS_STATE].shape[0]
        while len(self._env_queues) < batch_size:
            self._env_queues.append(deque(maxlen=self.config.n_action_steps))
        del self._env_queues[batch_size:]
        empty = [i for i, q in enumerate(self._env_queues) if len(q) == 0]
        if empty:
            actions = self._get_action_chunk(self._prepare_batch(dict(batch)), noise)
            actions = actions[:, : self.config.n_action_steps]
            for i in empty:
                self._env_queues[i].extend(actions[i].unbind(0))
        return torch.stack([q.popleft() for q in self._env_queues], dim=0)

    @torch.no_grad
    def select_action_normalized(self, batch: dict[str, Tensor]) -> Tensor:
        actions = self.select_action(batch)
        return self.normalize_targets({"action": actions})["action"]

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        batch = self._with_task(dict(batch))
        if "action_is_pad" in batch:
            batch["actions_id_pad"] = batch["action_is_pad"]  # the key LeRobot's forward reads
        loss, _ = super().forward(batch, noise, time)
        return loss, {"loss": loss.item()}


def load_pretrained_weights(policy: torch.nn.Module, checkpoint: str) -> dict:
    """Load a SmolVLA checkpoint's weights (a local folder or a Hugging Face repo id with a
    model.safetensors) into `policy`, except its normalization statistics. Every other weight
    of the policy must be in the checkpoint with the same shape. Returns counts for logging."""
    path = Path(checkpoint) / "model.safetensors"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(checkpoint, "model.safetensors"))
    weights = safetensors.torch.load_file(str(path))
    # Checkpoints saved from a torch.compile'd model prefix the model's keys (LeRobot's load_smolvla).
    weights = {k.replace("model._orig_mod.", "model."): v for k, v in weights.items()}
    own = policy.state_dict()
    to_load = {k: v for k, v in weights.items() if not k.startswith(NORMALIZATION_PREFIXES)}
    mismatched = [k for k, v in to_load.items() if k in own and own[k].shape != v.shape]
    missing = [k for k in own if not k.startswith(NORMALIZATION_PREFIXES) and k not in to_load]
    if mismatched or missing:
        raise ValueError(f"{checkpoint} does not fit this policy: {len(missing)} weights missing "
                         f"(e.g. {missing[:3]}), {len(mismatched)} with another shape (e.g. {mismatched[:3]})")
    unused = [k for k in to_load if k not in own]
    policy.load_state_dict({k: v for k, v in to_load.items() if k in own}, strict=False)
    return {"loaded": len(to_load) - len(unused), "unused_in_checkpoint": len(unused)}
