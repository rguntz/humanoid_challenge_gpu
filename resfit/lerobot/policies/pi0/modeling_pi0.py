"""PI0 (LeRobot's lerobot.common.policies.pi0) for this repository's training script, evaluation
and residual RL.

LeRobot's PI0Policy, with two changes (fewer than SmolVLA's - see resfit/lerobot/policies/
smolvla/modeling_smolvla.py's own docstring for the full list and why each one was needed; PI0's
own forward() already masks the loss on padded actions via batch.get("action_is_pad") directly -
no key-rename hack needed like SmolVLA's - and its own loss_dict mixes tensor and float entries,
which train_bc.py's logging already filters to the float ones, so no reshaping is needed either):
  - select_action / reset(env_ids) keep one action queue per environment (LeRobot's has one queue
    for the whole batch and resets all of it), for vectorized envs whose episodes end at different
    steps; a new chunk is predicted for the environments whose queue is empty. PI0 takes one
    observation frame (n_obs_steps=1, hard-enforced by PI0Config.__post_init__), so there is no
    history to keep.
  - The task sentence is added when the batch has none (the env's observations): the dataset's
    own task, DEFAULT_TASK.

Deliberately NOT changed, unlike SmolVLA: precision. LeRobot's own PaliGemmaWithExpertModel
already converts the VLM + action-expert weights to bf16 in its own __init__
(to_bfloat16_like_physical_intelligence) - matching how Physical Intelligence's own pi0 checkpoint
was trained/released - so this repo leaves that exactly as upstream sets it (no .float() override
here), rather than re-deriving SmolVLA's own all-fp32 choice: the pretrained checkpoint's own
precision is the reference point for this policy, not an independent decision.

load_pretrained_weights initializes the model from a PI0 checkpoint (lerobot/pi0): its weights,
not its normalization statistics, which come from the dataset trained on.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import safetensors.torch
import torch
from lerobot.common.constants import OBS_STATE
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as LeRobotPI0Policy
from torch import Tensor

DEFAULT_TASK = "pick up the pill box and place it in the container"
# NOTE: deliberately NOT the same as convert_pick_place_to_lerobot.TASK ("pick up the bottle
# and place it in the container", baked into the dataset's own per-episode task field at
# conversion time) - the object is actually a small pill box, not a bottle, but this wording is
# only used here, as this policy's own eval-time fallback when the env doesn't supply a task
# (see the module docstring). Left mismatched from the dataset's own training-time task string
# and from SmolVLA's own DEFAULT_TASK on purpose - explicit request, no dataset re-conversion.
NORMALIZATION_PREFIXES = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")


class PI0Policy(LeRobotPI0Policy):
    config_class = PI0Config
    name = "pi0"

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
        """PI0's own prepare_language (unlike SmolVLA's) does not broadcast a single task string
        across the batch - it iterates batch["task"] directly, so a bare string gets iterated
        character-by-character instead of once per sample. Build the per-sample list explicitly."""
        if "task" in batch:
            return batch
        batch_size = batch[OBS_STATE].shape[0]
        return {**batch, "task": [DEFAULT_TASK] * batch_size}

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """One action per environment (see the module docstring). Body otherwise identical to
        LeRobot's own PI0Policy.select_action - only the single shared queue there is replaced by
        one queue per environment here."""
        self.eval()
        batch = self._with_task(batch)
        batch_size = batch[OBS_STATE].shape[0]
        while len(self._env_queues) < batch_size:
            self._env_queues.append(deque(maxlen=self.config.n_action_steps))
        del self._env_queues[batch_size:]
        empty = [i for i, q in enumerate(self._env_queues) if len(q) == 0]
        if empty:
            if self.config.adapt_to_pi_aloha:
                batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            normalized = self.normalize_inputs(dict(batch))
            images, img_masks = self.prepare_images(normalized)
            state = self.prepare_state(normalized)
            lang_tokens, lang_masks = self.prepare_language(normalized)
            actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)
            for i in empty:
                self._env_queues[i].extend(actions[i].unbind(0))
        return torch.stack([q.popleft() for q in self._env_queues], dim=0)

    @torch.no_grad
    def select_action_normalized(self, batch: dict[str, Tensor]) -> Tensor:
        actions = self.select_action(batch)
        return self.normalize_targets({"action": actions})["action"]

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        batch = self._with_task(dict(batch))
        return super().forward(batch, noise, time)


def load_pretrained_weights(policy: torch.nn.Module, checkpoint: str) -> dict:
    """Load a PI0 checkpoint's weights (a local folder or a Hugging Face repo id with a
    model.safetensors) into `policy`, except its normalization statistics. Every other weight
    of the policy must be in the checkpoint with the same shape. Returns counts for logging."""
    path = Path(checkpoint) / "model.safetensors"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(checkpoint, "model.safetensors"))
    weights = safetensors.torch.load_file(str(path))
    # Checkpoints saved from a torch.compile'd model prefix the model's keys (LeRobot's own loaders).
    weights = {k.replace("model._orig_mod.", "model."): v for k, v in weights.items()}
    own = policy.state_dict()
    to_load = {k: v for k, v in weights.items() if not k.startswith(NORMALIZATION_PREFIXES)}
    mismatched = [k for k, v in to_load.items() if k in own and own[k].shape != v.shape]
    missing = [k for k in own if not k.startswith(NORMALIZATION_PREFIXES) and k not in to_load]
    # A checkpoint key tied to another one (e.g. PaliGemma/Gemma's input embedding tied to its
    # output head, tie_word_embeddings) is only saved once - PI0's own live model already shares
    # the same underlying tensor for both names (set up by the HF model's own tie_weights(), not
    # by load_pretrained_weights itself), so loading the OTHER (present) name of the pair already
    # updates this one too; not a real missing weight.
    missing = [k for k in missing
               if not any(k != k2 and k2 in to_load and own[k].data_ptr() == own[k2].data_ptr() for k2 in own)]
    if mismatched or missing:
        raise ValueError(f"{checkpoint} does not fit this policy: {len(missing)} weights missing "
                         f"(e.g. {missing[:3]}), {len(mismatched)} with another shape (e.g. {mismatched[:3]})")
    unused = [k for k in to_load if k not in own]
    policy.load_state_dict({k: v for k, v in to_load.items() if k in own}, strict=False)
    return {"loaded": len(to_load) - len(unused), "unused_in_checkpoint": len(unused)}
