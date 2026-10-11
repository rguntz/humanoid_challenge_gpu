"""MLP policy: ACT's vision with an MLP in place of its transformer.

The images go through ACT's backbone, unchanged (the same torchvision ResNet, layer4
feature map, frozen batch norm, shared by the cameras, its own learning rate), and ACT's
1x1 image projection (here to image_feature_dim channels instead of dim_model). Instead
of becoming transformer tokens, the projected feature maps of every camera are flattened
and concatenated with the robot state, and an MLP predicts the whole action chunk
(chunk_size x action_dim) at once. Same inputs, outputs, L1 loss, normalization and
interface as ACTPolicy (forward, select_action with per-environment action queues,
reset(env_ids), select_action_normalized), so train_bc.py, the evaluation and the
residual RL use it unchanged.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.common.policies.normalize import Normalize, Unnormalize
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from resfit.lerobot.policies.mlp.configuration_mlp import MLPConfig
from resfit.lerobot.policies.pretrained import PreTrainedPolicy


class MLPPolicy(PreTrainedPolicy):
    config_class = MLPConfig
    name = "mlp"

    def __init__(self, config: MLPConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = MLPModel(config)

        self.reset()

    def get_optim_params(self) -> list:
        # Same parameter groups as ACTPolicy: the backbone at optimizer_lr_backbone.
        return [
            {"params": [p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad],
             "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self, env_ids: list[int] | None = None):
        """Clear the action queues of the given environments (all if None)."""
        if not hasattr(self, "_action_queues"):
            return
        for idx in range(len(self._action_queues)) if env_ids is None else env_ids:
            if idx < len(self._action_queues):
                self._action_queues[idx].clear()

    def _ensure_action_queues(self, batch_size: int):
        if not hasattr(self, "_action_queues"):
            self._action_queues = []
        while len(self._action_queues) < batch_size:
            self._action_queues.append(deque([], maxlen=self.config.n_action_steps))
        del self._action_queues[batch_size:]

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """One action per environment; a new chunk is predicted for the environments whose
        queue is empty (same as ACTPolicy.select_action)."""
        self.eval()
        batch = self.normalize_inputs(batch)
        batch_size = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
        self._ensure_action_queues(batch_size)
        empty = [idx for idx, q in enumerate(self._action_queues) if len(q) == 0]
        if empty:
            actions = self.model(batch)[:, : self.config.n_action_steps]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            for idx in empty:
                self._action_queues[idx].extend(actions[idx].unbind(0))
        return torch.stack([q.popleft() for q in self._action_queues], dim=0)

    @torch.no_grad
    def select_action_normalized(self, batch: dict[str, Tensor]) -> Tensor:
        actions = self.select_action(batch)
        return self.normalize_targets({"action": actions})["action"]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """L1 loss on the action chunk, padded steps excluded (as ACT without its VAE)."""
        batch = self.normalize_targets(self.normalize_inputs(batch))
        actions_hat = self.model(batch)
        l1_loss = (F.l1_loss(batch["action"], actions_hat, reduction="none")
                   * ~batch["action_is_pad"].unsqueeze(-1)).mean()
        return l1_loss, {"l1_loss": l1_loss.item()}


class MLPModel(nn.Module):
    """Normalized inputs -> normalized action chunk (B, chunk_size, action_dim)."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_feature.shape[0]
        in_dim = 0
        if config.robot_state_feature:
            in_dim += config.robot_state_feature.shape[0]
        if config.env_state_feature:
            in_dim += config.env_state_feature.shape[0]

        if config.image_features:
            # Backbone for image feature extraction: as ACT (modeling_act.py, class ACT).
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.img_feat_proj = nn.Conv2d(backbone_model.fc.in_features, config.image_feature_dim, kernel_size=1)
            with torch.no_grad():
                for feature in config.image_features.values():
                    fmap = self.backbone(torch.zeros(1, *feature.shape))["feature_map"]
                    in_dim += config.image_feature_dim * fmap.shape[-2] * fmap.shape[-1]

        layers = []
        for hidden in config.hidden_dims:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.chunk_size * self.action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = []
        if self.config.robot_state_feature:
            features.append(batch["observation.state"])
        if self.config.env_state_feature:
            features.append(batch["observation.environment_state"])
        for key in self.config.image_features:
            fmap = self.img_feat_proj(self.backbone(batch[key])["feature_map"])
            features.append(fmap.flatten(1))
        out = self.mlp(torch.cat(features, dim=-1))
        return out.view(-1, self.config.chunk_size, self.action_dim)
