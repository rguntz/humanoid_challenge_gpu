"""Configuration of the MLP policy (modeling_mlp.py): ACT's vision with an MLP in place of
its transformer, a smaller and faster baseline with the same inputs, outputs and interface."""
from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.types import NormalizationMode

from resfit.lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("mlp")
@dataclass
class MLPConfig(PreTrainedConfig):
    """
    Args:
        n_obs_steps: Observation steps given to the policy (only 1 is supported, as in ACT).
        chunk_size: Actions predicted per call, all at once by the MLP's last layer.
        n_action_steps: Actions of each chunk executed before the next call.
        vision_backbone, pretrained_backbone_weights, replace_final_stride_with_dilation:
            the image backbone, exactly as ACT's (torchvision ResNet, layer4 feature map,
            frozen batch norm, shared by the cameras).
        image_feature_dim: Channels of the 1x1 convolution applied to the feature map (ACT's
            image input projection, which projects to dim_model=512 for its transformer).
            Its output, every cell of every camera, is flattened into the MLP's input.
        hidden_dims: Hidden layer sizes of the MLP, which outputs the whole action chunk.
        dropout: Dropout after each hidden layer.
    """

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    image_feature_dim: int = 64
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    dropout: float = 0.1

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})")
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled. Got `n_obs_steps={self.n_obs_steps}`")
        self.hidden_dims = tuple(self.hidden_dims)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
