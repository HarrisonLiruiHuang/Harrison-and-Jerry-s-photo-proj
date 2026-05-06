import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from fivek_project.suggestions import EDIT_NAMES


os.environ.setdefault("TORCH_HOME", str(Path("checkpoints/torch_cache").resolve()))


class ResNet18SuggestionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, len(EDIT_NAMES)),
            nn.Tanh(),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.backbone(x)


def build_suggestion_model(pretrained: bool = True) -> nn.Module:
    return ResNet18SuggestionModel(pretrained=pretrained)
