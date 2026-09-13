"""
Image encoders for COde baseline models.
"""

import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class ResNet50Encoder(nn.Module):
    """
    ResNet50 feature extractor.

    Output:
    - 2048-dimensional image representation
    """

    def __init__(
        self,
        pretrained=True,
        freeze=False,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(
            weights=weights
        )

        self.features = nn.Sequential(
            *list(
                backbone.children()
            )[:-1]
        )

        self.freeze = freeze

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

            # Keep frozen pretrained BatchNorm layers in eval mode.
            self.features.eval()

    def train(self, mode=True):
        """
        Keep the frozen image encoder in eval mode so that
        BatchNorm running statistics remain fixed.
        """
        super().train(mode)

        if self.freeze:
            self.features.eval()

        return self

    def forward(
        self,
        x,
    ):
        x = self.features(x)

        x = x.flatten(
            start_dim=1
        )

        return x