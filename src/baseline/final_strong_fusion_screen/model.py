"""Fast frozen-feature fusion models."""

import torch
import torch.nn as nn


class TextOnlyFusion(nn.Module):
    def __init__(
        self,
        text_dim,
        num_labels,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                text_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(
                256,
                num_labels,
            ),
        )

    def forward(self, text):
        return self.network(text)


class TextAnchoredFusion(nn.Module):
    """
    Text is the primary representation.

    Image information enters as a residual contribution.
    """

    def __init__(
        self,
        text_dim,
        image_dim,
        num_labels,
    ):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(
                text_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(
                image_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.alpha = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.output_norm = nn.LayerNorm(
            256
        )

        self.classifier = nn.Linear(
            256,
            num_labels,
        )

    def forward(
        self,
        text,
        image,
    ):
        text_features = (
            self.text_projection(text)
        )

        image_features = (
            self.image_projection(image)
        )

        alpha = torch.sigmoid(
            self.alpha
        )

        fused = (
            text_features
            + alpha * image_features
        )

        fused = self.output_norm(
            fused
        )

        return self.classifier(
            fused
        )


class FullTextAnchoredFusion(nn.Module):
    """
    Text-anchored fusion with separate photograph
    and radiograph residual contributions.
    """

    def __init__(
        self,
        text_dim,
        image_dim,
        num_labels,
    ):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(
                text_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.photograph_projection = nn.Sequential(
            nn.Linear(
                image_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.radiograph_projection = nn.Sequential(
            nn.Linear(
                image_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.photo_alpha = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.xray_alpha = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.output_norm = nn.LayerNorm(
            256
        )

        self.classifier = nn.Linear(
            256,
            num_labels,
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        text_features = (
            self.text_projection(text)
        )

        photograph_features = (
            self.photograph_projection(
                photograph
            )
        )

        radiograph_features = (
            self.radiograph_projection(
                radiograph
            )
        )

        photo_alpha = torch.sigmoid(
            self.photo_alpha
        )

        xray_alpha = torch.sigmoid(
            self.xray_alpha
        )

        fused = (
            text_features
            + photo_alpha * photograph_features
            + xray_alpha * radiograph_features
        )

        fused = self.output_norm(
            fused
        )

        return self.classifier(
            fused
        )