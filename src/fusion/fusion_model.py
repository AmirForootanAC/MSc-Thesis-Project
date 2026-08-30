"""
Fusion models for Milestone 7.3.

SimpleFusion
------------
Concatenates modality representations directly:

    image       2048
    radiograph  2048
    text         768
                  |
               concat
                  |
                4864
                  |
                512
                  |
                 6

MainFusion
----------
Projects each modality independently:

    image       2048 -> 512
    radiograph  2048 -> 512
    text         768 -> 512

Then:

    512 + 512 + 512
            |
          concat
            |
          1536 -> 512 -> 6

The SSL encoders are NOT part of these models.
Only frozen SSL representations are used.
"""

import torch
import torch.nn as nn


# ============================================================
# Simple Fusion
# ============================================================

class SimpleFusion(nn.Module):

    def __init__(
        self,
        image_dim=2048,
        radiograph_dim=2048,
        text_dim=768,
        hidden_dim=512,
        num_labels=6,
        dropout=0.3,
    ):

        super().__init__()

        self.image_dim = image_dim
        self.radiograph_dim = radiograph_dim
        self.text_dim = text_dim

        self.fusion_dim = (
            image_dim
            +
            radiograph_dim
            +
            text_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                self.fusion_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_labels,
            ),
        )

    def forward(
        self,
        image,
        radiograph,
        text,
    ):

        fused = torch.cat(
            [
                image,
                radiograph,
                text,
            ],
            dim=1,
        )

        return self.classifier(
            fused
        )


# ============================================================
# Main Fusion
# ============================================================

class MainFusion(nn.Module):

    def __init__(
        self,
        image_dim=2048,
        radiograph_dim=2048,
        text_dim=768,
        modality_dim=512,
        hidden_dim=512,
        num_labels=6,
        dropout=0.3,
    ):

        super().__init__()

        self.image_projection = nn.Sequential(

            nn.Linear(
                image_dim,
                modality_dim,
            ),

            nn.ReLU(),
        )

        self.radiograph_projection = nn.Sequential(

            nn.Linear(
                radiograph_dim,
                modality_dim,
            ),

            nn.ReLU(),
        )

        self.text_projection = nn.Sequential(

            nn.Linear(
                text_dim,
                modality_dim,
            ),

            nn.ReLU(),
        )

        self.fusion_dim = (
            modality_dim * 3
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                self.fusion_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_labels,
            ),
        )

    def forward(
        self,
        image,
        radiograph,
        text,
    ):

        image = self.image_projection(
            image
        )

        radiograph = self.radiograph_projection(
            radiograph
        )

        text = self.text_projection(
            text
        )

        fused = torch.cat(
            [
                image,
                radiograph,
                text,
            ],
            dim=1,
        )

        return self.classifier(
            fused
        )