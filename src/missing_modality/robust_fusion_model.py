"""
Robust multimodal fusion model for Milestone 8.3.2.

The model operates on frozen SSL representations.

Inputs
------
image       : 2048
radiograph  : 2048
text        : 768

Each modality is projected independently to 512 dimensions.

A modality-presence mask is applied explicitly so that the model
knows which modalities are available.

Architecture
------------
image       2048 -> 512
radiograph  2048 -> 512
text         768 -> 512

Each projected representation is multiplied by its availability mask.

Then:

    512 + 512 + 512 + 3 mask values
                    |
                  1539
                    |
                  512
                    |
                    6

The SSL encoders are not part of this model.
Only frozen SSL representations are used.
"""

import torch
import torch.nn as nn


class RobustFusion(nn.Module):
    """
    Mask-aware multimodal fusion model.

    Parameters
    ----------
    image_dim:
        Image SSL representation dimension.

    radiograph_dim:
        Radiograph SSL representation dimension.

    text_dim:
        Clinical text SSL representation dimension.

    modality_dim:
        Projected dimension for each modality.

    hidden_dim:
        Hidden classifier dimension.

    num_labels:
        Number of output labels.

    dropout:
        Classifier dropout probability.
    """

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
            + 3
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.fusion_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        modality_mask,
    ):
        """
        Parameters
        ----------
        image:
            Tensor [B, 2048]

        radiograph:
            Tensor [B, 2048]

        text:
            Tensor [B, 768]

        modality_mask:
            Tensor [B, 3]

            Column order:
                0 = image
                1 = radiograph
                2 = text

            1 means present.
            0 means missing.

        Returns
        -------
        logits:
            Tensor [B, num_labels]
        """

        image = self.image_projection(
            image
        )

        radiograph = self.radiograph_projection(
            radiograph
        )

        text = self.text_projection(
            text
        )

        image_mask = modality_mask[:, 0:1]
        radiograph_mask = modality_mask[:, 1:2]
        text_mask = modality_mask[:, 2:3]

        image = image * image_mask
        radiograph = radiograph * radiograph_mask
        text = text * text_mask

        fused = torch.cat(
            [
                image,
                radiograph,
                text,
                modality_mask,
            ],
            dim=1,
        )

        return self.classifier(
            fused
        )