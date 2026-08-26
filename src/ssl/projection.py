"""
Projection heads for self-supervised multimodal learning.

Each modality-specific encoder produces a representation with a
different dimensionality. Projection heads map these representations
into a shared embedding space for contrastive learning.

Inputs:
    Photograph: 2048
    Radiograph: 2048
    Text: 768

Output:
    Shared embedding dimension: 128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head.

    Architecture:
        input_dim -> hidden_dim -> embedding_dim

    The final embedding is L2-normalized so that cosine similarity
    can be computed directly using dot products.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        embedding_dim=128,
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(self, x):

        x = self.projector(x)

        x = F.normalize(
            x,
            p=2,
            dim=-1,
        )

        return x


class MultimodalProjectionHeads(nn.Module):
    """
    Projection heads for the three COde modalities.

    Photograph:
        2048 -> 512 -> 128

    Radiograph:
        2048 -> 512 -> 128

    Text:
        768 -> 512 -> 128
    """

    def __init__(
        self,
        image_dim=2048,
        radiograph_dim=2048,
        text_dim=768,
        hidden_dim=512,
        embedding_dim=128,
    ):
        super().__init__()

        self.image_projection = ProjectionHead(
            input_dim=image_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

        self.radiograph_projection = ProjectionHead(
            input_dim=radiograph_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

        self.text_projection = ProjectionHead(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

    def encode_image(self, x):

        return self.image_projection(x)

    def encode_radiograph(self, x):

        return self.radiograph_projection(x)

    def encode_text(self, x):

        return self.text_projection(x)