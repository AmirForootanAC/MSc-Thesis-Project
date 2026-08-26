"""
Encoders for self-supervised multimodal representation learning
on the COde dataset.

Each modality has an independent encoder.

Outputs:
    Photograph encoder  -> 2048-dimensional representation
    Radiograph encoder  -> 2048-dimensional representation
    Text encoder        -> 768-dimensional representation
"""

import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

from transformers import AutoModel


# ============================================================
# Image Encoder
# ============================================================

class ResNet50Encoder(nn.Module):
    """
    ResNet50 feature extractor.

    Used independently for:
        - photographs
        - radiographs

    Output:
        2048-dimensional representation.
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

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)

        x = x.flatten(
            start_dim=1
        )

        return x


# ============================================================
# Text Encoder
# ============================================================

class TextEncoder(nn.Module):
    """
    Transformer-based clinical text encoder.

    Default:
        DistilBERT

    Output:
        Model hidden size (768 for DistilBERT).
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        freeze=False,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # DistilBERT has no pooler.
        # Use the first-token representation.
        embedding = (
            outputs.last_hidden_state[:, 0]
        )

        return embedding


# ============================================================
# Multimodal Encoder Container
# ============================================================

class MultimodalEncoders(nn.Module):
    """
    Container for the three independent modality encoders.

    Photograph:
        ResNet50 -> 2048

    Radiograph:
        ResNet50 -> 2048

    Text:
        DistilBERT -> 768
    """

    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        pretrained=True,
        freeze=False,
    ):
        super().__init__()

        self.image_encoder = ResNet50Encoder(
            pretrained=pretrained,
            freeze=freeze,
        )

        self.radiograph_encoder = ResNet50Encoder(
            pretrained=pretrained,
            freeze=freeze,
        )

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze,
        )

    def encode_image(self, x):
        return self.image_encoder(x)

    def encode_radiograph(self, x):
        return self.radiograph_encoder(x)

    def encode_text(
        self,
        input_ids,
        attention_mask,
    ):
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )