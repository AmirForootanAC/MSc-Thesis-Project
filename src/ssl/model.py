"""
Multimodal SSL model for COde.

Encoders:
    photograph -> ResNet50 -> projection
    radiograph -> ResNet50 -> projection
    text -> DistilBERT -> projection

Objective:
    Dynamic multimodal contrastive learning
"""

import torch
import torch.nn as nn

from .encoders import MultimodalEncoders
from .projection import MultimodalProjectionHeads
from .contrastive import DynamicMultimodalContrastiveLoss


class MultimodalSSLModel(nn.Module):

    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        temperature=0.07,
    ):
        super().__init__()

        self.encoders = MultimodalEncoders(
            text_model_name=text_model_name,
            pretrained=True,
            freeze=False,
        )

        self.projectors = MultimodalProjectionHeads()

        self.loss_fn = DynamicMultimodalContrastiveLoss(
            temperature=temperature
        )


    def forward(
        self,
        image_embeddings=None,
        radiograph_embeddings=None,
        text_embeddings=None,
        pair_mask=None,
    ):

        return self.loss_fn(
            image_embeddings=image_embeddings,
            radiograph_embeddings=radiograph_embeddings,
            text_embeddings=text_embeddings,
            pair_mask=pair_mask,
        )


    def project_image(self, x):
        z = self.encoders.encode_image(x)
        return self.projectors.encode_image(z)


    def project_radiograph(self, x):
        z = self.encoders.encode_radiograph(x)
        return self.projectors.encode_radiograph(z)


    def project_text(
        self,
        input_ids,
        attention_mask,
    ):
        z = self.encoders.encode_text(
            input_ids,
            attention_mask,
        )

        return self.projectors.encode_text(z)