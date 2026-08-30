"""
Multimodal representation extraction for Milestone 7.

The module loads the pretrained SSL encoder and exposes a
single representation interface:

    image       -> 2048
    radiograph  -> 2048
    text        -> 768

Projection heads are intentionally NOT used for downstream
fusion.
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.ssl.model import MultimodalSSLModel


class SSLRepresentationEncoder(nn.Module):
    """
    Wrapper around the pretrained SSL encoders.

    Output dimensions
    -----------------
    image       : 2048
    radiograph  : 2048
    text        : 768
    """

    IMAGE_DIM = 2048
    RADIOGRAPH_DIM = 2048
    TEXT_DIM = 768

    def __init__(
        self,
        checkpoint_path,
        text_model_name="distilbert-base-uncased",
        device="cpu",
        freeze=True,
    ):
        super().__init__()

        self.device = torch.device(device)

        self.ssl_model = MultimodalSSLModel(
            text_model_name=text_model_name,
        )

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SSL checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        self.ssl_model.load_state_dict(
            checkpoint
        )

        # Projection heads are not used for fusion.
        self.ssl_model.projectors.requires_grad_(
            False
        )

        if freeze:
            self.ssl_model.encoders.requires_grad_(
                False
            )

        self.ssl_model.to(self.device)

        if freeze:
            self.ssl_model.eval()

    @property
    def image_dim(self):
        return self.IMAGE_DIM

    @property
    def radiograph_dim(self):
        return self.RADIOGRAPH_DIM

    @property
    def text_dim(self):
        return self.TEXT_DIM

    @torch.no_grad()
    def encode_image(self, images):
        """
        Encode a batch of already-prepared images.

        Parameters
        ----------
        images:
            Tensor [B, 3, 224, 224]

        Returns
        -------
        Tensor [B, 2048]
        """
        return (
            self.ssl_model
            .encoders
            .encode_image(
                images.to(self.device)
            )
        )

    @torch.no_grad()
    def encode_radiograph(self, radiographs):
        """
        Encode a batch of already-prepared radiographs.

        Parameters
        ----------
        radiographs:
            Tensor [B, 3, 224, 224]

        Returns
        -------
        Tensor [B, 2048]
        """
        return (
            self.ssl_model
            .encoders
            .encode_radiograph(
                radiographs.to(self.device)
            )
        )

    @torch.no_grad()
    def encode_text(
        self,
        input_ids,
        attention_mask,
    ):
        """
        Encode tokenized clinical text.

        Returns
        -------
        Tensor [B, 768]
        """
        return (
            self.ssl_model
            .encoders
            .encode_text(
                input_ids.to(self.device),
                attention_mask.to(self.device),
            )
        )