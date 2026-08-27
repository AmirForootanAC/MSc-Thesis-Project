"""
Preprocessing utilities for SSL multimodal fusion.
"""

import os

import torch

from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms

from . import config


class FusionPreprocessor:
    """
    Preprocessing utilities for multimodal fusion.

    Supports:
        - intraoral photographs
        - dental radiographs
        - clinical text

    Image paths stored in the dataset contain filenames only.
    Because the same filename may exist in both Photographs and
    Radiographs, image resolution is explicitly modality-aware.
    """

    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
    ):

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_name
        )

    # ======================================================
    # Image utilities
    # ======================================================

    @staticmethod
    def resolve_image_path(
        path,
        modality,
    ):
        """
        Resolve a dataset filename to its modality-specific path.

        Important:
        The COde dataset can contain identical filenames in
        Photographs and Radiographs. Therefore the modality must
        be specified explicitly.
        """

        if os.path.isabs(path):
            return path

        if modality == "image":

            root = (
                config.IMAGE_ROOT
                /
                "Photographs"
            )

        elif modality == "radiograph":

            root = (
                config.IMAGE_ROOT
                /
                "Radiographs"
            )

        else:

            raise ValueError(
                f"Unknown image modality: {modality}"
            )

        resolved_path = (
            root
            /
            path
        )

        if not resolved_path.exists():

            raise FileNotFoundError(
                f"{modality.capitalize()} not found: "
                f"{resolved_path}"
            )

        return str(
            resolved_path
        )

    def load_image(
        self,
        path,
        modality,
    ):
        """
        Load and preprocess one image.
        """

        path = self.resolve_image_path(
            path,
            modality,
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        return self.image_transform(
            image
        )

    def process_images(
        self,
        paths,
        modality,
    ):
        """
        Process multiple images belonging to one visit.

        Each image is independently transformed.

        Feature-level mean pooling is performed later by
        the encoder.
        """

        if not paths:

            raise ValueError(
                f"No {modality} images provided."
            )

        images = [
            self.load_image(
                path,
                modality,
            )
            for path in paths
        ]

        return torch.stack(
            images
        )

    # ======================================================
    # Text utilities
    # ======================================================

    def process_text(
        self,
        text,
        max_length=256,
    ):
        """
        Tokenize clinical text.
        """

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids":
                tokens["input_ids"].squeeze(0),

            "attention_mask":
                tokens["attention_mask"].squeeze(0),
        }