from pathlib import Path

from PIL import Image
import torch


class COdeImageLoader:
    """
    Modality-aware image loader for COde dataset.

    Supports:
    - photographs
    - radiographs
    """

    def __init__(
        self,
        image_root,
        transform=None,
    ):

        self.image_root = Path(image_root)
        self.transform = transform

        self.photograph_root = (
            self.image_root / "Photographs"
        )

        self.radiograph_root = (
            self.image_root / "Radiographs"
        )


    def _resolve_path(
        self,
        filename,
        modality,
    ):

        if modality == "photograph":

            path = (
                self.photograph_root /
                filename
            )

        elif modality == "radiograph":

            path = (
                self.radiograph_root /
                filename
            )

        else:

            raise ValueError(
                f"Unknown modality: {modality}"
            )


        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {path}"
            )


        return path


    def load(
        self,
        filename,
        modality,
    ):

        path = self._resolve_path(
            filename,
            modality,
        )


        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        if self.transform:

            image = self.transform(
                image
            )


        return image