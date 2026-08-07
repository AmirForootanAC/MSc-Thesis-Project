"""
Image loading utilities for the COde multimodal dataset.

This module provides a lightweight image loading layer.

Responsibilities:
    - Read image files from disk
    - Convert images to RGB
    - Support multiple images per visit
    - Preserve variable image counts
    - Handle missing image files

This module intentionally does NOT perform:
    - Data augmentation
    - Normalization
    - Resizing policy
    - Image aggregation

Those decisions belong to later preprocessing/model stages.
"""

from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
from torchvision.transforms import ToTensor


class ImageLoader:
    """
    Generic image loader for multimodal dental images.

    Supports:
        - Photographs
        - Radiographs

    Input:
        Image file paths

    Output:
        List of tensors
    """

    def __init__(
        self,
        strict: bool = False,
    ):
        """
        Parameters
        ----------
        strict:
            If True:
                Missing/corrupted files raise errors.

            If False:
                Invalid files are skipped.
        """

        self.strict = strict

        self.to_tensor = ToTensor()

    # --------------------------------------------------
    # Single image loading
    # --------------------------------------------------

    def load_image(
        self,
        image_path: str,
    ) -> Optional[torch.Tensor]:
        """
        Load one image.

        Returns
        -------
        Tensor:
            Image tensor with shape:
            (C, H, W)

        None:
            If loading fails and strict=False.
        """

        path = Path(image_path)

        try:

            with Image.open(path) as image:

                image = image.convert(
                    "RGB"
                )

                tensor = self.to_tensor(
                    image
                )

            return tensor

        except Exception as error:

            if self.strict:
                raise error

            return None

    # --------------------------------------------------
    # Multiple image loading
    # --------------------------------------------------

    def load_images(
        self,
        image_paths: List[str],
    ) -> List[torch.Tensor]:
        """
        Load multiple images belonging to
        one modality of a visit.

        Parameters
        ----------
        image_paths:
            List of image paths.

        Returns
        -------
        List of tensors.

        Missing modality:
            returns empty list.
        """

        loaded_images = []

        for image_path in image_paths:

            image = self.load_image(
                image_path
            )

            if image is not None:
                loaded_images.append(
                    image
                )

        return loaded_images

    # --------------------------------------------------
    # Modality-specific helpers
    # --------------------------------------------------

    def load_photographs(
        self,
        image_paths: List[str],
    ) -> List[torch.Tensor]:
        """
        Load photograph images.
        """

        return self.load_images(
            image_paths
        )

    def load_radiographs(
        self,
        image_paths: List[str],
    ) -> List[torch.Tensor]:
        """
        Load radiograph images.
        """

        return self.load_images(
            image_paths
        )