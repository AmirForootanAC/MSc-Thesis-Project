"""
Feature aggregation modules for COde baseline models.
"""

import torch
import torch.nn as nn


class MeanImageAggregator(nn.Module):
    """
    Encode variable-length image lists
    and aggregate features with mean pooling.
    """

    def __init__(
        self,
        encoder,
    ):

        super().__init__()

        self.encoder = encoder


    def forward(
        self,
        images,
    ):
        """
        Encode images and apply mean pooling.

        Input:
        - list of image tensors

        Output:
        - 2048 dimensional feature
        """

        if len(images) == 0:

            return torch.zeros(
                2048,
                device=next(
                    self.encoder.parameters()
                ).device,
            )


        images = torch.stack(
            images
        )


        if not any(
            p.requires_grad
            for p in self.encoder.parameters()
        ):

            with torch.no_grad():

                features = self.encoder(
                    images
                )

        else:

            features = self.encoder(
                images
            )


        feature = torch.mean(
            features,
            dim=0,
        )


        return feature