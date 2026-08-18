"""
Baseline classification models for COde dataset.
"""

import torch
import torch.nn as nn


class ImageOnlyBaseline(nn.Module):
    """
    Image-only baseline model.

    Pipeline:
    images
        ->
    encoder
        ->
    mean pooling
        ->
    classifier
    """

    def __init__(
        self,
        aggregator,
        num_labels=13,
    ):

        super().__init__()

        self.aggregator = aggregator

        self.classifier = nn.Sequential(
            nn.Linear(
                2048,
                512,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                num_labels,
            ),
        )


    def forward(
        self,
        batch_images,
    ):

        features = []


        for images in batch_images:

            feature = self.aggregator(
                images
            )

            features.append(
                feature
            )


        features = torch.stack(
            features
        )


        logits = self.classifier(
            features
        )


        return logits