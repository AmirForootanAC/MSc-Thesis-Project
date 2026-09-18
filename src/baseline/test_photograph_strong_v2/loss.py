"""Class-weighted BCEWithLogitsLoss."""

import numpy as np
import torch
import torch.nn as nn


class ClassWeightedBCEWithLogitsLoss(
    nn.Module
):
    def __init__(
        self,
        pos_weights,
    ):
        super().__init__()

        pos_weights = np.asarray(
            pos_weights,
            dtype=np.float32,
        )

        self.register_buffer(
            "pos_weights",
            torch.tensor(
                pos_weights,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits,
        targets,
    ):
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weights,
        )