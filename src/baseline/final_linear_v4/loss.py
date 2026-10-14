"""Class-weighted BCEWithLogitsLoss with Pos-Weight-Aware Label Smoothing."""

import numpy as np
import torch
import torch.nn as nn


class ClassWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_weights,
        label_smoothing=0.05,
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
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits,
        targets,
    ):
        if self.label_smoothing > 0.0:
            # برای لیبل 1: تبدیل به (1 - eps) جهت جلوگیری از انفجار مثبت Logit
            # برای لیبل 0: تقسیم eps بر pos_weight تا وزن کلاس مثبت باعث تورم احتمال نمونه‌های منفی نشود
            pos_target = 1.0 - self.label_smoothing
            neg_target = (0.25 * self.label_smoothing) / torch.clamp(
                self.pos_weights,
                min=1.0,
            )
            targets = (
                targets * pos_target
                + (1.0 - targets) * neg_target
            )

        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weights,
        )