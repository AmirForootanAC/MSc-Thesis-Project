"""
Dynamic multimodal contrastive learning for COde SSL.

Supports:
    - Image <-> Text
    - Image <-> Radiograph
    - Radiograph <-> Text

Missing modalities are handled dynamically:
    a pair contributes to the loss only when both modalities
    are available in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricContrastiveLoss(nn.Module):
    """
    CLIP-style symmetric InfoNCE loss.

    Given two sets of embeddings belonging to the same visits:

        z_a[i] <-> z_b[i]

    diagonal pairs are positives and all other pairs are negatives.
    """

    def __init__(
        self,
        temperature=0.07,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero"
            )

        self.temperature = temperature


    def forward(
        self,
        z_a,
        z_b,
    ):
        """
        Parameters
        ----------
        z_a:
            Tensor of shape (N, D)

        z_b:
            Tensor of shape (N, D)

        Returns
        -------
        loss:
            Scalar symmetric contrastive loss.
        """

        if z_a.ndim != 2 or z_b.ndim != 2:
            raise ValueError(
                "Embeddings must have shape (N, D)"
            )

        if z_a.shape != z_b.shape:
            raise ValueError(
                "Both embedding tensors must have the same shape"
            )

        batch_size = z_a.size(0)

        if batch_size < 2:
            raise ValueError(
                "Contrastive loss requires at least 2 pairs"
            )

        z_a = F.normalize(
            z_a,
            dim=1,
        )

        z_b = F.normalize(
            z_b,
            dim=1,
        )

        logits = (
            z_a @ z_b.T
        ) / self.temperature

        targets = torch.arange(
            batch_size,
            device=logits.device,
        )

        loss_a_to_b = F.cross_entropy(
            logits,
            targets,
        )

        loss_b_to_a = F.cross_entropy(
            logits.T,
            targets,
        )

        loss = (
            loss_a_to_b +
            loss_b_to_a
        ) / 2.0

        return loss


class DynamicMultimodalContrastiveLoss(nn.Module):
    """
    Dynamic multimodal contrastive objective.

    Available pairs:
        image_text
        image_radiograph
        radiograph_text

    A pair is included only if both modalities are available.

    The final loss is the mean over the valid pair losses.
    """

    def __init__(
        self,
        temperature=0.07,
    ):
        super().__init__()

        self.contrastive_loss = (
            SymmetricContrastiveLoss(
                temperature=temperature
            )
        )


    def forward(
        self,
        image_embeddings=None,
        radiograph_embeddings=None,
        text_embeddings=None,
        pair_mask=None,
    ):
        """
        Parameters
        ----------
        image_embeddings:
            (B, D) or None

        radiograph_embeddings:
            (B, D) or None

        text_embeddings:
            (B, D) or None

        pair_mask:
            Optional dictionary containing boolean masks:

                {
                    "image_text": ...,
                    "image_radiograph": ...,
                    "radiograph_text": ...
                }

        Returns
        -------
        total_loss:
            Mean loss over valid modality pairs.

        pair_losses:
            Dictionary containing individual pair losses.
        """

        pair_losses = {}

        # --------------------------------------------------
        # Image <-> Text
        # --------------------------------------------------

        if (
            image_embeddings is not None
            and text_embeddings is not None
        ):

            mask = self._get_mask(
                pair_mask,
                "image_text",
                image_embeddings.size(0),
                image_embeddings.device,
            )

            if mask.sum() >= 2:

                pair_losses["image_text"] = (
                    self.contrastive_loss(
                        image_embeddings[mask],
                        text_embeddings[mask],
                    )
                )


        # --------------------------------------------------
        # Image <-> Radiograph
        # --------------------------------------------------

        if (
            image_embeddings is not None
            and radiograph_embeddings is not None
        ):

            mask = self._get_mask(
                pair_mask,
                "image_radiograph",
                image_embeddings.size(0),
                image_embeddings.device,
            )

            if mask.sum() >= 2:

                pair_losses["image_radiograph"] = (
                    self.contrastive_loss(
                        image_embeddings[mask],
                        radiograph_embeddings[mask],
                    )
                )


        # --------------------------------------------------
        # Radiograph <-> Text
        # --------------------------------------------------

        if (
            radiograph_embeddings is not None
            and text_embeddings is not None
        ):

            mask = self._get_mask(
                pair_mask,
                "radiograph_text",
                radiograph_embeddings.size(0),
                radiograph_embeddings.device,
            )

            if mask.sum() >= 2:

                pair_losses["radiograph_text"] = (
                    self.contrastive_loss(
                        radiograph_embeddings[mask],
                        text_embeddings[mask],
                    )
                )


        # --------------------------------------------------
        # No valid pairs
        # --------------------------------------------------

        if not pair_losses:

            # Keep the loss connected to the computation graph
            # when embeddings are available.
            reference = next(
                (
                    x
                    for x in [
                        image_embeddings,
                        radiograph_embeddings,
                        text_embeddings,
                    ]
                    if x is not None
                ),
                None,
            )

            if reference is None:

                raise ValueError(
                    "At least one modality embedding is required"
                )

            return (
                reference.sum() * 0.0,
                pair_losses,
            )


        # --------------------------------------------------
        # Dynamic aggregation
        # --------------------------------------------------

        total_loss = torch.stack(
            list(
                pair_losses.values()
            )
        ).mean()

        return total_loss, pair_losses


    @staticmethod
    def _get_mask(
        pair_mask,
        pair_name,
        batch_size,
        device,
    ):
        """
        Return the validity mask for a modality pair.

        If no explicit mask is provided, all samples are
        considered valid.
        """

        if pair_mask is None:

            return torch.ones(
                batch_size,
                dtype=torch.bool,
                device=device,
            )

        mask = pair_mask.get(
            pair_name
        )

        if mask is None:

            return torch.ones(
                batch_size,
                dtype=torch.bool,
                device=device,
            )

        return mask.to(
            device=device,
            dtype=torch.bool,
        )