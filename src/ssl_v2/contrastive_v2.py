"""
Medical Multimodal Self-Supervised Learning Objective (SSL v2).

Combines three complementary self-supervised signals:
1. Semantic-Aware Soft-Target InfoNCE (MedCLIP-inspired):
   Prevents false-negative repulsion when two visits in the same batch
   share identical dental pathologies.
2. Local View-to-Visit Max-Mean Alignment (GLoRIA-inspired):
   Directly supervises individual intraoral/radiograph views against the
   cross-modal anchor so every image in a multi-view visit receives strong gradients.
3. Intra-Visit Multi-View Consistency:
   Splits multi-view visits into two disjoint view subsets and enforces
   patient/visit invariance across different camera angles and X-ray projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """Maps 256-d encoder embeddings to 128-d L2-normalized SSL space."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class SemanticSoftContrastiveLoss(nn.Module):
    """
    Symmetric contrastive loss with clinical semantic soft targets.
    """

    def __init__(self, temperature: float = 0.07, soft_alpha: float = 0.25):
        super().__init__()
        self.temperature = temperature
        self.soft_alpha = soft_alpha

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        semantic_guide: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = z_a.size(0)
        if N < 2:
            return z_a.sum() * 0.0

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits_ab = (z_a @ z_b.T) / self.temperature
        logits_ba = logits_ab.T

        hard_targets = torch.eye(N, device=z_a.device, dtype=z_a.dtype)

        if semantic_guide is not None and self.soft_alpha > 0.0:
            with torch.no_grad():
                g = F.normalize(semantic_guide.detach(), dim=-1)
                sim = (g @ g.T) / 0.12
                soft_targets = F.softmax(sim, dim=-1)
                targets = (
                    (1.0 - self.soft_alpha) * hard_targets
                    + self.soft_alpha * soft_targets
                )
        else:
            targets = hard_targets

        loss_ab = -(targets * F.log_softmax(logits_ab, dim=-1)).sum(dim=-1).mean()
        loss_ba = -(targets * F.log_softmax(logits_ba, dim=-1)).sum(dim=-1).mean()
        return 0.5 * (loss_ab + loss_ba)


class LocalViewAlignmentLoss(nn.Module):
    """
    Aligns individual views {v_{i,1}, ..., v_{i,M}} of visit i with the
    cross-modal representation u_i using attention-weighted view similarity.
    Ensures every single view in a visit is trained without averaging.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        projected_views: torch.Tensor,
        view_mask: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        projected_views : [N, V, D]
        view_mask       : [N, V]
        target_embeds   : [N, D]
        """
        N, V, D = projected_views.shape
        if N < 2:
            return projected_views.sum() * 0.0

        views_norm = F.normalize(projected_views, dim=-1)
        target_norm = F.normalize(target_embeds, dim=-1)

        # Similarity between all views of sample i and target j: [N, V, N]
        sim_matrix = torch.einsum("nvd,md->nvm", views_norm, target_norm)
        sim_matrix = sim_matrix.masked_fill(~view_mask.unsqueeze(-1), -1e4)

        # Log-Sum-Exp pooling over views of visit i with respect to target j:
        # Smoothly approximates max_v sim(v_{i,v}, u_j) so focal lesion views dominate
        tau_v = 0.15
        view_Counts = view_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        lse_sim = tau_v * (
            torch.logsumexp(sim_matrix / tau_v, dim=1) - torch.log(view_Counts)
        )  # [N, N]

        logits = lse_sim / self.temperature
        labels = torch.arange(N, device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )


def compute_intra_visit_multiview_loss(
    projected_views: torch.Tensor,
    view_mask: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """
    For visits that contain >= 2 views, splits the views into two disjoint subsets
    (even-indexed vs odd-indexed views) and enforces contrastive invariance
    between the two subsets of the same visit.
    """
    view_counts = view_mask.sum(dim=1)
    multi_mask = view_counts >= 2

    if multi_mask.sum() < 2:
        return projected_views.sum() * 0.0

    sub_views = projected_views[multi_mask]  # [M, V, D]
    sub_mask = view_mask[multi_mask]         # [M, V]

    even_views = sub_views[:, 0::2, :]
    even_mask = sub_mask[:, 0::2].unsqueeze(-1).float()
    z_even = (even_views * even_mask).sum(dim=1) / even_mask.sum(dim=1).clamp(min=1.0)

    odd_views = sub_views[:, 1::2, :]
    odd_mask = sub_mask[:, 1::2].unsqueeze(-1).float()
    z_odd = (odd_views * odd_mask).sum(dim=1) / odd_mask.sum(dim=1).clamp(min=1.0)

    z_even = F.normalize(z_even, dim=-1)
    z_odd = F.normalize(z_odd, dim=-1)

    logits = (z_even @ z_odd.T) / temperature
    targets = torch.arange(z_even.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )


class DynamicMultimodalSSLV2Loss(nn.Module):
    """
    Master SSL v2 loss combining Global Soft-Contrastive, Local View-Level,
    and Intra-Visit Multi-View Consistency across all valid modality pairs.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.07,
        soft_alpha: float = 0.25,
        local_view_weight: float = 0.35,
        intra_view_weight: float = 0.30,
    ):
        super().__init__()
        self.photo_proj = ProjectionMLP(embed_dim, embed_dim, proj_dim)
        self.radio_proj = ProjectionMLP(embed_dim, embed_dim, proj_dim)
        self.text_proj = ProjectionMLP(embed_dim, embed_dim, proj_dim)

        self.global_contrastive = SemanticSoftContrastiveLoss(
            temperature=temperature,
            soft_alpha=soft_alpha,
        )
        self.local_alignment = LocalViewAlignmentLoss(temperature=temperature)
        self.local_view_weight = local_view_weight
        self.intra_view_weight = intra_view_weight

    def forward(
        self,
        photo_out: dict,
        radio_out: dict,
        text_out: dict,
        has_photo: torch.Tensor,
        has_radio: torch.Tensor,
        has_text: torch.Tensor,
        photo_view_mask: torch.Tensor,
        radio_view_mask: torch.Tensor,
    ):
        z_photo = self.photo_proj(photo_out["visit_embedding"])
        z_radio = self.radio_proj(radio_out["visit_embedding"])
        z_text = self.text_proj(text_out["visit_embedding"])
        raw_text_guide = text_out["raw_text_feat"].detach()

        pair_losses = {}

        # 1. Photograph <-> Text
        mask_pt = has_photo & has_text
        if mask_pt.sum() >= 2:
            loss_global_pt = self.global_contrastive(
                z_photo[mask_pt],
                z_text[mask_pt],
                semantic_guide=raw_text_guide[mask_pt],
            )
            loss_local_pt = self.local_alignment(
                photo_out["projected_views"][mask_pt],
                photo_view_mask[mask_pt],
                text_out["visit_embedding"][mask_pt],
            )
            pair_losses["image_text"] = (
                loss_global_pt + self.local_view_weight * loss_local_pt
            )

        # 2. Photograph <-> Radiograph
        mask_pr = has_photo & has_radio
        if mask_pr.sum() >= 2:
            guide_pr = raw_text_guide[mask_pr] if has_text[mask_pr].all() else None
            loss_global_pr = self.global_contrastive(
                z_photo[mask_pr],
                z_radio[mask_pr],
                semantic_guide=guide_pr,
            )
            loss_local_pr = 0.5 * (
                self.local_alignment(
                    photo_out["projected_views"][mask_pr],
                    photo_view_mask[mask_pr],
                    radio_out["visit_embedding"][mask_pr],
                )
                + self.local_alignment(
                    radio_out["projected_views"][mask_pr],
                    radio_view_mask[mask_pr],
                    photo_out["visit_embedding"][mask_pr],
                )
            )
            pair_losses["image_radiograph"] = (
                loss_global_pr + self.local_view_weight * loss_local_pr
            )

        # 3. Radiograph <-> Text
        mask_rt = has_radio & has_text
        if mask_rt.sum() >= 2:
            loss_global_rt = self.global_contrastive(
                z_radio[mask_rt],
                z_text[mask_rt],
                semantic_guide=raw_text_guide[mask_rt],
            )
            loss_local_rt = self.local_alignment(
                radio_out["projected_views"][mask_rt],
                radio_view_mask[mask_rt],
                text_out["visit_embedding"][mask_rt],
            )
            pair_losses["radiograph_text"] = (
                loss_global_rt + self.local_view_weight * loss_local_rt
            )

        # 4. Intra-Visit Multi-View Consistency (Photo & Radio)
        intra_losses = []
        if has_photo.sum() >= 2:
            l_intra_p = compute_intra_visit_multiview_loss(
                photo_out["projected_views"][has_photo],
                photo_view_mask[has_photo],
            )
            if l_intra_p.item() > 0:
                intra_losses.append(l_intra_p)

        if has_radio.sum() >= 2:
            l_intra_r = compute_intra_visit_multiview_loss(
                radio_out["projected_views"][has_radio],
                radio_view_mask[has_radio],
            )
            if l_intra_r.item() > 0:
                intra_losses.append(l_intra_r)

        if intra_losses:
            pair_losses["intra_multiview"] = (
                torch.stack(intra_losses).mean() * self.intra_view_weight
            )

        if not pair_losses:
            zero_loss = (z_photo.sum() + z_radio.sum() + z_text.sum()) * 0.0
            return zero_loss, {}

        total_loss = torch.stack(list(pair_losses.values())).mean()
        return total_loss, pair_losses