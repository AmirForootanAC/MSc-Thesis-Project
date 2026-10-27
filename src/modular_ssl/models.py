"""
Modular SSL & Multi-View Cross-Attention Fusion Models.

1. SingleVisualModalitySSLModel:
   Trains either Photograph (ConvNeXt-Tiny) OR Radiograph (DenseNet121)
   with Multi-Head Gated Attention MIL against Bio_ClinicalBERT.

2. MultiViewGatedCrossAttentionFusion:
   Operates on the extracted multi-view SSL representations (both visit-level
   and view-level embeddings) across all 4 scenarios:
       - image_only
       - xray_only
       - text_only
       - full_multimodal
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ssl_v2.encoders import MultiViewVisionEncoder, ClinicalTextEncoder
from src.ssl_v2.contrastive_v2 import (
    ProjectionMLP,
    SemanticSoftContrastiveLoss,
    LocalViewAlignmentLoss,
    compute_intra_visit_multiview_loss,
)
from src.ssl_v2.fusion_model import (
    SafeCrossModalViewAttention,
    scale_with_full_gradient,
)
from . import config


# ============================================================
# 1. Modular Single-Branch Visual <-> Text SSL Model
# ============================================================

class SingleVisualModalitySSLModel(nn.Module):
    """
    Lightweight, decoupled SSL model for one visual modality ('photograph' or 'radiograph')
    paired with clinical text.
    """

    def __init__(self, modality: str = "photograph"):
        super().__init__()
        if modality not in ("photograph", "radiograph"):
            raise ValueError(f"Unsupported visual modality: {modality}")

        self.modality = modality
        backbone_name = (
            config.PHOTO_BACKBONE
            if modality == "photograph"
            else config.RADIO_BACKBONE
        )

        self.vision_encoder = MultiViewVisionEncoder(
            backbone_name=backbone_name,
            pretrained=True,
            out_dim=config.SHARED_EMBED_DIM,
            dropout=0.20,
            freeze_early_stages=config.SSL_FREEZE_VISION_STAGES,
        )

        self.text_encoder = ClinicalTextEncoder(
            model_name=config.TEXT_MODEL_NAME,
            out_dim=config.SHARED_EMBED_DIM,
            freeze_layers=config.SSL_FREEZE_TEXT_LAYERS,
            internal_dropout=0.15,
            dropout=0.20,
        )

        self.vision_proj = ProjectionMLP(
            config.SHARED_EMBED_DIM,
            config.SHARED_EMBED_DIM,
            config.PROJECTION_DIM,
        )
        self.text_proj = ProjectionMLP(
            config.SHARED_EMBED_DIM,
            config.SHARED_EMBED_DIM,
            config.PROJECTION_DIM,
        )

    def forward(
        self,
        padded_images: torch.Tensor,
        view_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        vis_out = self.vision_encoder(padded_images, view_mask)
        txt_out = self.text_encoder(input_ids, attention_mask)

        z_vis = self.vision_proj(vis_out["visit_embedding"])
        z_txt = self.text_proj(txt_out["visit_embedding"])

        return {
            "z_vis": z_vis,                                     # [B, 128]
            "z_txt": z_txt,                                     # [B, 128]
            "vis_visit_embed": vis_out["visit_embedding"],      # [B, 256]
            "vis_view_embeds": vis_out["projected_views"],      # [B, V, 256]
            "txt_visit_embed": txt_out["visit_embedding"],      # [B, 256]
            "txt_raw_feat": txt_out["raw_text_feat"],           # [B, 768]
        }


class DecoupledModalitySSLLoss(nn.Module):
    """
    Computes:
      1. Semantic-Soft Global Contrastive Loss (Visit <-> Text)
      2. Local View-to-Text Alignment Loss (Each individual view <-> Text)
      3. Intra-Visit Multi-View Consistency Loss (Even views <-> Odd views)
    """

    def __init__(self):
        super().__init__()
        self.global_loss = SemanticSoftContrastiveLoss(
            temperature=config.SSL_TEMPERATURE,
            soft_alpha=config.SSL_SOFT_ALPHA,
        )
        self.local_loss = LocalViewAlignmentLoss(
            temperature=config.SSL_TEMPERATURE
        )

    def forward(
        self,
        outputs: dict,
        view_mask: torch.Tensor,
        has_text: torch.Tensor,
    ):
        z_vis = outputs["z_vis"]
        z_txt = outputs["z_txt"]
        vis_views = outputs["vis_view_embeds"]
        txt_visit = outputs["txt_visit_embed"]
        txt_guide = outputs["txt_raw_feat"].detach()

        losses = {}

        if has_text.sum() >= 2:
            l_global = self.global_loss(
                z_vis[has_text],
                z_txt[has_text],
                semantic_guide=txt_guide[has_text],
            )
            l_local = self.local_loss(
                vis_views[has_text],
                view_mask[has_text],
                txt_visit[has_text],
            )
            losses["global_contrastive"] = l_global
            losses["local_view_text"] = l_local * config.SSL_LOCAL_VIEW_WEIGHT

        l_intra = compute_intra_visit_multiview_loss(vis_views, view_mask)
        if l_intra.item() > 0:
            losses["intra_multiview"] = l_intra * config.SSL_INTRA_VIEW_WEIGHT

        if not losses:
            return z_vis.sum() * 0.0, {}

        total = sum(losses.values())
        return total, {k: v.item() for k, v in losses.items()}


# ============================================================
# 2. Multi-View Gated Cross-Modal Fusion (4 Scenarios)
# ============================================================

class MultiViewGatedCrossAttentionFusion(nn.Module):
    """
    Operates directly on pre-extracted multi-view SSL representations!
    Receives:
        photo_visit : [B, 256], photo_views : [B, V_p, 256], photo_mask : [B, V_p]
        radio_visit : [B, 256], radio_views : [B, V_r, 256], radio_mask : [B, V_r]
        text_visit  : [B, 256], text_raw    : [B, 768]

    Supports all 4 scenarios:
        - image_only
        - xray_only
        - text_only
        - full_multimodal
    """

    def __init__(
        self,
        modalities: tuple,
        embed_dim: int = config.SHARED_EMBED_DIM,
        num_labels: int = config.NUM_LABELS,
        dropout: float = config.FUSION_DROPOUT,
        modality_dropout: float = config.MODALITY_DROPOUT,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.has_photo = "photograph" in self.modalities
        self.has_radio = "radiograph" in self.modalities
        self.has_text = "text" in self.modalities
        self.is_full = self.has_photo and self.has_radio and self.has_text
        self.modality_dropout = float(modality_dropout)

        if self.has_photo:
            self.photo_adapter = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.LayerNorm(embed_dim),
            )
            self.photo_view_self_attn = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )

        if self.has_radio:
            self.radio_adapter = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.LayerNorm(embed_dim),
            )
            self.radio_view_self_attn = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )

        if self.has_text:
            self.text_adapter = nn.Sequential(
                nn.LayerNorm(embed_dim + 768),
                nn.Dropout(dropout),
                nn.Linear(embed_dim + 768, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            self.raw_text_norm = nn.LayerNorm(768)

        if self.is_full:
            # Cross-modal view attention: Text queries individual Photo & X-ray views
            self.text_to_photo_views = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )
            self.text_to_radio_views = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )
            # Mutual visual cross-attention: Photo queries X-ray views & vice versa
            self.photo_to_radio_views = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )
            self.radio_to_photo_views = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )

            self.photo_scale = nn.Parameter(torch.full((1, embed_dim), -0.35))
            self.radio_scale = nn.Parameter(torch.full((1, embed_dim), -0.35))

            self.synergy_gate = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

            # 6 streams of 256-d + 768-d raw clinical text = 2304-d
            fused_dim = embed_dim * 6 + 768
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fused_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_labels),
            )

        elif self.has_text and not self.has_photo and not self.has_radio:
            # text_only
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim + 768, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_labels),
            )
        else:
            # image_only or xray_only (visit embedding + self-attended view context)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_labels),
            )

    def forward(self, batch: dict) -> torch.Tensor:
        if not self.is_full:
            if self.has_photo:
                p_visit = self.photo_adapter(batch["photo_visit"])
                p_ctx = self.photo_view_self_attn(
                    p_visit, batch["photo_views"], batch["photo_mask"]
                )
                return self.classifier(torch.cat([p_visit, p_ctx], dim=-1))

            if self.has_radio:
                r_visit = self.radio_adapter(batch["radio_visit"])
                r_ctx = self.radio_view_self_attn(
                    r_visit, batch["radio_views"], batch["radio_mask"]
                )
                return self.classifier(torch.cat([r_visit, r_ctx], dim=-1))

            if self.has_text:
                t_raw = self.raw_text_norm(batch["text_raw"])
                t_emb = self.text_adapter(
                    torch.cat([batch["text_visit"], t_raw], dim=-1)
                )
                return self.classifier(torch.cat([t_emb, t_raw], dim=-1))

        # Full Multimodal Scenario
        p_visit = self.photo_adapter(batch["photo_visit"])
        r_visit = self.radio_adapter(batch["radio_visit"])
        t_raw = self.raw_text_norm(batch["text_raw"])
        t_visit = self.text_adapter(
            torch.cat([batch["text_visit"], t_raw], dim=-1)
        )

        # Cross-modal attention over individual views (no view averaging!)
        t_guided_p = self.text_to_photo_views(
            t_visit, batch["photo_views"], batch["photo_mask"]
        )
        t_guided_r = self.text_to_radio_views(
            t_visit, batch["radio_views"], batch["radio_mask"]
        )
        p_guided_r = self.photo_to_radio_views(
            p_visit, batch["radio_views"], batch["radio_mask"]
        )
        r_guided_p = self.radio_to_photo_views(
            r_visit, batch["photo_views"], batch["photo_mask"]
        )
        visual_cross = 0.5 * (p_guided_r + r_guided_p)

        p_scaled = scale_with_full_gradient(p_visit, self.photo_scale)
        r_scaled = scale_with_full_gradient(r_visit, self.radio_scale)
        v_cross_scaled = scale_with_full_gradient(
            visual_cross, 0.5 * (self.photo_scale + self.radio_scale)
        )

        if self.training and self.modality_dropout > 0.0:
            B = p_scaled.size(0)
            u = torch.rand(B, 1, device=p_scaled.device)
            drop_p = (u < self.modality_dropout).float()
            drop_r = (
                (u >= self.modality_dropout) & (u < 2.0 * self.modality_dropout)
            ).float()
            p_scaled = p_scaled * (1.0 - drop_p) + r_scaled * drop_p * 0.5
            r_scaled = r_scaled * (1.0 - drop_r) + p_scaled * drop_r * 0.5

        synergy = self.synergy_gate(
            torch.cat([p_scaled, r_scaled, t_visit], dim=-1)
        )

        scale_norm = 1.0 / math.sqrt(6.0)
        fused = torch.cat(
            [
                p_scaled * scale_norm,
                r_scaled * scale_norm,
                v_cross_scaled * scale_norm,
                t_guided_p * scale_norm,
                t_guided_r * scale_norm,
                synergy * scale_norm,
                t_raw,
            ],
            dim=-1,
        )
        return self.classifier(fused)