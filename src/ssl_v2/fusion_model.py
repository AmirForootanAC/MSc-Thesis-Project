"""
Unified 4-Scenario Multimodal Architecture (SSL v2 Downstream & Fusion v2).

Supports all 4 execution scenarios:
    1. image_only      : ConvNeXt-Tiny + CBAM + Multi-Head Gated Attention MIL
    2. xray_only       : DenseNet121 + CLAHE + CBAM + Multi-Head Gated Attention MIL
    3. text_only       : Bio_ClinicalBERT + Token Attention Pooler
    4. full_multimodal : View-Level Cross-Modal Co-Attention + Gradient-Preserving
                         Modality Balancing (GPMB) + Gated Tri-Modal Synergy
"""

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .encoders import MultiViewVisionEncoder, ClinicalTextEncoder


def scale_with_full_gradient(
    features: torch.Tensor, scale_param: torch.Tensor
) -> torch.Tensor:
    """
    Scales forward activation amplitude by sigmoid(scale_param) while
    preserving 100% of backward gradient flow to prevent visual encoder starvation.
    """
    scale = torch.sigmoid(scale_param)
    return features * scale + (features - features.detach()) * (1.0 - scale)


class SafeCrossModalViewAttention(nn.Module):
    """
    Allows the clinical text representation (or cross-visual query) to attend
    directly over the individual image/radiograph views {v_1, ..., v_M} of a visit.
    FP16-safe and immune to NaN on padded views.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query_vec: torch.Tensor,
        view_tokens: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_vec   : [B, D]
        view_tokens : [B, V, D]
        view_mask   : [B, V] (True for valid views)
        """
        B, V, D = view_tokens.shape
        H = self.num_heads
        d_k = self.head_dim

        q = self.q_proj(query_vec).reshape(B, 1, H, d_k).transpose(1, 2)    # [B, H, 1, d_k]
        k = self.k_proj(view_tokens).reshape(B, V, H, d_k).transpose(1, 2)  # [B, H, V, d_k]
        v = self.v_proj(view_tokens).reshape(B, V, H, d_k).transpose(1, 2)  # [B, H, V, d_k]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale          # [B, H, 1, V]

        mask_exp = view_mask.unsqueeze(1).unsqueeze(2)                      # [B, 1, 1, V]
        has_any = view_mask.any(dim=1, keepdim=True).unsqueeze(1).unsqueeze(2)
        safe_mask = torch.where(has_any, mask_exp, torch.ones_like(mask_exp))

        scores = scores.masked_fill(~safe_mask, -1e4)
        attn = F.softmax(scores, dim=-1) * mask_exp.float()

        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, D)       # [B, D]
        return self.norm(query_vec + self.out_proj(context))


class MultimodalScenarioClassifier(nn.Module):
    """
    Unified classifier supporting image_only, xray_only, text_only, and full_multimodal.
    Can be initialized from the SSL v2 checkpoint (`best_ssl_model.pt`).
    """

    def __init__(
        self,
        modalities: tuple,
        num_labels: int = config.NUM_LABELS,
        embed_dim: int = config.SHARED_EMBED_DIM,
        dropout: float = config.DROPOUT,
        modality_dropout: float = config.MODALITY_DROPOUT,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.modality_dropout = float(modality_dropout)

        self.has_photo = "photograph" in self.modalities
        self.has_radio = "radiograph" in self.modalities
        self.has_text = "text" in self.modalities
        self.is_full_multimodal = (
            self.has_photo and self.has_radio and self.has_text
        )

        if self.has_photo:
            self.photo_encoder = MultiViewVisionEncoder(
                backbone_name=config.PHOTO_BACKBONE,
                pretrained=True,
                out_dim=embed_dim,
                dropout=dropout * 0.7,
                freeze_early_stages=config.FREEZE_PHOTO_STAGES,
            )

        if self.has_radio:
            self.radio_encoder = MultiViewVisionEncoder(
                backbone_name=config.RADIO_BACKBONE,
                pretrained=True,
                out_dim=embed_dim,
                dropout=dropout * 0.7,
                freeze_early_stages=config.FREEZE_RADIO_BLOCKS,
            )

        if self.has_text:
            self.text_encoder = ClinicalTextEncoder(
                model_name=config.TEXT_MODEL_NAME,
                out_dim=embed_dim,
                freeze_layers=config.FREEZE_TEXT_LAYERS,
                internal_dropout=0.20,
                dropout=dropout * 0.7,
            )

        if self.is_full_multimodal:
            # View-level cross-modal attention (Text queries Photo views & Radio views)
            self.text_to_photo_attn = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )
            self.text_to_radio_attn = SafeCrossModalViewAttention(
                dim=embed_dim, num_heads=4, dropout=0.20
            )

            # Gradient-Preserving Modality Balancing (GPMB) scales
            self.photo_scale = nn.Parameter(torch.full((1, embed_dim), -0.50))
            self.radio_scale = nn.Parameter(torch.full((1, embed_dim), -0.50))

            # Gated Multimodal Synergy Unit
            self.synergy_gate = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

            # Final fused dimension:
            # [Scaled Photo (256), Scaled Radio (256), Visual-Text Synergy (256),
            #  Text-Guided Photo (256), Text-Guided Radio (256), Raw Text (768)] = 2048
            fusion_in_dim = embed_dim * 5 + 768
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion_in_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_labels),
            )

        elif self.has_text and not self.has_photo and not self.has_radio:
            # text_only: combine 256-d SSL embedding + 768-d raw clinical pooling
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim + 768, num_labels),
            )

        else:
            # image_only or xray_only
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_labels),
            )

    def load_ssl_pretrained_weights(self, checkpoint_path: Path) -> bool:
        """
        Loads encoder weights from `best_ssl_model.pt` into the active branches.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(
                f"[Warning] SSL checkpoint not found at {checkpoint_path}. "
                "Using ImageNet / Bio_ClinicalBERT initialization."
            )
            return False

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        enc_state = ckpt.get("encoders_state_dict", ckpt)

        loaded_modules = []
        if self.has_photo:
            sub_dict = {
                k.replace("photo_encoder.", ""): v
                for k, v in enc_state.items()
                if k.startswith("photo_encoder.")
            }
            if sub_dict:
                self.photo_encoder.load_state_dict(sub_dict, strict=False)
                loaded_modules.append("photo_encoder")

        if self.has_radio:
            sub_dict = {
                k.replace("radio_encoder.", ""): v
                for k, v in enc_state.items()
                if k.startswith("radio_encoder.")
            }
            if sub_dict:
                self.radio_encoder.load_state_dict(sub_dict, strict=False)
                loaded_modules.append("radio_encoder")

        if self.has_text:
            sub_dict = {
                k.replace("text_encoder.", ""): v
                for k, v in enc_state.items()
                if k.startswith("text_encoder.")
            }
            if sub_dict:
                self.text_encoder.load_state_dict(sub_dict, strict=False)
                loaded_modules.append("text_encoder")

        print(
            f"[SSL Init] Successfully loaded SSL v2 weights for: {loaded_modules} "
            f"(from {checkpoint_path.name})"
        )
        return True

    def forward(
        self,
        photo_images: torch.Tensor | None = None,
        photo_view_mask: torch.Tensor | None = None,
        radio_images: torch.Tensor | None = None,
        radio_view_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        photo_out = None
        radio_out = None
        text_out = None

        if self.has_photo:
            photo_out = self.photo_encoder(photo_images, photo_view_mask)
        if self.has_radio:
            radio_out = self.radio_encoder(radio_images, radio_view_mask)
        if self.has_text:
            text_out = self.text_encoder(input_ids, attention_mask)

        # Single-Modality Scenarios
        if not self.is_full_multimodal:
            if self.has_photo:
                return self.classifier(photo_out["visit_embedding"])
            if self.has_radio:
                return self.classifier(radio_out["visit_embedding"])
            if self.has_text:
                fused_text = torch.cat(
                    [text_out["visit_embedding"], F.layer_norm(text_out["raw_text_feat"], (768,))],
                    dim=-1,
                )
                return self.classifier(fused_text)

        # Full Multimodal Scenario (Photograph + Radiograph + Clinical Text)
        p_visit = photo_out["visit_embedding"]    # [B, 256]
        r_visit = radio_out["visit_embedding"]    # [B, 256]
        t_visit = text_out["visit_embedding"]     # [B, 256]
        t_raw = F.layer_norm(text_out["raw_text_feat"], (768,))

        # 1. View-Level Cross-Modal Co-Attention
        t_guided_p = self.text_to_photo_attn(
            t_visit, photo_out["projected_views"], photo_view_mask
        )
        t_guided_r = self.text_to_radio_attn(
            t_visit, radio_out["projected_views"], radio_view_mask
        )

        # 2. Gradient-Preserving Modality Balancing (GPMB)
        p_scaled = scale_with_full_gradient(p_visit, self.photo_scale)
        r_scaled = scale_with_full_gradient(r_visit, self.radio_scale)

        # 3. Disjoint Per-Sample Modality Dropout during training
        if self.training and self.modality_dropout > 0.0:
            B = p_scaled.size(0)
            u = torch.rand(B, 1, device=p_scaled.device)
            drop_p = (u < self.modality_dropout).float()
            drop_r = (
                (u >= self.modality_dropout) & (u < 2.0 * self.modality_dropout)
            ).float()
            p_scaled = p_scaled * (1.0 - drop_p) + r_scaled * drop_p * 0.5
            r_scaled = r_scaled * (1.0 - drop_r) + p_scaled * drop_r * 0.5

        # 4. Tri-Modal Synergy Stream
        synergy = self.synergy_gate(torch.cat([p_scaled, r_scaled, t_visit], dim=-1))

        norm_factor = 1.0 / math.sqrt(5.0)
        fused = torch.cat(
            [
                p_scaled * norm_factor,
                r_scaled * norm_factor,
                synergy * norm_factor,
                t_guided_p * norm_factor,
                t_guided_r * norm_factor,
                t_raw,
            ],
            dim=-1,
        )
        return self.classifier(fused)