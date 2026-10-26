"""
Specialized Multimodal Encoders with Multi-Head Gated Attention MIL (ABMIL).

Backbones:
    - Intraoral Photographs : ConvNeXt-Tiny (768-d) + CBAM + Gated ABMIL
    - Dental Radiographs    : DenseNet121 (1024-d)  + CBAM + Gated ABMIL
    - Clinical Text         : Bio_ClinicalBERT (768-d) + Attention Token Pooler

Crucial Design for Dual-T4 (nn.DataParallel):
    All forward methods accept padded 5D tensors for visual modalities:
        images:    [B, V_max, 3, H, W]
        view_mask: [B, V_max] (True = valid view, False = padding)
    Only valid views (view_mask == True) are passed through the CNN backbone,
    saving ~45% VRAM and compute while remaining 100% compatible with DataParallel!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    densenet121,
    DenseNet121_Weights,
    resnet50,
    ResNet50_Weights,
)
from transformers import AutoConfig, AutoModel


# ============================================================
# Convolutional Block Attention Module (CBAM)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attn(self.channel_attn(x))


# ============================================================
# Multi-Head Gated Attention MIL Aggregator (No View Averaging!)
# ============================================================

class MultiHeadGatedAttentionMIL(nn.Module):
    """
    Aggregates a variable-size set of view embeddings {h_1, ..., h_V}
    into a unified visit representation without diluting localized lesions.

    Combines:
        1. Ilse et al. Gated Attention Pooling (learns which view has the pathology)
        2. Residual Max-Feature Routing (captures peak lesion activation across views)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        attn_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.view_norm = nn.LayerNorm(in_dim)

        # Gated Attention mechanism: tanh(V h) * sigmoid(U h)
        self.attn_V = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
        )
        self.attn_U = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Sigmoid(),
        )
        self.attn_weights = nn.Linear(attn_dim, num_heads)

        # Fuse multi-head attended features + peak max-pooled features across views
        self.head_projector = nn.Sequential(
            nn.Linear(in_dim * num_heads + in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

        # View-level projection to the shared space (for local view-to-text alignment)
        self.view_projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        view_features: torch.Tensor,
        view_mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        view_features : Tensor [B, V_max, in_dim]
        view_mask     : BoolTensor [B, V_max] (True for valid views)

        Returns
        -------
        visit_embedding : Tensor [B, out_dim]
        projected_views : Tensor [B, V_max, out_dim]
        attn_probs      : Tensor [B, num_heads, V_max]
        """
        B, V, D = view_features.shape
        normed = self.view_norm(view_features)
        projected_views = self.view_projector(normed)

        # Compute gated attention logits: [B, V, num_heads] -> [B, num_heads, V]
        gated = self.attn_V(normed) * self.attn_U(normed)
        logits = self.attn_weights(gated).transpose(1, 2)  # [B, H, V]

        # Mask out padded views
        mask_expanded = view_mask.unsqueeze(1)  # [B, 1, V]
        # Safe masking for samples that might have 0 views (when modality is missing)
        has_any_view = view_mask.any(dim=1, keepdim=True).unsqueeze(1)  # [B, 1, 1]
        safe_mask = torch.where(has_any_view, mask_expanded, torch.ones_like(mask_expanded))

        logits = logits.masked_fill(~safe_mask, -1e4)
        attn_probs = F.softmax(logits, dim=-1)  # [B, H, V]
        attn_probs = attn_probs * mask_expanded.float()

        # Multi-head weighted sum: [B, H, V] @ [B, V, D] -> [B, H, D] -> [B, H * D]
        attended = torch.bmm(attn_probs, normed).reshape(B, self.num_heads * D)

        # Max-pooling over valid views to preserve focal lesion spikes
        masked_for_max = normed.masked_fill(~ view_mask.unsqueeze(-1), -1e4)
        max_pooled = masked_for_max.max(dim=1).values
        max_pooled = torch.where(
            view_mask.any(dim=1, keepdim=True),
            max_pooled,
            torch.zeros_like(max_pooled),
        )

        combined = torch.cat([attended, max_pooled], dim=-1)
        visit_embedding = self.head_projector(combined)

        # Zero out samples that had no views at all
        valid_sample_mask = view_mask.any(dim=1, keepdim=True).float()
        visit_embedding = visit_embedding * valid_sample_mask
        projected_views = projected_views * view_mask.unsqueeze(-1).float()

        return visit_embedding, projected_views, attn_probs


# ============================================================
# Multi-View Visual Encoder (ConvNeXt-Tiny & DenseNet121)
# ============================================================

class MultiViewVisionEncoder(nn.Module):
    """
    Processes each image view independently through a specialized backbone
    (ConvNeXt-Tiny for Photographs, DenseNet121 for Radiographs) and
    aggregates all views of a visit via MultiHeadGatedAttentionMIL.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        out_dim: int = 256,
        dropout: float = 0.25,
        freeze_early_stages: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_dim = out_dim

        if backbone_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            net = convnext_tiny(weights=weights)
            self.spatial_extractor = net.features  # Outputs [N, 768, H/32, W/32]
            self.feature_dim = 768

            if freeze_early_stages > 0:
                # ConvNeXt features has 8 blocks (0..7); freeze early blocks
                for idx in range(min(freeze_early_stages * 2, len(self.spatial_extractor))):
                    for p in self.spatial_extractor[idx].parameters():
                        p.requires_grad = False

        elif backbone_name == "densenet121":
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            net = densenet121(weights=weights)
            self.spatial_extractor = net.features  # Outputs [N, 1024, H/32, W/32]
            self.feature_dim = 1024

            if freeze_early_stages > 0:
                freeze_names = ["conv0", "norm0", "denseblock1", "transition1"]
                if freeze_early_stages >= 2:
                    freeze_names.extend(["denseblock2", "transition2"])
                for name, module in self.spatial_extractor.named_children():
                    if name in freeze_names:
                        for p in module.parameters():
                            p.requires_grad = False

        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            self.spatial_extractor = nn.Sequential(*list(net.children())[:-2])
            self.feature_dim = 2048

        else:
            raise ValueError(f"Unsupported vision backbone: {backbone_name}")

        self.cbam = CBAMBlock(channels=self.feature_dim, reduction=16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.mil_aggregator = MultiHeadGatedAttentionMIL(
            in_dim=self.feature_dim,
            out_dim=out_dim,
            attn_dim=256,
            num_heads=4,
            dropout=dropout,
        )

    def encode_valid_views(self, flat_images: torch.Tensor) -> torch.Tensor:
        """
        Pass only valid (non-padded) 2D images through the CNN + CBAM.
        """
        feat_map = self.spatial_extractor(flat_images)
        if self.backbone_name == "densenet121":
            feat_map = F.relu(feat_map, inplace=False)
        feat_map = self.cbam(feat_map)
        avg_f = self.avg_pool(feat_map).flatten(1)
        max_f = self.max_pool(feat_map).flatten(1)
        return 0.5 * avg_f + 0.5 * max_f

    def forward(
        self,
        padded_images: torch.Tensor,
        view_mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        padded_images : Tensor [B, V_max, 3, H, W]
        view_mask     : BoolTensor [B, V_max]

        Returns
        -------
        visit_embedding : Tensor [B, out_dim]
        projected_views : Tensor [B, V_max, out_dim]
        raw_view_feats  : Tensor [B, V_max, feature_dim]
        attn_probs      : Tensor [B, num_heads, V_max]
        """
        B, V, C, H, W = padded_images.shape
        device = padded_images.device

        flat_mask = view_mask.reshape(-1)
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)

        if valid_indices.numel() == 0:
            # Handle edge case where a mini-batch slice on one GPU has no valid views
            dummy = padded_images.reshape(-1, C, H, W)[:1]
            dummy_feat = self.encode_valid_views(dummy) * 0.0
            raw_view_feats = dummy_feat.new_zeros((B, V, self.feature_dim))
        else:
            flat_valid_imgs = padded_images.reshape(B * V, C, H, W).index_select(
                0, valid_indices
            )
            valid_feats = self.encode_valid_views(flat_valid_imgs)

            raw_flat = valid_feats.new_zeros((B * V, self.feature_dim))
            raw_flat.index_copy_(0, valid_indices, valid_feats)
            raw_view_feats = raw_flat.reshape(B, V, self.feature_dim)

        visit_embedding, projected_views, attn_probs = self.mil_aggregator(
            raw_view_feats,
            view_mask,
        )

        return {
            "visit_embedding": visit_embedding,     # [B, out_dim]
            "projected_views": projected_views,     # [B, V, out_dim]
            "raw_view_feats": raw_view_feats,       # [B, V, feature_dim]
            "attn_probs": attn_probs,               # [B, H, V]
        }


# ============================================================
# Clinical Text Encoder (Bio_ClinicalBERT + Attention Pooler)
# ============================================================

class ClinicalTextEncoder(nn.Module):
    """
    Transformer encoder for dental clinical notes using Bio_ClinicalBERT.
    Combines the [CLS] token with attention-weighted token pooling over all
    clinical examination tokens.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        out_dim: int = 256,
        freeze_layers: int = 4,
        internal_dropout: float = 0.20,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_dim = out_dim

        cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(cfg, "hidden_dropout_prob"):
            cfg.hidden_dropout_prob = float(internal_dropout)
        if hasattr(cfg, "attention_probs_dropout_prob"):
            cfg.attention_probs_dropout_prob = float(internal_dropout)
        if hasattr(cfg, "dropout"):
            cfg.dropout = float(internal_dropout)

        self.transformer = AutoModel.from_pretrained(model_name, config=cfg)
        self.hidden_size = cfg.hidden_size  # 768

        # Freeze embeddings and bottom transformer layers if requested
        if freeze_layers > 0:
            if hasattr(self.transformer, "embeddings"):
                for p in self.transformer.embeddings.parameters():
                    p.requires_grad = False

            if hasattr(self.transformer, "encoder") and hasattr(
                self.transformer.encoder, "layer"
            ):
                for layer in self.transformer.encoder.layer[:freeze_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False
            elif hasattr(self.transformer, "transformer") and hasattr(
                self.transformer.transformer, "layer"
            ):
                for layer in self.transformer.transformer.layer[:freeze_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False

        # Token-level attention pooling to complement [CLS]
        self.token_attn = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        seq_hidden = outputs.last_hidden_state  # [B, L, 768]
        cls_feat = seq_hidden[:, 0]             # [B, 768]

        # Attention pooling over valid tokens
        scores = self.token_attn(seq_hidden).squeeze(-1)  # [B, L]
        mask_bool = attention_mask.bool()
        scores = scores.masked_fill(~mask_bool, -1e4)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        attn_pooled = torch.sum(seq_hidden * weights, dim=1)  # [B, 768]

        raw_text_feat = 0.5 * cls_feat + 0.5 * attn_pooled   # [B, 768]
        concat_feat = torch.cat([cls_feat, attn_pooled], dim=-1)  # [B, 1536]
        visit_embedding = self.projector(concat_feat)        # [B, out_dim]

        return {
            "visit_embedding": visit_embedding,  # [B, out_dim]
            "raw_text_feat": raw_text_feat,      # [B, 768]
            "token_features": seq_hidden,        # [B, L, 768]
        }