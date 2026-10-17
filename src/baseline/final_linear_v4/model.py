import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


# ================================================================
# CBAM
# ================================================================

class ChannelAttention(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            1,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = (
            avg_attention
            + max_attention
        )

        return x * self.activation(
            attention
        )


class SpatialAttention(nn.Module):
    def __init__(
        self,
        kernel_size=7,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_attention = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_attention = torch.max(
            x,
            dim=1,
            keepdim=True,
        ).values

        attention = torch.cat(
            [
                avg_attention,
                max_attention,
            ],
            dim=1,
        )

        attention = self.conv(
            attention
        )

        return x * self.activation(
            attention
        )


class CBAM(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
        spatial_kernel_size=7,
    ):
        super().__init__()

        self.channel_attention = (
            ChannelAttention(
                channels=channels,
                reduction=reduction,
            )
        )

        self.spatial_attention = (
            SpatialAttention(
                kernel_size=spatial_kernel_size,
            )
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ================================================================
# Image Aggregator
# ================================================================

class StrongImageAggregator(nn.Module):
    def __init__(
        self,
        pretrained=True,
        freeze_encoder=True,
        dropout=0.50,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(
            weights=weights
        )

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.cbam = CBAM(
            channels=2048
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.25),
        )

        if freeze_encoder:
            for module in [
                self.layer0,
                self.layer1,
                self.layer2,
                self.layer3[:3],
            ]:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def encode_views(self, flat_images):
        x = self.layer0(flat_images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.cbam(x)

        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        return 0.5 * avg_feat + 0.5 * max_feat

    def forward(self, image_lists):
        view_counts = [len(images) for images in image_lists]
        if any(count == 0 for count in view_counts):
            raise RuntimeError(
                "Encountered a sample with zero images."
            )

        flat_images = torch.cat(
            [torch.stack(images, dim=0) for images in image_lists],
            dim=0,
        )

        flat_features = self.encode_views(flat_images)

        patient_views = torch.split(
            flat_features,
            view_counts,
            dim=0,
        )

        sample_features = []
        for views in patient_views:
            if views.size(0) == 1:
                sample_feature = views.squeeze(0)
            else:
                sample_feature = (
                    0.5 * views.mean(dim=0)
                    + 0.5 * views.max(dim=0).values
                )
            sample_features.append(sample_feature)

        stacked_patients = torch.stack(sample_features, dim=0)
        return self.projection(stacked_patients)

# ================================================================
# Final Linear V4 (With Gradient-Preserving Modality Balancing)
# ================================================================

def scale_with_full_gradient(features, scale_param):
    """
    Scales feature amplitude in the forward pass by `scale_param` to prevent
    initial noise injection into the text anchor, while preserving 100% of
    the gradient magnitude for the visual encoder in the backward pass
    (resolving Multimodal Gradient Starvation).
    """
    scale = torch.sigmoid(scale_param)  # bounded in (0, 1)
    return features * scale + (features - features.detach()) * (1.0 - scale)


class FinalLinearV4(nn.Module):
    """
    Regularized end-to-end multimodal model with:
    1. Gradient-Preserving Modality Balancing (GPMB)
    2. Zero-Mean Normalized Tri-Stream Visual Synergy (Photo + Radio + Max-Mean Synergy)
    3. Disjoint Per-Sample Modality Drop during training to prevent trimodal overfitting
    """

    def __init__(
        self,
        modalities,
        text_model_name,
        num_labels,
        pretrained_image=True,
        freeze_image_encoder=True,
        freeze_text_layers=3,
        text_internal_dropout=0.20,
        dropout=0.45,
        modality_dropout=0.15,
    ):
        super().__init__()

        self.modalities = tuple(modalities)
        self.num_labels = num_labels
        self.modality_dropout_prob = float(modality_dropout)

        self.has_text_and_vision = (
            "text" in self.modalities
            and len(self.modalities) > 1
        )
        self.has_both_visual = (
            "photograph" in self.modalities
            and "radiograph" in self.modalities
        )

        if "photograph" in self.modalities:
            self.photograph_aggregator = StrongImageAggregator(
                pretrained=pretrained_image,
                freeze_encoder=freeze_image_encoder,
                dropout=dropout,
            )
            if self.has_text_and_vision:
                self.photo_scale = nn.Parameter(
                    torch.full((1, 128), -0.85)
                )

        if "radiograph" in self.modalities:
            self.radiograph_aggregator = StrongImageAggregator(
                pretrained=pretrained_image,
                freeze_encoder=freeze_image_encoder,
                dropout=dropout,
            )
            if self.has_text_and_vision:
                self.radio_scale = nn.Parameter(
                    torch.full((1, 128), -0.85)
                )

        if "text" in self.modalities:
            text_config = AutoConfig.from_pretrained(
                text_model_name
            )
            if hasattr(text_config, "dropout"):
                text_config.dropout = float(text_internal_dropout)
            if hasattr(text_config, "attention_dropout"):
                text_config.attention_dropout = float(text_internal_dropout)

            self.text_encoder = AutoModel.from_pretrained(
                text_model_name,
                config=text_config,
            )

            if hasattr(self.text_encoder, "embeddings"):
                for param in self.text_encoder.embeddings.parameters():
                    param.requires_grad = False

            if (
                hasattr(self.text_encoder, "transformer")
                and freeze_text_layers > 0
            ):
                for layer in self.text_encoder.transformer.layer[:freeze_text_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

            self.text_norm = nn.LayerNorm(768)

        input_dim = 0
        if "text" in self.modalities:
            input_dim += 768

        if self.has_both_visual:
            # 3 streams of 128 dims: [Pure Photo (128), Pure Radio (128), Zero-Mean Synergy (128)]
            # Allows each of the 6 disease labels to weight Photo, Radio, and Synergy independently
            input_dim += 384
        else:
            if "photograph" in self.modalities:
                input_dim += 128
            if "radiograph" in self.modalities:
                input_dim += 128

        if input_dim <= 0:
            raise ValueError(
                "At least one modality must be enabled."
            )

        self.fusion_dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            input_dim,
            num_labels,
        )

    def forward(
        self,
        images=None,
        radiographs=None,
        input_ids=None,
        attention_mask=None,
    ):
        raw_photo = None
        raw_radio = None
        text_features = None

        if "photograph" in self.modalities:
            if images is None:
                raise RuntimeError("Images were not provided.")
            raw_photo = self.photograph_aggregator(images)

        if "radiograph" in self.modalities:
            if radiographs is None:
                raise RuntimeError("Radiographs were not provided.")
            raw_radio = self.radiograph_aggregator(radiographs)

        if "text" in self.modalities:
            if input_ids is None:
                raise RuntimeError("Input_ids were not provided.")
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_features = self.text_norm(
                outputs.last_hidden_state[:, 0]
            )
            text_features = self.fusion_dropout(text_features)

        features_to_cat = []

        if self.has_both_visual:
            # 1. Zero-Mean Normalized Max-Mean Synergy (removes the +0.564 threshold shift!)
            raw_synergy = F.layer_norm(
                torch.max(raw_photo, raw_radio)
                + 0.5 * (raw_photo + raw_radio),
                (128,),
            )

            if self.has_text_and_vision:
                photo_feat = scale_with_full_gradient(
                    raw_photo, self.photo_scale
                )
                radio_feat = scale_with_full_gradient(
                    raw_radio, self.radio_scale
                )
                synergy_scale = 0.5 * (
                    self.photo_scale + self.radio_scale
                )
                synergy_feat = scale_with_full_gradient(
                    raw_synergy, synergy_scale
                )
            else:
                photo_feat = raw_photo
                radio_feat = raw_radio
                synergy_feat = raw_synergy

            # 2. Disjoint Per-Sample Modality Drop during training (prevents 99% Train F1 overfitting)
            if self.training and self.modality_dropout_prob > 0.0:
                batch_size = photo_feat.size(0)
                rand_u = torch.rand(
                    batch_size, 1, device=photo_feat.device
                )
                drop_p = (
                    rand_u < self.modality_dropout_prob
                ).float()
                drop_r = (
                    (rand_u >= self.modality_dropout_prob)
                    & (rand_u < 2.0 * self.modality_dropout_prob)
                ).float()

                # When one visual modality is dropped for a sample, route the remaining one cleanly
                eff_photo = (
                    photo_feat * (1.0 - drop_p)
                    + radio_feat * drop_p * 0.5
                )
                eff_radio = (
                    radio_feat * (1.0 - drop_r)
                    + photo_feat * drop_r * 0.5
                )
                eff_synergy = (
                    synergy_feat * (1.0 - drop_p) * (1.0 - drop_r)
                    + photo_feat * drop_r
                    + radio_feat * drop_p
                )
            else:
                eff_photo = photo_feat
                eff_radio = radio_feat
                eff_synergy = synergy_feat

            # 3. Variance-preserving tri-stream concatenation (total visual variance = 1.0)
            norm_factor = 1.0 / math.sqrt(3.0)
            features_to_cat.extend(
                [
                    eff_photo * norm_factor,
                    eff_radio * norm_factor,
                    eff_synergy * norm_factor,
                ]
            )
        else:
            if raw_photo is not None:
                photo_feat = (
                    scale_with_full_gradient(
                        raw_photo, self.photo_scale
                    )
                    if self.has_text_and_vision
                    else raw_photo
                )
                features_to_cat.append(photo_feat)

            if raw_radio is not None:
                radio_feat = (
                    scale_with_full_gradient(
                        raw_radio, self.radio_scale
                    )
                    if self.has_text_and_vision
                    else raw_radio
                )
                features_to_cat.append(radio_feat)

        if text_features is not None:
            features_to_cat.append(text_features)

        fused = torch.cat(features_to_cat, dim=1)
        logits = self.classifier(fused)

        return logits