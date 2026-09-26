"""Experimental strong multimodal model with label-aware gated fusion.

Important:
- Photograph-only architecture is unchanged.
- Radiograph-only architecture is unchanged.
- Text-only architecture is unchanged.
- Only multimodal fusion is replaced.

The new multimodal fusion:
    modality-specific projection
        -> common 256-D space
        -> label-aware modality gates
        -> weighted modality fusion
        -> residual fusion MLP
        -> label-specific classification heads
"""

import torch
import torch.nn as nn

from transformers import AutoModel
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


# ---------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        maximum = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg + maximum)

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        maximum = torch.max(
            x,
            dim=1,
            keepdim=True,
        ).values

        attention = torch.cat(
            [avg, maximum],
            dim=1,
        )

        attention = self.sigmoid(
            self.conv(attention)
        )

        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


# ---------------------------------------------------------------------
# Strong image encoder
# ---------------------------------------------------------------------


class StrongImageAggregator(nn.Module):
    """Strong V2 ResNet50 + CBAM + mean-view feature extractor."""

    def __init__(
        self,
        pretrained=True,
        dropout=0.50,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(weights=weights)

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

        self.cbam = CBAM(channels=2048)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(dropout),

            nn.Linear(
                2048,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),
        )

    def extract_feature_map(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, image_lists):
        features = []

        for image_list in image_lists:

            if len(image_list) == 0:
                raise RuntimeError(
                    "Encountered a sample with zero images."
                )

            stacked = torch.stack(image_list)

            x = self.extract_feature_map(stacked)

            x = self.cbam(x)

            x = self.pool(x)

            x = x.flatten(start_dim=1)

            # Mean aggregation across available views.
            x = torch.mean(
                x,
                dim=0,
                keepdim=True,
            )

            x = self.projection(x)

            features.append(x)

        return torch.cat(
            features,
            dim=0,
        )


# ---------------------------------------------------------------------
# New label-aware gated multimodal fusion
# ---------------------------------------------------------------------


class LabelAwareGatedFusion(nn.Module):
    """Label-aware gated fusion in a shared representation space.

    Each modality is first projected into the same dimensional space.

    For every modality and every target label, a gate score is generated.

    Softmax is applied across modalities:

        gate(modality | label)

    This allows, for example, caries to rely more heavily on one modality
    while another label can rely more heavily on another modality.

    The resulting label-specific fused representations are processed by
    a small residual fusion network and six independent classification
    heads.
    """

    def __init__(
        self,
        modality_dims,
        num_labels,
        fusion_dim=256,
        gate_hidden_dim=64,
        projection_dropout=0.10,
        fusion_dropout=0.35,
        head_dropout=0.20,
    ):
        super().__init__()

        self.modality_names = tuple(
            modality_dims.keys()
        )

        self.num_labels = num_labels
        self.fusion_dim = fusion_dim

        # -------------------------------------------------------------
        # Modality projections
        # -------------------------------------------------------------

        self.projections = nn.ModuleDict()

        for modality, input_dim in modality_dims.items():

            self.projections[modality] = nn.Sequential(
                nn.Linear(
                    input_dim,
                    fusion_dim,
                ),

                nn.LayerNorm(fusion_dim),

                nn.GELU(),

                nn.Dropout(
                    projection_dropout
                ),
            )

        # -------------------------------------------------------------
        # Label-aware modality gates
        # -------------------------------------------------------------

        self.gates = nn.ModuleDict()

        for modality in self.modality_names:

            gate = nn.Sequential(
                nn.Linear(
                    fusion_dim,
                    gate_hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    projection_dropout
                ),

                nn.Linear(
                    gate_hidden_dim,
                    num_labels,
                ),
            )

            # Start with approximately uniform modality weighting.
            # The model then learns the modality importance.
            nn.init.zeros_(
                gate[-1].weight
            )

            nn.init.zeros_(
                gate[-1].bias
            )

            self.gates[modality] = gate

        # -------------------------------------------------------------
        # Fusion refinement
        # -------------------------------------------------------------

        self.pre_fusion_norm = nn.LayerNorm(
            fusion_dim
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(
                fusion_dim,
                fusion_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                fusion_dropout
            ),

            nn.Linear(
                fusion_dim,
                fusion_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                fusion_dropout
            ),
        )

        self.post_fusion_norm = nn.LayerNorm(
            fusion_dim
        )

        # -------------------------------------------------------------
        # Label-specific heads
        # -------------------------------------------------------------

        self.label_heads = nn.ModuleList()

        for _ in range(num_labels):

            self.label_heads.append(
                nn.Sequential(
                    nn.Linear(
                        fusion_dim,
                        64,
                    ),

                    nn.GELU(),

                    nn.Dropout(
                        head_dropout
                    ),

                    nn.Linear(
                        64,
                        1,
                    ),
                )
            )

    def forward(self, features):
        projected = []
        gate_logits = []

        for modality in self.modality_names:

            z = self.projections[modality](
                features[modality]
            )

            projected.append(z)

            logits = self.gates[modality](z)

            gate_logits.append(logits)

        # -------------------------------------------------------------
        # Shapes
        #
        # projected:
        #     M x [B, D]
        #
        # gate_logits:
        #     M x [B, L]
        #
        # after stack:
        #     [B, M, L]
        # -------------------------------------------------------------

        gate_logits = torch.stack(
            gate_logits,
            dim=1,
        )

        # Normalize modality importance for each label.
        gates = torch.softmax(
            gate_logits,
            dim=1,
        )

        stacked = torch.stack(
            projected,
            dim=1,
        )
        # [B, M, D]

        # -------------------------------------------------------------
        # Label-specific weighted fusion
        #
        # gates:
        #     [B, M, L]
        #
        # stacked:
        #     [B, M, D]
        #
        # result:
        #     [B, L, D]
        # -------------------------------------------------------------

        fused = torch.sum(
            gates.permute(
                0,
                2,
                1,
            ).unsqueeze(-1)
            * stacked.unsqueeze(1),
            dim=2,
        )

        # -------------------------------------------------------------
        # Small residual shared representation.
        #
        # This prevents the gate mechanism from completely discarding
        # information from modalities when training starts.
        # -------------------------------------------------------------

        modality_mean = torch.mean(
            stacked,
            dim=1,
            keepdim=True,
        )

        fused = fused + (
            0.10 * modality_mean
        )

        # -------------------------------------------------------------
        # Residual fusion refinement
        # -------------------------------------------------------------

        normalized = self.pre_fusion_norm(
            fused
        )

        refined = self.fusion_mlp(
            normalized
        )

        fused = self.post_fusion_norm(
            fused + refined
        )

        # -------------------------------------------------------------
        # Label-specific prediction
        # -------------------------------------------------------------

        outputs = []

        for label_index, head in enumerate(
            self.label_heads
        ):
            label_feature = fused[
                :,
                label_index,
                :,
            ]

            label_logit = head(
                label_feature
            )

            outputs.append(label_logit)

        return torch.cat(
            outputs,
            dim=1,
        )


# ---------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------


class FinalStrongV2(nn.Module):
    """Experimental strong V2 model with new multimodal fusion."""

    def __init__(
        self,
        modalities,
        text_model_name,
        num_labels,
        pretrained_image=True,
        freeze_image_encoder=False,
    ):
        super().__init__()

        self.modalities = tuple(modalities)

        # -------------------------------------------------------------
        # Image branches
        # -------------------------------------------------------------

        if "photograph" in self.modalities:

            self.photograph_aggregator = (
                StrongImageAggregator(
                    pretrained=pretrained_image,
                    dropout=0.50,
                )
            )

        if "radiograph" in self.modalities:

            self.radiograph_aggregator = (
                StrongImageAggregator(
                    pretrained=pretrained_image,
                    dropout=0.50,
                )
            )

        # -------------------------------------------------------------
        # Optional image freezing
        # -------------------------------------------------------------

        if freeze_image_encoder:

            for name in (
                "photograph_aggregator",
                "radiograph_aggregator",
            ):

                if hasattr(self, name):

                    module = getattr(
                        self,
                        name,
                    )

                    for parameter in module.parameters():
                        parameter.requires_grad = False

        # -------------------------------------------------------------
        # Text
        # -------------------------------------------------------------

        text_dim = 0

        if "text" in self.modalities:

            self.text_encoder = AutoModel.from_pretrained(
                text_model_name
            )

            text_dim = (
                self.text_encoder.config.hidden_size
            )

        # -------------------------------------------------------------
        # Image-only scenarios remain EXACTLY the same.
        # -------------------------------------------------------------

        image_dim = 128

        image_only = (
            len(self.modalities) == 1
            and (
                "photograph" in self.modalities
                or "radiograph" in self.modalities
            )
        )

        if image_only:

            self.classifier = nn.Linear(
                image_dim,
                num_labels,
            )

            self.multimodal_fusion = None

        # -------------------------------------------------------------
        # Text-only remains the previous architecture.
        # -------------------------------------------------------------

        elif self.modalities == ("text",):

            self.classifier = nn.Sequential(
                nn.Linear(
                    text_dim,
                    256,
                ),

                nn.ReLU(inplace=True),

                nn.Dropout(0.30),

                nn.Linear(
                    256,
                    num_labels,
                ),
            )

            self.multimodal_fusion = None

        # -------------------------------------------------------------
        # NEW MULTIMODAL FUSION
        # -------------------------------------------------------------

        else:

            modality_dims = {}

            if "photograph" in self.modalities:
                modality_dims["photograph"] = 128

            if "radiograph" in self.modalities:
                modality_dims["radiograph"] = 128

            if "text" in self.modalities:
                modality_dims["text"] = text_dim

            self.multimodal_fusion = (
                LabelAwareGatedFusion(
                    modality_dims=modality_dims,
                    num_labels=num_labels,
                    fusion_dim=256,
                    gate_hidden_dim=64,
                    projection_dropout=0.10,
                    fusion_dropout=0.35,
                    head_dropout=0.20,
                )
            )

            # IMPORTANT:
            #
            # The training code already places model.classifier
            # parameters in the classifier/head LR group.
            #
            # Therefore we expose the new fusion module as classifier
            # so the existing optimizer automatically trains all
            # fusion parameters with CLASSIFIER_LR.
            self.classifier = self.multimodal_fusion

    def forward(
        self,
        images=None,
        radiographs=None,
        input_ids=None,
        attention_mask=None,
    ):
        # -------------------------------------------------------------
        # Image representation
        # -------------------------------------------------------------

        features = {}

        if "photograph" in self.modalities:

            features["photograph"] = (
                self.photograph_aggregator(
                    images
                )
            )

        if "radiograph" in self.modalities:

            features["radiograph"] = (
                self.radiograph_aggregator(
                    radiographs
                )
            )

        # -------------------------------------------------------------
        # Text representation
        # -------------------------------------------------------------

        if "text" in self.modalities:

            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            features["text"] = (
                outputs.last_hidden_state[:, 0]
            )

        # -------------------------------------------------------------
        # New multimodal fusion
        # -------------------------------------------------------------

        if self.multimodal_fusion is not None:

            return self.multimodal_fusion(
                features
            )

        # -------------------------------------------------------------
        # Single modality
        # -------------------------------------------------------------

        if self.modalities == ("text",):

            return self.classifier(
                features["text"]
            )

        if self.modalities == ("photograph",):

            return self.classifier(
                features["photograph"]
            )

        if self.modalities == ("radiograph",):

            return self.classifier(
                features["radiograph"]
            )

        raise RuntimeError(
            f"Unsupported modality configuration: "
            f"{self.modalities}"
        )