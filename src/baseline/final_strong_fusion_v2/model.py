"""Strong V2 residual multimodal fusion architecture.

Image branches reproduce the established strong V2 architecture:

ResNet50 + CBAM
+ global average pooling
+ mean-view aggregation
+ 2048 -> 256 -> 128 projection.

For image-only scenarios:

128 -> num_labels

For multimodal scenarios, the established modality representations
are preserved and only the fusion head is improved:

concatenated modality features
-> 1024 -> 256
-> LayerNorm + GELU
-> residual MLP
-> LayerNorm
-> 256 -> num_labels
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(
            self.avg_pool(x)
        )

        maximum = self.mlp(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg + maximum
        )

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
    def __init__(
        self,
        channels,
    ):
        super().__init__()

        self.channel_attention = (
            ChannelAttention(channels)
        )

        self.spatial_attention = (
            SpatialAttention()
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class StrongImageAggregator(nn.Module):
    """Established strong V2 image feature extractor."""

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

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                2048,
                256,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.30
            ),
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

            stacked = torch.stack(
                image_list
            )

            x = self.extract_feature_map(
                stacked
            )

            x = self.cbam(x)

            x = self.pool(x)

            x = x.flatten(
                start_dim=1
            )

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


class ResidualFusionBlock(nn.Module):
    """Small regularized residual refinement block."""

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                dim,
            ),

            nn.Dropout(
                dropout
            ),
        )

    def forward(self, x):
        residual = x

        x = self.norm(x)

        x = self.mlp(x)

        return residual + x


class ResidualMultimodalFusion(nn.Module):
    """Improved but deliberately lightweight multimodal fusion."""

    def __init__(
        self,
        input_dim,
        num_labels,
        fusion_dim=256,
        hidden_dim=128,
        dropout=0.30,
        residual_dropout=0.20,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                fusion_dim,
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.residual_block = (
            ResidualFusionBlock(
                dim=fusion_dim,
                hidden_dim=hidden_dim,
                dropout=residual_dropout,
            )
        )

        self.output_norm = nn.LayerNorm(
            fusion_dim
        )

        self.classifier = nn.Linear(
            fusion_dim,
            num_labels,
        )

    def forward(self, x):
        x = self.input_projection(x)

        x = self.residual_block(x)

        x = self.output_norm(x)

        return self.classifier(x)


class FinalStrongFusionV2(nn.Module):
    def __init__(
        self,
        modalities,
        text_model_name,
        num_labels,
        pretrained_image=True,
        freeze_image_encoder=False,
        fusion_dim=256,
        fusion_hidden_dim=128,
        fusion_dropout=0.30,
        fusion_residual_dropout=0.20,
    ):
        super().__init__()

        self.modalities = tuple(
            modalities
        )

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

        text_dim = 0

        if "text" in self.modalities:
            self.text_encoder = (
                AutoModel.from_pretrained(
                    text_model_name
                )
            )

            text_dim = (
                self.text_encoder.config.hidden_size
            )

        image_dim = 128

        input_dim = 0

        if "photograph" in self.modalities:
            input_dim += image_dim

        if "radiograph" in self.modalities:
            input_dim += image_dim

        input_dim += text_dim

        image_only = (
            len(self.modalities) == 1
            and (
                "photograph" in self.modalities
                or "radiograph" in self.modalities
            )
        )

        self.image_only = image_only

        if image_only:
            self.classifier = nn.Linear(
                image_dim,
                num_labels,
            )

            self.fusion = None

        else:
            self.fusion = (
                ResidualMultimodalFusion(
                    input_dim=input_dim,
                    num_labels=num_labels,
                    fusion_dim=fusion_dim,
                    hidden_dim=fusion_hidden_dim,
                    dropout=fusion_dropout,
                    residual_dropout=(
                        fusion_residual_dropout
                    ),
                )
            )

            # Keep a classifier attribute for compatibility
            # with generic training/checkpoint inspection code.
            self.classifier = self.fusion.classifier

    def forward(
        self,
        images=None,
        radiographs=None,
        input_ids=None,
        attention_mask=None,
    ):
        features = []

        if "photograph" in self.modalities:
            features.append(
                self.photograph_aggregator(
                    images
                )
            )

        if "radiograph" in self.modalities:
            features.append(
                self.radiograph_aggregator(
                    radiographs
                )
            )

        if "text" in self.modalities:
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            features.append(
                outputs.last_hidden_state[:, 0]
            )

        fused = torch.cat(
            features,
            dim=1,
        )

        if self.image_only:
            return self.classifier(
                fused
            )

        return self.fusion(
            fused
        )