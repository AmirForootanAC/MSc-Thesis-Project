import torch
import torch.nn as nn
from transformers import AutoModel
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
        freeze_encoder=False,
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

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.50),

            nn.Linear(
                2048,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.50),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),
        )

        if freeze_encoder:
            for module in [
                self.layer0,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
            ]:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def encode_single(self, image):
        x = self.layer0(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = self.projection(x)

        return x

    def forward(self, image_lists):
        sample_features = []

        for images in image_lists:
            if len(images) == 0:
                raise RuntimeError(
                    "Encountered a sample with zero images."
                )

            view_features = []

            for image in images:
                view_features.append(
                    self.encode_single(
                        image.unsqueeze(0)
                    ).squeeze(0)
                )

            stacked = torch.stack(
                view_features,
                dim=0,
            )

            sample_feature = stacked.mean(
                dim=0
            )

            sample_features.append(
                sample_feature
            )

        return torch.stack(
            sample_features,
            dim=0,
        )


# ================================================================
# Final Linear V3
# ================================================================

class FinalLinearV3(nn.Module):
    """
    End-to-end multimodal model with a strict
    linear fusion classifier.

    Photograph:
        ResNet50 + CBAM -> 128

    Radiograph:
        ResNet50 + CBAM -> 128

    Text:
        DistilBERT -> 768

    Fusion:
        Concatenation -> 1024

    Classifier:
        Linear(1024 -> 6)
    """

    def __init__(
        self,
        modalities,
        text_model_name,
        num_labels,
        pretrained_image=True,
        freeze_image_encoder=False,
    ):
        super().__init__()

        self.modalities = tuple(
            modalities
        )

        self.num_labels = num_labels

        if "photograph" in self.modalities:
            self.photograph_aggregator = (
                StrongImageAggregator(
                    pretrained=pretrained_image,
                    freeze_encoder=(
                        freeze_image_encoder
                    ),
                )
            )

        if "radiograph" in self.modalities:
            self.radiograph_aggregator = (
                StrongImageAggregator(
                    pretrained=pretrained_image,
                    freeze_encoder=(
                        freeze_image_encoder
                    ),
                )
            )

        if "text" in self.modalities:
            self.text_encoder = AutoModel.from_pretrained(
                text_model_name
            )

        input_dim = 0

        if "photograph" in self.modalities:
            input_dim += 128

        if "radiograph" in self.modalities:
            input_dim += 128

        if "text" in self.modalities:
            input_dim += 768

        if input_dim <= 0:
            raise ValueError(
                "At least one modality must be enabled."
            )

        # ========================================================
        # IMPORTANT:
        # Strict linear fusion.
        # No hidden layer.
        # No activation.
        # No dropout.
        # ========================================================

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
        features = []

        if "photograph" in self.modalities:
            if images is None:
                raise RuntimeError(
                    "Photograph modality is enabled "
                    "but images were not provided."
                )

            photograph_features = (
                self.photograph_aggregator(
                    images
                )
            )

            features.append(
                photograph_features
            )

        if "radiograph" in self.modalities:
            if radiographs is None:
                raise RuntimeError(
                    "Radiograph modality is enabled "
                    "but radiographs were not provided."
                )

            radiograph_features = (
                self.radiograph_aggregator(
                    radiographs
                )
            )

            features.append(
                radiograph_features
            )

        if "text" in self.modalities:
            if input_ids is None:
                raise RuntimeError(
                    "Text modality is enabled "
                    "but input_ids were not provided."
                )

            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            text_features = (
                outputs.last_hidden_state[:, 0]
            )

            features.append(
                text_features
            )

        fused = torch.cat(
            features,
            dim=1,
        )

        logits = self.classifier(
            fused
        )

        return logits