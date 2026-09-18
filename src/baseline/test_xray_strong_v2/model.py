"""Maximum-strength ResNet50 + CBAM X-ray classifier."""

import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
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

        max_value = self.mlp(
            self.max_pool(x)
        )

        return x * self.sigmoid(
            avg + max_value
        )


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


class MaxStrengthXrayResNet50(nn.Module):
    def __init__(
        self,
        num_labels,
        pretrained=True,
        dropout=0.5,
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
            1
        )

        self.classifier = nn.Sequential(
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

            nn.Linear(
                128,
                num_labels,
            ),
        )

    def extract_feature_map(
        self,
        x,
    ):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(
        self,
        images,
    ):
        feature_maps = []

        for image_list in images:
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

            feature_maps.append(
                x
            )

        features = torch.cat(
            feature_maps,
            dim=0,
        )

        return self.classifier(
            features
        )