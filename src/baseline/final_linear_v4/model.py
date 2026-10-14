import torch
import torch.nn as nn
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
# Final Linear V4
# ================================================================

# ================================================================
# Cross-Modal Gated & Disease-Aware Fusion Head (CMG-DAF)
# ================================================================

class DiseaseAwareGatedFusionHead(nn.Module):
    """
    Lightweight Cross-Modal Gated & Disease-Aware Fusion Head.
    1. Filters modality features per-patient via Cross-Modal Gated Units (GMU).
    2. Computes synergistic cross-modal interaction features.
    3. Routes modality contributions per-disease (6 labels) around the anchor modality.
    """

    def __init__(
        self,
        modalities,
        modality_dims,
        num_labels,
        proj_dim=128,
        dropout=0.45,
    ):
        super().__init__()

        self.modalities = tuple(modalities)
        self.modality_dims = list(modality_dims)
        self.num_modalities = len(self.modalities)
        self.num_labels = num_labels

        # تعیین مدالیته لنگر (Anchor): اگر متن حضور دارد متن، در غیر این صورت مدالیته اول
        self.anchor_idx = (
            self.modalities.index("text")
            if "text" in self.modalities
            else 0
        )

        # 1. تصویرسازی هر مدالیته به فضای مشترک 128 بعدی برای استخراج زمینه متقاطع (Context)
        self.align_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                )
                for dim in self.modality_dims
            ]
        )

        joint_context_dim = proj_dim * self.num_modalities

        # 2. گیت فیلترکننده ویژگی در سطح بیمار (Cross-Modal Feature Gate - GMU)
        self.feature_gates = nn.ModuleList()
        for idx, dim in enumerate(self.modality_dims):
            gate_linear = nn.Linear(joint_context_dim, dim)
            # مقداردهی اولیه بایاس گیت: باز بودن کامل مسیر لنگر در شروع آموزش
            bias_init = 2.0 if idx == self.anchor_idx else 0.0
            nn.init.constant_(gate_linear.bias, bias_init)
            self.feature_gates.append(gate_linear)

        # 3. شاخه استخراج تعامل غیرخطی بین مدالیته‌ها (Cross-Modal Synergy Branch)
        self.cross_interaction = nn.Sequential(
            nn.Linear(joint_context_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 4. هدهای پیش‌بینی اختصاصی هر مدالیته + هد تعامل متقاطع
        self.modality_classifiers = nn.ModuleList(
            [
                nn.Linear(dim, num_labels)
                for dim in self.modality_dims
            ]
        )
        self.cross_classifier = nn.Linear(proj_dim, num_labels)

        # 5. مسیردهنده آگاه از نوع بیماری (Disease-Specific Modality Gate)
        # به ازای هر مدالیته فرعی + شاخه تعاملی، یک وزن به ازای هر یک از 6 بیماری تولید می‌کند
        self.num_aux_branches = self.num_modalities  # (M - 1) aux modalities + 1 cross branch
        self.disease_gate_net = nn.Linear(
            joint_context_dim,
            self.num_aux_branches * num_labels,
        )
        # مقداردهی اولیه ملایم (-0.5 -> sigmoid ≈ 0.38) برای جلوگیری از شوک اولیه به مدالیته لنگر
        nn.init.normal_(self.disease_gate_net.weight, std=0.01)
        nn.init.constant_(self.disease_gate_net.bias, -0.5)

    def forward(self, fused_features):
        # تفکیک تانسور ورودی به بردارهای هر مدالیته
        modality_feats = torch.split(
            fused_features,
            self.modality_dims,
            dim=1,
        )

        # هم‌ترازسازی در فضای مشترک و ساخت بردار زمینه چندوجهی (Joint Context)
        aligned_feats = [
            proj(feat)
            for proj, feat in zip(self.align_projections, modality_feats)
        ]
        joint_context = torch.cat(aligned_feats, dim=1)

        # اعمال گیت متقاطع روی ویژگی‌های هر مدالیته
        gated_feats = []
        for idx, feat in enumerate(modality_feats):
            gate = torch.sigmoid(self.feature_gates[idx](joint_context))
            gated_feats.append(feat * gate)

        # محاسبه Logit پایه از مدالیته لنگر (مثلاً متن) بدون تضعیف دامنه
        anchor_logits = self.modality_classifiers[self.anchor_idx](
            gated_feats[self.anchor_idx]
        )

        # جمع‌آوری Logitهای کمکی از سایر مدالیته‌ها و شاخه تعامل متقاطع
        aux_logits_list = []
        for idx, feat in enumerate(gated_feats):
            if idx != self.anchor_idx:
                aux_logits_list.append(
                    self.modality_classifiers[idx](feat)
                )

        cross_feat = self.cross_interaction(joint_context)
        cross_logits = self.cross_classifier(cross_feat)
        aux_logits_list.append(cross_logits)

        # تجمیع نهایی با وزن‌دهی اختصاصی به ازای هر بیماری (Disease-Aware Routing)
        # ابعاد aux_stack: [Batch, Num_Aux_Branches, Num_Labels]
        aux_stack = torch.stack(aux_logits_list, dim=1)

        disease_gates = torch.sigmoid(
            self.disease_gate_net(joint_context)
        ).view(-1, self.num_aux_branches, self.num_labels)

        final_logits = anchor_logits + torch.sum(
            disease_gates * aux_stack,
            dim=1,
        )

        return final_logits


# ================================================================
# Final Linear / Gated-Fusion V4
# ================================================================

class FinalLinearV4(nn.Module):
    """
    Regularized end-to-end multimodal model:
    - Uses strict linear classifier for unimodal scenarios (preserving unimodal baselines).
    - Uses Cross-Modal Gated & Disease-Aware Fusion (CMG-DAF) for multimodal scenarios.
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
        modality_dropout=0.0,
    ):
        super().__init__()

        self.modalities = tuple(modalities)
        self.num_labels = num_labels

        modality_dims = []
        active_order = []

        if "photograph" in self.modalities:
            self.photograph_aggregator = StrongImageAggregator(
                pretrained=pretrained_image,
                freeze_encoder=freeze_image_encoder,
                dropout=dropout,
            )
            modality_dims.append(128)
            active_order.append("photograph")

        if "radiograph" in self.modalities:
            self.radiograph_aggregator = StrongImageAggregator(
                pretrained=pretrained_image,
                freeze_encoder=freeze_image_encoder,
                dropout=dropout,
            )
            modality_dims.append(128)
            active_order.append("radiograph")

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
            modality_dims.append(768)
            active_order.append("text")

        input_dim = sum(modality_dims)

        if input_dim <= 0:
            raise ValueError(
                "At least one modality must be enabled."
            )

        self.fusion_dropout = nn.Dropout(dropout)

        # در حالت تک‌وجهی دقیقاً همان لایه خطی سابق، و در حالت چندوجهی ماژول فیوژن هوشمند
        if len(active_order) == 1:
            self.classifier = nn.Linear(
                input_dim,
                num_labels,
            )
        else:
            self.classifier = DiseaseAwareGatedFusionHead(
                modalities=active_order,
                modality_dims=modality_dims,
                num_labels=num_labels,
                proj_dim=128,
                dropout=dropout,
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

            photograph_features = self.photograph_aggregator(
                images
            )
            features.append(photograph_features)

        if "radiograph" in self.modalities:
            if radiographs is None:
                raise RuntimeError(
                    "Radiograph modality is enabled "
                    "but radiographs were not provided."
                )

            radiograph_features = self.radiograph_aggregator(
                radiographs
            )
            features.append(radiograph_features)

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

            text_features = self.text_norm(
                outputs.last_hidden_state[:, 0]
            )
            text_features = self.fusion_dropout(text_features)

            features.append(text_features)

        fused = torch.cat(
            features,
            dim=1,
        )

        logits = self.classifier(fused)

        return logits