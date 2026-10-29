import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=8192, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # اصلاح این بخش برای جریان درست گرادیان در DINO Head
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer.weight.data.normal_(mean=0.0, std=0.01)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class BackboneWrapper(nn.Module):
    """ساخت بک‌بون استخراج ویژگی (سازگار با ابعاد ورودی متغیر ۲۲۴ و ۹۶)"""
    def __init__(self, arch="convnext_tiny", pretrained=True):
        super().__init__()
        if arch == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.convnext_tiny(weights=weights)
            self.features = net.features
            self.out_dim = 768
        elif arch == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.densenet121(weights=weights)
            self.features = net.features
            self.out_dim = 1024
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward(self, x):
        feat = self.features(x)
        if feat.dim() == 4:
            feat = F.relu(feat, inplace=False)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return feat


class MultiCropWrapper(nn.Module):
    """عبور دادن همزمان کراپ‌های Global (224x224) و Local (96x96) به صورت دسته‌بندی شده"""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        # گروه‌بندی تصاویر بر اساس سایز پیکسل (۲۲۴ در یک گروه، ۹۶ در گروه دیگر)
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output.append(_out)
            start_idx = end_idx
            
        return self.head(torch.cat(output))


def build_dino_models(arch="convnext_tiny", out_dim=8192):
    """ساخت جفت شبکه Student و Teacher"""
    student_backbone = BackboneWrapper(arch=arch, pretrained=True)
    teacher_backbone = BackboneWrapper(arch=arch, pretrained=True)
    embed_dim = student_backbone.out_dim

    student = MultiCropWrapper(student_backbone, DINOHead(embed_dim, out_dim))
    teacher = MultiCropWrapper(teacher_backbone, DINOHead(embed_dim, out_dim))

    # شبکه Teacher در ابتدا کپیِ دقیق Student است و گرادیان مستقیم نمی‌گیرد
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher