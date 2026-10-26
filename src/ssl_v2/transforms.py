"""
Modality-Specific Image Transformations for SSL v2 & Fusion v2.

Includes:
1. Pure-NumPy/PIL Vectorized CLAHE (Contrast Limited Adaptive Histogram Equalization)
   for Dental Radiographs — zero OpenCV (cv2) dependency and zero IDE warnings.
2. Multi-View SSL & Downstream Augmentations for Photographs and Radiographs.
"""

import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DentalRadiographCLAHE:
    """
    Pure NumPy/PIL Contrast Limited Adaptive Histogram Equalization (CLAHE)
    with bilinear tile interpolation on the luminance (Y) channel.
    Enhances enamel-dentin caries boundaries and periapical radiolucency
    without requiring OpenCV (cv2).
    """

    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8):
        self.clip_limit = float(clip_limit)
        self.grid_size = int(grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        ycbcr = img.convert("YCbCr")
        y_chan, cb_chan, cr_chan = ycbcr.split()
        y_arr = np.asarray(y_chan, dtype=np.uint8)
        orig_h, orig_w = y_arr.shape

        g = self.grid_size
        pad_h = (g - (orig_h % g)) % g
        pad_w = (g - (orig_w % g)) % g

        if pad_h > 0 or pad_w > 0:
            y_padded = np.pad(y_arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            y_padded = y_arr

        h_pad, w_pad = y_padded.shape
        th, tw = h_pad // g, w_pad // g
        num_pixels = th * tw
        limit = max(int(self.clip_limit * (num_pixels / 256.0)), 1)

        # Reshape into (g * g, num_pixels) and compute histograms across all tiles
        flat_tiles = (
            y_padded.reshape(g, th, g, tw)
            .swapaxes(1, 2)
            .reshape(g * g, num_pixels)
        )

        hists = np.zeros((g * g, 256), dtype=np.int32)
        row_idx = np.arange(g * g)[:, None]
        np.add.at(hists, (row_idx, flat_tiles), 1)

        excess = np.maximum(hists - limit, 0).sum(axis=1, keepdims=True)
        clipped = np.minimum(hists, limit) + (excess // 256)
        cdfs = np.cumsum(clipped, axis=1)
        cdf_mins = cdfs[:, 0:1]
        denoms = np.maximum(cdfs[:, -1:] - cdf_mins, 1)

        luts = (
            np.clip((cdfs - cdf_mins) * 255.0 / denoms, 0.0, 255.0)
            .astype(np.float32)
            .reshape(g, g, 256)
        )

        # Bilinear interpolation across tile centers to eliminate block boundaries
        y_coords = (np.arange(h_pad, dtype=np.float32) + 0.5) / th - 0.5
        x_coords = (np.arange(w_pad, dtype=np.float32) + 0.5) / tw - 0.5

        y0 = np.clip(np.floor(y_coords).astype(np.int32), 0, g - 1)
        y1 = np.clip(y0 + 1, 0, g - 1)
        wy = np.clip(y_coords - y0, 0.0, 1.0)[:, None]

        x0 = np.clip(np.floor(x_coords).astype(np.int32), 0, g - 1)
        x1 = np.clip(x0 + 1, 0, g - 1)
        wx = np.clip(x_coords - x0, 0.0, 1.0)[None, :]

        v00 = luts[y0[:, None], x0[None, :], y_padded]
        v01 = luts[y0[:, None], x1[None, :], y_padded]
        v10 = luts[y1[:, None], x0[None, :], y_padded]
        v11 = luts[y1[:, None], x1[None, :], y_padded]

        top = v00 * (1.0 - wx) + v01 * wx
        bot = v10 * (1.0 - wx) + v11 * wx
        out_y = np.clip(top * (1.0 - wy) + bot * wy, 0.0, 255.0).astype(np.uint8)
        out_y = out_y[:orig_h, :orig_w]

        enhanced_y = Image.fromarray(out_y, mode="L")
        return Image.merge("YCbCr", (enhanced_y, cb_chan, cr_chan)).convert("RGB")


def get_ssl_train_transform(
    image_size: int = 256,
    modality: str = "photograph",
    use_clahe: bool = True,
):
    """
    Anatomically safe multi-view augmentations for SSL v2 pretraining.
    Resizes first so CLAHE runs on a fixed 256x256 grid in ~10ms.
    """
    ops = [
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.78, 1.0),
            ratio=(0.90, 1.10),
        ),
    ]

    if modality == "radiograph" and use_clahe:
        ops.append(DentalRadiographCLAHE(clip_limit=2.0, grid_size=8))

    ops.extend(
        [
            transforms.RandomHorizontalFlip(p=0.50),
            transforms.RandomRotation(degrees=10),
        ]
    )

    if modality == "photograph":
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.12,
                        hue=0.02,
                    )
                ],
                p=0.60,
            )
        )
    elif modality == "radiograph":
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.12,
                        contrast=0.18,
                    )
                ],
                p=0.50,
            )
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    ops.extend(
        [
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(ops)


def get_downstream_train_transform(
    image_size: int = 256,
    modality: str = "photograph",
    use_clahe: bool = True,
):
    """
    Regularized transformations for downstream/fusion training.
    """
    ops = [transforms.Resize((image_size, image_size))]

    if modality == "radiograph" and use_clahe:
        ops.append(DentalRadiographCLAHE(clip_limit=2.0, grid_size=8))

    ops.extend(
        [
            transforms.RandomHorizontalFlip(p=0.50),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.94, 1.06),
            ),
        ]
    )

    if modality == "photograph":
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.12,
                        contrast=0.12,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.50,
            )
        )
    elif modality == "radiograph":
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.08,
                        contrast=0.12,
                    )
                ],
                p=0.35,
            )
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(ops)


def get_eval_transform(
    image_size: int = 256,
    modality: str = "photograph",
    use_clahe: bool = True,
):
    """
    Deterministic evaluation transform (applies CLAHE to radiographs).
    """
    ops = [transforms.Resize((image_size, image_size))]

    if modality == "radiograph" and use_clahe:
        ops.append(DentalRadiographCLAHE(clip_limit=2.0, grid_size=8))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(ops)