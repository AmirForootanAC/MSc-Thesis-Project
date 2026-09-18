"""Medical-friendly transforms for X-ray fine-tuning."""

from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_train_transform(
    image_size=256,
):
    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=7
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(
                    0.02,
                    0.02,
                ),
                scale=(
                    0.95,
                    1.05,
                ),
            ),

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.08,
                        contrast=0.08,
                    )
                ],
                p=0.30,
            ),

            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(
                            0.1,
                            0.6,
                        ),
                    )
                ],
                p=0.10,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transform(
    image_size=256,
):
    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )