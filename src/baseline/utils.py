"""
Utility functions for baseline experiments.
"""


def move_image_batch_to_device(
    batch_images,
    device,
):
    """
    Move variable-length image batches to device.
    """

    return [
        [
            image.to(device)
            for image in images
        ]
        for images in batch_images
    ]