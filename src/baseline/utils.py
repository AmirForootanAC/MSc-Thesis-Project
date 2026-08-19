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

def get_modality_batch(batch, modality):
    """
    Select input modality from batch.
    """

    if modality == "photograph":
        return batch["images"]

    elif modality == "radiograph":
        return batch["radiographs"]

    else:
        raise ValueError(
            f"Unknown modality: {modality}"
        )