"""
Extract SSL encoder representations for multimodal fusion.

For each complete multimodal visit:

    Photographs
        ↓
    pixel-level mean pooling
        ↓
    SSL Image Encoder
        ↓
    2048-dim embedding

    Radiographs
        ↓
    pixel-level mean pooling
        ↓
    SSL Radiograph Encoder
        ↓
    2048-dim embedding

    Clinical Text
        ↓
    DistilBERT SSL Text Encoder
        ↓
    768-dim embedding

Projection heads are NOT used.

Missing-modality handling is NOT performed here.
Only complete multimodal samples are included by the dataset.
"""

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ssl.model import MultimodalSSLModel
from src.ssl.fusion.dataset import MultimodalFusionDataset
from src.ssl.fusion import config


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4

NUM_WORKERS = 0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Collate
# ============================================================

def fusion_collate(batch):
    """
    Collate multimodal samples while preserving variable-length
    image/radiograph collections.

    Each visit may contain a different number of photographs
    and radiographs, so the image tensors must remain as lists.
    """

    return {
        "checkup_id": [
            sample["checkup_id"]
            for sample in batch
        ],

        "patient_id": [
            sample["patient_id"]
            for sample in batch
        ],

        "images": [
            sample["images"]
            for sample in batch
        ],

        "radiographs": [
            sample["radiographs"]
            for sample in batch
        ],

        "input_ids": torch.stack(
            [
                sample["input_ids"]
                for sample in batch
            ]
        ),

        "attention_mask": torch.stack(
            [
                sample["attention_mask"]
                for sample in batch
            ]
        ),

        "labels": torch.stack(
            [
                sample["labels"]
                for sample in batch
            ]
        ),
    }


# ============================================================
# Image pooling
# ============================================================

def mean_pool_images(
    image_tensors,
    device,
):
    """
    Mean-pool all images belonging to one visit.

    Input:
        [N, 3, 224, 224]

    Output:
        [1, 3, 224, 224]

    This matches the image preprocessing strategy used
    during SSL pretraining.
    """

    image_tensors = image_tensors.to(
        device,
        non_blocking=True,
    )

    return image_tensors.mean(
        dim=0,
        keepdim=True,
    )


# ============================================================
# Embedding extraction
# ============================================================

def extract_embeddings(
    model,
    dataloader,
    device,
):
    """
    Extract encoder-level representations.

    Projection heads are intentionally excluded.
    """

    model.eval()

    embeddings = []

    with torch.no_grad():

        progress = tqdm(
            dataloader,
            desc="Extracting embeddings",
        )

        for batch in progress:

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            input_ids = (
                batch["input_ids"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            attention_mask = (
                batch["attention_mask"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            text_embeddings = (
                model.encoders.encode_text(
                    input_ids,
                    attention_mask,
                )
            )


            # ------------------------------------------------
            # Image + Radiograph
            # ------------------------------------------------

            batch_image_embeddings = []

            batch_radiograph_embeddings = []


            for images in batch["images"]:

                pooled_images = mean_pool_images(
                    images,
                    device,
                )

                image_embedding = (
                    model.encoders.encode_image(
                        pooled_images
                    )
                )

                batch_image_embeddings.append(
                    image_embedding.squeeze(0).cpu()
                )


            for radiographs in batch["radiographs"]:

                pooled_radiographs = mean_pool_images(
                    radiographs,
                    device,
                )

                radiograph_embedding = (
                    model.encoders.encode_radiograph(
                        pooled_radiographs
                    )
                )

                batch_radiograph_embeddings.append(
                    radiograph_embedding.squeeze(0).cpu()
                )


            # ------------------------------------------------
            # Store sample-level results
            # ------------------------------------------------

            for index in range(
                len(batch["checkup_id"])
            ):

                embeddings.append(
                    {
                        "checkup_id":
                            batch["checkup_id"][index],

                        "patient_id":
                            batch["patient_id"][index],

                        "image_embedding":
                            batch_image_embeddings[index],

                        "radiograph_embedding":
                            batch_radiograph_embeddings[index],

                        "text_embedding":
                            text_embeddings[index].cpu(),

                        "labels":
                            batch["labels"][index].cpu(),
                    }
                )


    return embeddings


# ============================================================
# Save
# ============================================================

def save_embeddings(
    embeddings,
    output_path,
):
    """
    Save extracted representations.
    """

    output_dir = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    torch.save(
        embeddings,
        output_path,
    )

    print(
        f"Saved {len(embeddings)} samples to:"
    )

    print(
        output_path
    )


# ============================================================
# Check embedding dimensions
# ============================================================

def validate_embeddings(
    embeddings,
):
    """
    Validate representation dimensions before saving.
    """

    if not embeddings:
        raise RuntimeError(
            "No embeddings were extracted."
        )

    expected_dimensions = {
        "image_embedding": 2048,
        "radiograph_embedding": 2048,
        "text_embedding": 768,
    }

    first = embeddings[0]

    for key, expected_dim in expected_dimensions.items():

        actual_dim = (
            first[key]
            .shape[-1]
        )

        if actual_dim != expected_dim:

            raise RuntimeError(
                f"Unexpected {key} dimension: "
                f"{actual_dim}. "
                f"Expected {expected_dim}."
            )

    print(
        "\nEmbedding dimensions:"
    )

    print(
        f"  Image:       "
        f"{first['image_embedding'].shape}"
    )

    print(
        f"  Radiograph:  "
        f"{first['radiograph_embedding'].shape}"
    )

    print(
        f"  Text:        "
        f"{first['text_embedding'].shape}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )


    # --------------------------------------------------------
    # Load SSL model
    # --------------------------------------------------------

    print(
        "\nLoading SSL model..."
    )

    model = MultimodalSSLModel()

    checkpoint = torch.load(
        config.SSL_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "SSL checkpoint loaded:"
    )

    print(
        config.SSL_CHECKPOINT
    )


    # --------------------------------------------------------
    # Extract each split
    # --------------------------------------------------------

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Split: {split}"
        )

        print(
            f"{'=' * 60}"
        )


        dataset = MultimodalFusionDataset(
            csv_path=str(
                config.DATASET_PATH
            ),
            split=split,
        )


        if len(dataset) == 0:

            raise RuntimeError(
                f"No fusion samples found for split: "
                f"{split}"
            )


        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=fusion_collate,
            num_workers=NUM_WORKERS,
            pin_memory=(
                DEVICE == "cuda"
            ),
        )


        embeddings = extract_embeddings(
            model=model,
            dataloader=dataloader,
            device=DEVICE,
        )


        validate_embeddings(
            embeddings
        )


        output_path = (
            config.PROJECT_ROOT
            /
            "results"
            /
            "ssl_pretraining"
            /
            "fusion_embeddings"
            /
            f"{split}.pt"
        )


        save_embeddings(
            embeddings,
            str(output_path),
        )


    print(
        "\nFusion embedding extraction finished."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()