"""
Milestone 7.1 sanity check.

This test verifies only the shared dataset protocol.

It does NOT train a model.
It does NOT perform missing-modality experiments.
"""

from torch.utils.data import DataLoader

from src.fusion.dataset import FusionDataset


def main():
    print("=" * 60)
    print("MILESTONE 7.1 — FUSION PROTOCOL CHECK")
    print("=" * 60)

    for split in [
        "train",
        "validation",
        "test",
    ]:
        dataset = FusionDataset(
            split=split
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )

        batch = next(iter(loader))

        print()
        print(f"Split: {split}")
        print(f"Samples: {len(dataset)}")
        print(
            f"Batch size: {len(batch)}"
        )

        for i, sample in enumerate(batch):
            print(
                f"  sample[{i}] "
                f"images={len(sample['images'])} "
                f"radiographs={len(sample['radiographs'])} "
                f"text_chars={len(sample['text'])} "
                f"labels={tuple(sample['labels'].shape)}"
            )

            assert len(sample["images"]) > 0
            assert len(sample["radiographs"]) > 0
            assert len(sample["text"]) > 0
            assert sample["labels"].shape == (6,)

    print()
    print("=" * 60)
    print("PASS — DATASET PROTOCOL IS VALID")
    print("=" * 60)


if __name__ == "__main__":
    main()