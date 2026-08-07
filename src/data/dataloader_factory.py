"""
DataLoader factory for COde multimodal dataset.

Creates train, validation, and test DataLoaders
based on patient-level split information.
"""

from torch.utils.data import DataLoader, Subset

from src.data.code_dataset import COdeDataset

from src.data.collate import multimodal_collate


class DataLoaderFactory:
    """
    Factory for creating split-specific DataLoaders.
    """

    def __init__(
        self,
        dataset: COdeDataset,
        batch_size: int = 8,
        num_workers: int = 0,
    ):

        self.dataset = dataset

        self.batch_size = batch_size

        self.num_workers = num_workers


    def create_split_loader(
        self,
        split: str,
        shuffle: bool,
    ):

        indices = []

        for idx, row in self.dataset.df.iterrows():

            if row["split"] == split:

                indices.append(idx)


        subset = Subset(
            self.dataset,
            indices,
        )


        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=multimodal_collate,
        )


        return loader



    def create_train_loader(self):

        return self.create_split_loader(
            "train",
            shuffle=True,
        )


    def create_validation_loader(self):
        """
        Create validation DataLoader if validation split exists.
        """

        if "validation" not in self.dataset.df["split"].unique():

            raise ValueError(
                "Validation split does not exist in dataset."
            )

        return self.create_split_loader(
            "validation",
            shuffle=False,
        )


    def create_test_loader(self):

        return self.create_split_loader(
            "test",
            shuffle=False,
        )