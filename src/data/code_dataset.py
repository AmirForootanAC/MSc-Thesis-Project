"""
PyTorch Dataset implementation for COde multimodal dataset.

Each dataset item represents one dental visit.
"""

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from src.data.sample import MultimodalSample
from src.data.text_loader import ClinicalTextLoader
from src.data.missing_modality import attach_missing_flags


class COdeDataset(Dataset):
    """
    PyTorch Dataset for the COde multimodal dental dataset.

    Unit of sample:
        Visit / Checkup

    Split unit:
        Patient
    """

    def __init__(
        self,
        csv_path: str,
    ):

        self.csv_path = Path(csv_path)

        self.df = pd.read_csv(
            self.csv_path
        )

        self.text_loader = ClinicalTextLoader()


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        index: int,
    ) -> MultimodalSample:

        row = self.df.iloc[index]

        sample = MultimodalSample(

            patient_id=str(
                row["patient_id"]
            ),

            visit_id=str(
                row["checkup_id"]
            ),

            photographs=self._parse_images(
                row["photographs"]
            ),

            radiographs=self._parse_images(
                row["radiographs"]
            ),

            clinical_text=self._load_clinical_text(
                row
            ),

            metadata={
                "age": row["age"],
                "gender": row["gender"],
                "checkup_time": row["checkup_time"],
            },

        )


        sample = attach_missing_flags(
            sample
        )


        return sample

    def _load_clinical_text(
        self,
        row,
    ):
        """
        Load clinical text and remove empty fields.

        A visit is considered to have clinical text
        only if at least one field contains meaningful content.
        """

        text = self.text_loader.load(
            row
        )

        cleaned_text = {
            key: value.strip()
            for key, value in text.items()
            if isinstance(value, str)
            and value.strip()
        }

        return cleaned_text

    @staticmethod
    def _parse_images(
        value,
    ):

        if pd.isna(value):
            return []

        return [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]