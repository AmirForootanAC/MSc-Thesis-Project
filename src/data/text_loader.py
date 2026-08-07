"""
File: src/data/text_loader.py

Description
-----------
Utility for extracting and normalizing clinical text from a single
COde dataset row.

This module does not perform tokenization or label extraction.
Its responsibility is limited to preparing clean raw text for
later NLP models.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


class ClinicalTextLoader:
    """
    Extract and normalize English clinical text fields.

    Notes
    -----
    Diagnosis is intentionally excluded because it will become part
    of the Diagnostic Label Engineering milestone.
    """

    TEXT_FIELDS = [
        "patient_record",
        "chief_complaint",
        "present_illness",
        "past_medical_record",
        "examination",
        "radiographs_examination",
        "treatment_plan",
        "treatment_recommendations",
        "management",
        "medical_instructions",
        "remarks",
        "anomalies_en",
    ]

    @staticmethod
    def normalize_text(text: object) -> str:
        """
        Basic text normalization.

        Parameters
        ----------
        text : object

        Returns
        -------
        str
        """

        if pd.isna(text):
            return ""

        text = str(text)

        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = " ".join(text.split())

        return text.strip()

    def load(self, row: pd.Series) -> Dict[str, str]:
        """
        Extract all supported clinical text fields.

        Parameters
        ----------
        row : pandas.Series

        Returns
        -------
        Dict[str, str]
        """

        output = {}

        for field in self.TEXT_FIELDS:
            output[field] = self.normalize_text(row.get(field, ""))

        return output