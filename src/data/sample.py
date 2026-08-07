"""
Multimodal sample representation for the COde dataset.

This module defines the fundamental data structure used throughout
the multimodal data pipeline.

Sample unit:
    Visit (checkup)

Split unit:
    Patient
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class MultimodalSample:
    """
    Representation of one multimodal dental visit.

    A single sample corresponds to one visit/checkup.
    All visits belonging to the same patient are assigned to the
    same dataset split during patient-level splitting.
    """

    # --------------------------------------------------
    # Identity information
    # --------------------------------------------------

    patient_id: str

    visit_id: str

    split: str | None = None

    # --------------------------------------------------
    # Multimodal inputs
    # --------------------------------------------------

    photographs: List[str] = field(default_factory=list)

    radiographs: List[str] = field(default_factory=list)

    clinical_text: Dict[str, str] = field(default_factory=dict)

    # --------------------------------------------------
    # Additional information
    # --------------------------------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------
    # Labels
    #
    # Raw labels only.
    # Processing and encoding will be handled in the
    # Task Definition milestone.
    # --------------------------------------------------

    labels: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------
    # Missing modality indicators
    # --------------------------------------------------

    missing_flags: Dict[str, bool] = field(default_factory=dict)

    def has_modality(self, modality: str) -> bool:
        """
        Check whether a specific modality exists.

        Supported modalities:
            photographs
            radiographs
            clinical_text
        """

        if modality == "photographs":
            return len(self.photographs) > 0

        if modality == "radiographs":
            return len(self.radiographs) > 0

        if modality == "clinical_text":
            return len(self.clinical_text) > 0

        raise ValueError(
            f"Unsupported modality: {modality}"
        )

    def num_images(self, modality: str) -> int:
        """
        Return number of images for a modality.
        """

        if modality == "photographs":
            return len(self.photographs)

        if modality == "radiographs":
            return len(self.radiographs)

        raise ValueError(
            f"Unsupported image modality: {modality}"
        )