"""
Validation utilities for multimodal samples.

This module validates the consistency of MultimodalSample
objects before entering the data pipeline.
"""

from pathlib import Path
from typing import Dict, Any

from src.data.sample import MultimodalSample


def validate_sample(
    sample: MultimodalSample,
    check_files: bool = False,
) -> Dict[str, Any]:
    """
    Validate a single MultimodalSample.

    Parameters
    ----------
    sample:
        MultimodalSample object.

    check_files:
        If True, validate image path existence.

    Returns
    -------
    Dict containing validation status and errors.
    """

    errors = []

    # --------------------------------------------------
    # Identity validation
    # --------------------------------------------------

    if not sample.patient_id:
        errors.append(
            "Missing patient_id"
        )

    if not sample.visit_id:
        errors.append(
            "Missing visit_id"
        )

    # --------------------------------------------------
    # Missing flag validation
    # --------------------------------------------------

    if (
        sample.has_modality("photographs")
        != sample.missing_flags.get(
            "has_photographs",
            False
        )
    ):
        errors.append(
            "Photograph missing flag mismatch"
        )

    if (
        sample.has_modality("radiographs")
        != sample.missing_flags.get(
            "has_radiographs",
            False
        )
    ):
        errors.append(
            "Radiograph missing flag mismatch"
        )

    if (
        sample.has_modality("clinical_text")
        != sample.missing_flags.get(
            "has_clinical_text",
            False
        )
    ):
        errors.append(
            "Clinical text missing flag mismatch"
        )

    # --------------------------------------------------
    # File validation
    # --------------------------------------------------

    if check_files:

        for image_path in (
            sample.photographs
            + sample.radiographs
        ):

            if not Path(image_path).exists():
                errors.append(
                    f"Missing file: {image_path}"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }