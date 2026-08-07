"""
Missing modality handling utilities.

This module provides explicit missing-modality detection
for multimodal samples.

Missing modalities are preserved as information and are
not imputed or removed.
"""

from src.data.sample import MultimodalSample


EMPTY_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
}



def is_empty_text(value) -> bool:
    """
    Check whether a text value should be considered missing.
    """

    if value is None:
        return True

    normalized = str(value).strip().lower()

    return normalized in EMPTY_VALUES



def generate_missing_flags(
    sample: MultimodalSample,
) -> dict[str, bool]:
    """
    Generate missing modality indicators for a sample.
    """

    photographs_missing = (
        len(sample.photographs) == 0
    )

    radiographs_missing = (
        len(sample.radiographs) == 0
    )

    clinical_text_missing = (
        len(sample.clinical_text) == 0
        or all(
            is_empty_text(value)
            for value in sample.clinical_text.values()
        )
    )


    return {
        "photographs_missing": photographs_missing,
        "radiographs_missing": radiographs_missing,
        "clinical_text_missing": clinical_text_missing,
    }



def attach_missing_flags(
    sample: MultimodalSample,
) -> MultimodalSample:
    """
    Attach generated missing flags to sample.
    """

    sample.missing_flags = generate_missing_flags(
        sample
    )

    return sample