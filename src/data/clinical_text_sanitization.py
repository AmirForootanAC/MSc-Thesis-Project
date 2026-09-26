"""
Clinical text sanitization for the canonical supervised datasets.

The purpose of this module is to prevent explicit target terminology
from being directly available to downstream models through clinical text.

Target-specific terms are removed rather than replaced with a marker,
because a marker such as [TARGET] could itself reveal the presence,
frequency, or location of a target-related finding.

The surrounding clinical context is preserved as much as possible.

This module does NOT modify labels or inspect anomalies_en.
"""

from __future__ import annotations

import re
from typing import Iterable


# ============================================================
# Target Terms
# ============================================================

TARGET_PATTERNS: tuple[str, ...] = (
    # --------------------------------------------------------
    # Caries
    # --------------------------------------------------------
    r"\bdental\s+caries\b",
    r"\bcarious\s+lesions?\b",
    r"\bcarious\b",
    r"\bcaries\b",

    # --------------------------------------------------------
    # Gingivitis
    # --------------------------------------------------------
    r"\bgingivitis\b",

    # --------------------------------------------------------
    # Malocclusion
    # --------------------------------------------------------
    r"\bclass\s+(?:i|ii|iii)\s+malocclusion\b",
    r"\bclass\s+(?:1|2|3)\s+malocclusion\b",
    r"\bmalocclusion\b",

    # --------------------------------------------------------
    # Pulpitis
    # --------------------------------------------------------
    r"\bpulpitis\b",

    # --------------------------------------------------------
    # Tooth Loss
    # --------------------------------------------------------
    r"\btooth\s+loss\b",
    r"\bmissing\s+teeth\b",
    r"\bmissing\s+tooth\b",
    r"\bteeth\s+(?:are|is)\s+missing\b",
    r"\btooth\s+(?:is|was)\s+missing\b",
    r"\bedentulism\b",
    r"\bedentulous\b",

    # --------------------------------------------------------
    # Tooth Structure Loss
    # --------------------------------------------------------
    r"\btooth\s+structure\s+loss\b",
)


_COMPILED_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in TARGET_PATTERNS
)


# ============================================================
# Sanitization
# ============================================================

def sanitize_clinical_text(text: object) -> object:
    """
    Remove explicit target terminology from one text field.

    The matched terminology is replaced with a single whitespace
    character so that the presence of a replacement token does not
    reveal whether a target-related term originally existed.
    """
    if not isinstance(text, str):
        return text

    sanitized = text

    for pattern in _COMPILED_PATTERNS:
        sanitized = pattern.sub(" ", sanitized)

    # Normalize whitespace introduced by removal.
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    return sanitized


def sanitize_text_columns(dataframe, columns: Iterable[str]):
    """
    Return a copy of dataframe with target terminology removed
    from the specified clinical text columns.
    """
    result = dataframe.copy()

    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Clinical text column not found: {column}")

        result[column] = result[column].map(sanitize_clinical_text)

    return result


# ============================================================
# Auditing
# ============================================================

def count_target_matches(text: object) -> int:
    """
    Count explicit target-term matches in one text field.
    """
    if not isinstance(text, str):
        return 0

    return sum(len(pattern.findall(text)) for pattern in _COMPILED_PATTERNS)


def count_dataframe_matches(dataframe, columns: Iterable[str]) -> int:
    """
    Count explicit target-term matches across selected text columns.
    """
    total = 0

    for column in columns:
        if column not in dataframe.columns:
            raise ValueError(f"Clinical text column not found: {column}")

        total += int(
            dataframe[column]
            .map(count_target_matches)
            .sum()
        )

    return total