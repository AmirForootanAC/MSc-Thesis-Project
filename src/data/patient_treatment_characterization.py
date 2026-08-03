# ============================================================
# Patient Treatment Characterization
# ============================================================
#
# Purpose:
# Characterize treatment, clinical, tooth-level, and episode-
# level differences between cross-visit image reuse events
# within the same patient.
#
# This module does NOT determine whether image reuse is valid
# or invalid. It provides an auditable characterization of
# the clinical context associated with reused images.
#
# Main analysis stages:
# 1. Normalize patient_id
# 2. Extract all mentioned teeth per visit
# 3. Add treatment-stage characterization
# 4. Add tooth overlap classification
# 5. Add treatment progression classification
# 6. Add treatment episode relation
# 7. Generate episode-level summaries
# ============================================================

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path("configs/audit.yaml")


# ============================================================
# Utility Functions
# ============================================================

def normalize_text(value: object) -> str:
    """Normalize text for comparison and keyword analysis."""

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text


def tokenize_text(value: object) -> set[str]:
    """Convert text into a normalized token set."""

    text = normalize_text(value)

    if not text:
        return set()

    return set(
        re.findall(
            r"\b\w+\b",
            text,
        )
    )


def jaccard_similarity(
    tokens_a: set[str],
    tokens_b: set[str],
) -> float:
    """Calculate Jaccard similarity between two token sets."""

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def classify_text_relation(
    text_a: object,
    text_b: object,
) -> str:
    """Classify the relationship between two text fields."""

    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a and not b:
        return "both_missing"

    if not a or not b:
        return "one_missing"

    if a == b:
        return "exact_same"

    similarity = jaccard_similarity(
        tokenize_text(a),
        tokenize_text(b),
    )

    if similarity >= 0.80:
        return "high_overlap"

    if similarity >= 0.40:
        return "moderate_overlap"

    if similarity > 0:
        return "low_overlap"

    return "no_overlap"


def classify_treatment_relation(
    treatment_a: object,
    treatment_b: object,
) -> str:
    """Classify treatment relationship conservatively."""

    a = normalize_text(treatment_a)
    b = normalize_text(treatment_b)

    if not a and not b:
        return "both_missing"

    if not a or not b:
        return "one_missing"

    if a == b:
        return "same_exact"

    similarity = jaccard_similarity(
        tokenize_text(a),
        tokenize_text(b),
    )

    if similarity >= 0.80:
        return "high_overlap"

    if similarity >= 0.40:
        return "moderate_overlap"

    if similarity > 0:
        return "low_overlap"

    return "different"


def classify_temporal_gap(
    days: float,
) -> str:
    """Classify temporal distance between two visits."""

    if pd.isna(days):
        return "unknown"

    if days == 0:
        return "same_day"

    if days <= 7:
        return "1_7_days"

    if days <= 30:
        return "8_30_days"

    if days <= 90:
        return "31_90_days"

    if days <= 365:
        return "91_365_days"

    return "over_1_year"


def extract_keywords(
    text: object,
    keyword_groups: dict[str, list[str]],
) -> list[str]:
    """Extract configured keywords from clinical text."""

    normalized = normalize_text(text)

    if not normalized:
        return []

    matched = []

    for group_name, keywords in keyword_groups.items():

        for keyword in keywords:

            keyword_normalized = normalize_text(
                keyword
            )

            if not keyword_normalized:
                continue

            if keyword_normalized in normalized:

                matched.append(
                    f"{group_name}:{keyword_normalized}"
                )

    return sorted(
        set(matched)
    )


# ============================================================
# Tooth Extraction
# ============================================================

def extract_teeth(
    text: object,
) -> list[str]:
    """
    Extract all tooth identifiers mentioned in a clinical text.

    Supports:
    - FDI notation: 11, 12, 21, 36, 46, ...
    - FDI notation with separators: 1-1, 4-6
    - Common dental notation with '#': #11, #36
    - Multiple teeth in the same visit

    The function returns all unique normalized tooth identifiers
    as a sorted list.

    This is an extraction utility, not a clinical interpretation.
    """

    text = normalize_text(text)

    if not text:
        return []

    teeth = set()

    # --------------------------------------------------------
    # Explicit # notation
    # --------------------------------------------------------

    for match in re.findall(
        r"#\s*(\d{2})\b",
        text,
    ):
        teeth.add(match)

    # --------------------------------------------------------
    # FDI notation with optional separator
    # --------------------------------------------------------

    for match in re.findall(
        r"\b([1-4])\s*[-./]?\s*([1-8])\b",
        text,
    ):
        tooth = f"{match[0]}{match[1]}"
        teeth.add(tooth)

    return sorted(
        teeth,
        key=lambda value: (
            int(value[0]),
            int(value[1]),
        ),
    )


def build_tooth_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract all teeth mentioned across all relevant visit text.

    Multiple teeth are preserved in one pipe-separated string.
    """

    df = df.copy()

    tooth_source_columns = [
        "diagnosis",
        "treatment_plan",
        "treatment_recommendations",
        "management",
        "examination",
        "present_illness",
        "patient_record",
        "chief_complaint",
    ]

    def extract_row_teeth(row: pd.Series) -> list[str]:

        all_teeth = set()

        for column in tooth_source_columns:

            all_teeth.update(
                extract_teeth(
                    row[column]
                )
            )

        return sorted(
            all_teeth,
            key=lambda value: (
                int(value[0]),
                int(value[1]),
            ),
        )

    df["teeth_mentioned_list"] = df.apply(
        extract_row_teeth,
        axis=1,
    )

    df["teeth_mentioned"] = (
        df["teeth_mentioned_list"]
        .apply(
            lambda values: "|".join(values)
        )
    )

    df["num_teeth_mentioned"] = (
        df["teeth_mentioned_list"]
        .apply(len)
    )

    return df


def parse_teeth_string(
    value: object,
) -> set[str]:
    """Parse pipe-separated tooth identifiers."""

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    return set(
        tooth.strip()
        for tooth in text.split("|")
        if tooth.strip()
    )


def classify_tooth_overlap(
    teeth_a: object,
    teeth_b: object,
) -> tuple[str, list[str]]:
    """
    Classify overlap between all teeth mentioned in two visits.
    """

    set_a = parse_teeth_string(
        teeth_a
    )

    set_b = parse_teeth_string(
        teeth_b
    )

    if not set_a and not set_b:
        return (
            "both_missing",
            [],
        )

    if not set_a or not set_b:
        return (
            "one_missing",
            [],
        )

    shared = sorted(
        set_a & set_b,
        key=lambda value: (
            int(value[0]),
            int(value[1]),
        ),
    )

    if set_a == set_b:
        return (
            "exact_overlap",
            shared,
        )

    if shared:
        return (
            "partial_overlap",
            shared,
        )

    return (
        "no_overlap",
        [],
    )


# ============================================================
# Treatment Stage
# ============================================================

TREATMENT_STAGE_KEYWORDS = {
    "diagnostic": [
        "diagnosis",
        "diagnostic",
        "examination",
        "assessment",
        "evaluation",
        "consultation",
    ],
    "preventive": [
        "prevention",
        "preventive",
        "fluoride",
        "sealant",
        "prophylaxis",
    ],
    "periodontal": [
        "periodontal",
        "scaling",
        "root planing",
        "deep cleaning",
    ],
    "restorative": [
        "filling",
        "restoration",
        "composite",
        "amalgam",
    ],
    "endodontic": [
        "root canal",
        "endodontic",
        "pulpotomy",
        "pulpectomy",
    ],
    "prosthodontic": [
        "crown",
        "bridge",
        "prosthesis",
        "denture",
        "prosthodontic",
    ],
    "extraction": [
        "extraction",
        "extract",
        "remove tooth",
        "tooth removal",
    ],
    "implant": [
        "implant",
        "implantation",
    ],
    "orthodontic": [
        "orthodontic",
        "braces",
        "aligner",
    ],
    "surgical": [
        "surgery",
        "surgical",
        "operation",
    ],
}


TREATMENT_STAGE_ORDER = {
    "unknown": 0,
    "diagnostic": 1,
    "preventive": 2,
    "periodontal": 3,
    "restorative": 4,
    "endodontic": 5,
    "prosthodontic": 6,
    "extraction": 7,
    "implant": 8,
    "orthodontic": 9,
    "surgical": 10,
}


def extract_treatment_stages(
    text: object,
) -> list[str]:
    """Extract all treatment stages mentioned in text."""

    normalized = normalize_text(
        text
    )

    if not normalized:
        return []

    stages = []

    for stage, keywords in TREATMENT_STAGE_KEYWORDS.items():

        for keyword in keywords:

            if normalize_text(keyword) in normalized:

                stages.append(stage)
                break

    return sorted(
        set(stages),
        key=lambda value: TREATMENT_STAGE_ORDER.get(
            value,
            999,
        ),
    )


def classify_treatment_stage_relation(
    stages_a: object,
    stages_b: object,
) -> str:
    """Classify treatment-stage relationship."""

    set_a = set(
        filter(
            None,
            str(stages_a).split("|"),
        )
    )

    set_b = set(
        filter(
            None,
            str(stages_b).split("|"),
        )
    )

    if not set_a and not set_b:
        return "both_missing"

    if not set_a or not set_b:
        return "one_missing"

    if set_a == set_b:
        return "same_stage"

    if set_a & set_b:
        return "overlapping_stage"

    return "different_stage"


# ============================================================
# Treatment Progression
# ============================================================

def classify_treatment_progression(
    stages_a: object,
    stages_b: object,
    temporal_gap_days: float,
) -> str:
    """
    Characterize longitudinal treatment progression.

    This is a heuristic characterization and does not claim
    clinical correctness.
    """

    if pd.isna(temporal_gap_days):
        return "unknown_temporal_order"

    set_a = set(
        filter(
            None,
            str(stages_a).split("|"),
        )
    )

    set_b = set(
        filter(
            None,
            str(stages_b).split("|"),
        )
    )

    if not set_a or not set_b:
        return "insufficient_stage_information"

    max_a = max(
        TREATMENT_STAGE_ORDER.get(
            stage,
            0,
        )
        for stage in set_a
    )

    max_b = max(
        TREATMENT_STAGE_ORDER.get(
            stage,
            0,
        )
        for stage in set_b
    )

    if set_a == set_b:
        return "same_stage"

    if max_b > max_a:
        return "progression_forward"

    if max_b < max_a:
        return "progression_backward"

    if set_a & set_b:
        return "continued_or_overlapping"

    return "different_treatment_path"


# ============================================================
# Treatment Episode Relation
# ============================================================

def classify_episode_relation(
    temporal_gap_days: float,
    tooth_overlap_relation: str,
    treatment_stage_relation: str,
    treatment_relation: str,
    diagnosis_relation: str,
    clinical_context_relation: str,
) -> str:
    """
    Characterize whether two reused-image visits likely belong
    to the same or different treatment episodes.

    This is a conservative, evidence-based heuristic.

    Important:
    Missing tooth information does NOT automatically produce
    an unknown result. When tooth information is unavailable,
    temporal, treatment, stage, diagnosis, and clinical context
    are used as secondary evidence.

    Possible outputs:
    - likely_same_episode
    - possible_same_episode
    - likely_different_episode
    - possible_different_episode
    - insufficient_evidence
    """

    # --------------------------------------------------------
    # Temporal information
    # --------------------------------------------------------

    if pd.isna(temporal_gap_days):
        temporal_gap_days = None

    # --------------------------------------------------------
    # Evidence definitions
    # --------------------------------------------------------

    same_tooth = tooth_overlap_relation == "exact_overlap"

    partial_tooth = (
        tooth_overlap_relation
        == "partial_overlap"
    )

    no_tooth_overlap = (
        tooth_overlap_relation
        == "no_overlap"
    )

    tooth_information_missing = (
        tooth_overlap_relation
        in {
            "one_missing",
            "both_missing",
        }
    )

    compatible_treatment = (
        treatment_relation
        in {
            "same_exact",
            "high_overlap",
            "moderate_overlap",
        }
    )

    weakly_compatible_treatment = (
        treatment_relation
        == "low_overlap"
    )

    different_treatment = (
        treatment_relation
        == "different"
    )

    compatible_stage = (
        treatment_stage_relation
        in {
            "same_stage",
            "overlapping_stage",
        }
    )

    different_stage = (
        treatment_stage_relation
        == "different_stage"
    )

    compatible_diagnosis = (
        diagnosis_relation
        in {
            "exact_same",
            "high_overlap",
            "moderate_overlap",
        }
    )

    compatible_clinical_context = (
        clinical_context_relation
        in {
            "exact_same",
            "high_overlap",
            "moderate_overlap",
        }
    )

    # --------------------------------------------------------
    # Rule 1:
    # Exact same teeth + compatible clinical/treatment context
    # --------------------------------------------------------

    if same_tooth:

        if temporal_gap_days is not None:

            if (
                temporal_gap_days <= 90
                and (
                    compatible_treatment
                    or compatible_stage
                    or compatible_diagnosis
                    or compatible_clinical_context
                )
            ):
                return "likely_same_episode"

            if temporal_gap_days <= 365:

                if (
                    compatible_treatment
                    or compatible_stage
                    or compatible_diagnosis
                ):
                    return "possible_same_episode"

        # Same teeth but clearly different treatment
        if (
            different_treatment
            and different_stage
        ):
            return "possible_different_episode"

        if (
            different_treatment
            and different_stage
            and temporal_gap_days is not None
            and temporal_gap_days > 90
        ):
            return "likely_different_episode"

    # --------------------------------------------------------
    # Rule 2:
    # Partial tooth overlap
    # --------------------------------------------------------

    if partial_tooth:

        if temporal_gap_days is not None:

            if (
                temporal_gap_days <= 30
                and (
                    compatible_treatment
                    or compatible_stage
                )
            ):
                return "likely_same_episode"

            if (
                temporal_gap_days <= 180
                and (
                    compatible_treatment
                    or compatible_stage
                    or compatible_diagnosis
                )
            ):
                return "possible_same_episode"

            if (
                temporal_gap_days > 180
                and different_treatment
                and different_stage
            ):
                return "possible_different_episode"

    # --------------------------------------------------------
    # Rule 3:
    # Explicitly different teeth
    # --------------------------------------------------------

    if no_tooth_overlap:

        if (
            different_treatment
            and different_stage
        ):
            return "likely_different_episode"

        if (
            temporal_gap_days is not None
            and temporal_gap_days > 90
            and (
                different_treatment
                or different_stage
            )
        ):
            return "likely_different_episode"

        return "possible_different_episode"

    # --------------------------------------------------------
    # Rule 4:
    # Tooth information missing
    #
    # We do NOT immediately return unknown.
    # Instead, use all available contextual evidence.
    # --------------------------------------------------------

    if tooth_information_missing:

        evidence_score = 0

        # Treatment evidence
        if compatible_treatment:
            evidence_score += 2

        elif weakly_compatible_treatment:
            evidence_score += 1

        elif different_treatment:
            evidence_score -= 2

        # Treatment stage evidence
        if compatible_stage:
            evidence_score += 2

        elif different_stage:
            evidence_score -= 1

        # Diagnosis evidence
        if compatible_diagnosis:
            evidence_score += 1

        # Clinical context evidence
        if compatible_clinical_context:
            evidence_score += 1

        # Temporal evidence
        if temporal_gap_days is not None:

            if temporal_gap_days <= 30:
                evidence_score += 2

            elif temporal_gap_days <= 90:
                evidence_score += 1

            elif temporal_gap_days > 365:
                evidence_score -= 1

        # ----------------------------------------------------
        # Strong evidence for same episode
        # ----------------------------------------------------

        if (
            evidence_score >= 4
            and temporal_gap_days is not None
            and temporal_gap_days <= 90
        ):
            return "possible_same_episode"

        # ----------------------------------------------------
        # Moderate evidence for same episode
        # ----------------------------------------------------

        if (
            evidence_score >= 3
            and temporal_gap_days is not None
            and temporal_gap_days <= 30
        ):
            return "possible_same_episode"

        # ----------------------------------------------------
        # Strong evidence for different episode
        # ----------------------------------------------------

        if evidence_score <= -3:

            return "possible_different_episode"

        # ----------------------------------------------------
        # Insufficient evidence
        # ----------------------------------------------------

        return "insufficient_evidence"

    # --------------------------------------------------------
    # Rule 5:
    # Fallback
    # --------------------------------------------------------

    return "insufficient_evidence"


# ============================================================
# Data Loading
# ============================================================

def load_config(
    config_path: Path,
) -> dict:
    """Load YAML configuration."""

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def load_duplication_data(
    output_dir: Path,
) -> pd.DataFrame:
    """Load cross-visit image reuse information."""

    path = (
        output_dir
        / "visit_pair_reuse.csv"
    )

    print(
        "[INFO] Loading cross-visit image reuse data..."
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "patient_id": "string",
            "visit_1": "string",
            "visit_2": "string",
        },
    )

    df["patient_id"] = (
        df["patient_id"]
        .astype("string")
        .str.strip()
        .str.zfill(4)
    )

    print(
        f"[INFO] Loaded {len(df):,} "
        "reused visit-pair rows."
    )

    return df


def load_visit_metadata(
    metadata_path: Path,
) -> pd.DataFrame:
    """Load original visit-level metadata."""

    print(
        "[INFO] Loading visit metadata..."
    )

    df = pd.read_csv(
        metadata_path,
        low_memory=False,
    )

    print(
        f"[INFO] Loaded {len(df):,} "
        "visit metadata rows."
    )

    return df


# ============================================================
# Metadata Preparation
# ============================================================

def prepare_visit_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare visit metadata."""

    required_columns = [
        "patient_id",
        "checkup_id",
        "checkup_time",
        "diagnosis",
        "treatment_plan",
        "treatment_recommendations",
        "management",
        "examination",
        "present_illness",
        "patient_record",
        "chief_complaint",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required metadata columns: "
            + ", ".join(
                missing_columns
            )
        )

    df = df.copy()

    df["checkup_id"] = (
        df["checkup_id"]
        .astype("string")
        .str.strip()
    )

    df["patient_id"] = (
        df["patient_id"]
        .astype("string")
        .str.strip()
        .str.zfill(4)
    )

    df["checkup_datetime"] = pd.to_datetime(
        df["checkup_time"],
        dayfirst=True,
        errors="coerce",
    )

    return df


# ============================================================
# Clinical and Treatment Context
# ============================================================

def build_clinical_text(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build combined clinical and treatment context."""

    df = df.copy()

    clinical_columns = [
        "patient_record",
        "chief_complaint",
        "present_illness",
        "examination",
        "diagnosis",
    ]

    treatment_columns = [
        "treatment_plan",
        "treatment_recommendations",
        "management",
    ]

    df["clinical_context"] = (
        df[clinical_columns]
        .fillna("")
        .astype(str)
        .agg(
            " | ".join,
            axis=1,
        )
        .str.strip()
    )

    df["treatment_context"] = (
        df[treatment_columns]
        .fillna("")
        .astype(str)
        .agg(
            " | ".join,
            axis=1,
        )
        .str.strip()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Extract all teeth BEFORE pair-level merge.
    # Each visit keeps all mentioned teeth.
    # --------------------------------------------------------

    df = build_tooth_context(
        df
    )

    df["treatment_stages"] = (
        df["treatment_context"]
        .apply(
            lambda value: "|".join(
                extract_treatment_stages(
                    value
                )
            )
        )
    )

    return df


# ============================================================
# Visit Pair Characterization
# ============================================================

def characterize_visit_pairs(
    reuse_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    keyword_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Attach visit-level context to reused visit pairs."""

    visit_columns = [
        "checkup_id",
        "patient_id",
        "checkup_datetime",
        "diagnosis",
        "treatment_plan",
        "treatment_recommendations",
        "management",
        "examination",
        "present_illness",
        "patient_record",
        "chief_complaint",
        "clinical_context",
        "treatment_context",
        "teeth_mentioned",
        "num_teeth_mentioned",
        "treatment_stages",
    ]

    metadata = metadata_df[
        visit_columns
    ].copy()

    visit_1 = metadata.rename(
        columns={
            "checkup_id": "_checkup_id_1",
            **{
                column: f"{column}_1"
                for column in visit_columns
                if column != "checkup_id"
            },
        }
    )

    visit_2 = metadata.rename(
        columns={
            "checkup_id": "_checkup_id_2",
            **{
                column: f"{column}_2"
                for column in visit_columns
                if column != "checkup_id"
            },
        }
    )

    result = reuse_df.merge(
        visit_1,
        left_on="visit_1",
        right_on="_checkup_id_1",
        how="left",
    )

    result = result.merge(
        visit_2,
        left_on="visit_2",
        right_on="_checkup_id_2",
        how="left",
    )

    result = result.drop(
        columns=[
            "_checkup_id_1",
            "_checkup_id_2",
        ]
    )

    # --------------------------------------------------------
    # Patient ID normalization
    # --------------------------------------------------------

    result["patient_id"] = (
        result["patient_id"]
        .astype("string")
        .str.strip()
        .str.zfill(4)
    )

    # --------------------------------------------------------
    # Temporal characterization
    # --------------------------------------------------------

    result["temporal_gap_days_calculated"] = (
        result["checkup_datetime_2"]
        - result["checkup_datetime_1"]
    ).dt.total_seconds() / 86400

    result["temporal_gap_category"] = (
        result[
            "temporal_gap_days_calculated"
        ]
        .apply(
            classify_temporal_gap
        )
    )

    # --------------------------------------------------------
    # Clinical and treatment relations
    # --------------------------------------------------------

    result["diagnosis_relation"] = result.apply(
        lambda row: classify_text_relation(
            row["diagnosis_1"],
            row["diagnosis_2"],
        ),
        axis=1,
    )

    result["treatment_relation"] = result.apply(
        lambda row: classify_treatment_relation(
            row["treatment_context_1"],
            row["treatment_context_2"],
        ),
        axis=1,
    )

    result["clinical_context_relation"] = result.apply(
        lambda row: classify_text_relation(
            row["clinical_context_1"],
            row["clinical_context_2"],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Tooth overlap
    # --------------------------------------------------------

    tooth_results = result.apply(
        lambda row: classify_tooth_overlap(
            row["teeth_mentioned_1"],
            row["teeth_mentioned_2"],
        ),
        axis=1,
    )

    result["tooth_overlap_relation"] = (
        tooth_results
        .apply(
            lambda value: value[0]
        )
    )

    result["shared_teeth"] = (
        tooth_results
        .apply(
            lambda value: "|".join(
                value[1]
            )
        )
    )

    result["num_shared_teeth"] = (
        tooth_results
        .apply(
            lambda value: len(
                value[1]
            )
        )
    )

    # --------------------------------------------------------
    # Treatment-stage relation
    # --------------------------------------------------------

    result["treatment_stage_relation"] = result.apply(
        lambda row: classify_treatment_stage_relation(
            row["treatment_stages_1"],
            row["treatment_stages_2"],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Treatment progression
    # --------------------------------------------------------

    result["treatment_progression"] = result.apply(
        lambda row: classify_treatment_progression(
            row["treatment_stages_1"],
            row["treatment_stages_2"],
            row[
                "temporal_gap_days_calculated"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Treatment episode relation
    # --------------------------------------------------------

    result["episode_relation"] = result.apply(
        lambda row: classify_episode_relation(
            row[
                "temporal_gap_days_calculated"
            ],
            row[
                "tooth_overlap_relation"
            ],
            row[
                "treatment_stage_relation"
            ],
            row[
                "treatment_relation"
            ],
            row[
                "diagnosis_relation"
            ],
            row[
                "clinical_context_relation"
            ],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Treatment keywords
    # --------------------------------------------------------

    result["treatment_keywords_1"] = (
        result["treatment_context_1"]
        .apply(
            lambda value: "|".join(
                extract_keywords(
                    value,
                    keyword_groups,
                )
            )
        )
    )

    result["treatment_keywords_2"] = (
        result["treatment_context_2"]
        .apply(
            lambda value: "|".join(
                extract_keywords(
                    value,
                    keyword_groups,
                )
            )
        )
    )

    result["treatment_keyword_overlap"] = result.apply(
        lambda row: len(
            set(
                filter(
                    None,
                    str(
                        row[
                            "treatment_keywords_1"
                        ]
                    ).split("|"),
                )
            )
            &
            set(
                filter(
                    None,
                    str(
                        row[
                            "treatment_keywords_2"
                        ]
                    ).split("|"),
                )
            )
        ),
        axis=1,
    )

    return result


# ============================================================
# Summary Generation
# ============================================================

def build_pair_summary(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build pair-level treatment and episode summary."""

    summary = (
        pair_df.groupby(
            [
                "modality",
                "visit_relationship",
                "temporal_gap_category",
                "tooth_overlap_relation",
                "treatment_stage_relation",
                "treatment_progression",
                "episode_relation",
            ],
            dropna=False,
        )
        .agg(
            num_visit_pairs=(
                "patient_id",
                "size",
            ),
            total_shared_images=(
                "num_shared_images",
                "sum",
            ),
            mean_shared_images=(
                "num_shared_images",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "modality",
                "num_visit_pairs",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    return summary


def build_episode_summary(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build episode-level summary."""

    summary = (
        pair_df.groupby(
            [
                "episode_relation",
                "tooth_overlap_relation",
                "treatment_stage_relation",
                "treatment_progression",
            ],
            dropna=False,
        )
        .agg(
            num_visit_pairs=(
                "patient_id",
                "size",
            ),
            num_unique_patients=(
                "patient_id",
                "nunique",
            ),
            total_shared_images=(
                "num_shared_images",
                "sum",
            ),
            mean_temporal_gap_days=(
                "temporal_gap_days_calculated",
                "mean",
            ),
            mean_shared_teeth=(
                "num_shared_teeth",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "num_visit_pairs",
            ascending=False,
        )
    )

    return summary


def build_patient_summary(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build patient-level longitudinal summary."""

    summary = (
        pair_df.groupby(
            "patient_id"
        )
        .agg(
            num_reused_visit_pairs=(
                "visit_1",
                "count",
            ),
            num_same_treatment_pairs=(
                "treatment_relation",
                lambda x: (
                    x == "same_exact"
                ).sum(),
            ),
            num_different_treatment_pairs=(
                "treatment_relation",
                lambda x: (
                    x == "different"
                ).sum(),
            ),
            num_overlapping_treatment_pairs=(
                "treatment_relation",
                lambda x: x.isin(
                    [
                        "high_overlap",
                        "moderate_overlap",
                        "low_overlap",
                    ]
                ).sum(),
            ),
            num_exact_tooth_overlap_pairs=(
                "tooth_overlap_relation",
                lambda x: (
                    x == "exact_overlap"
                ).sum(),
            ),
            num_partial_tooth_overlap_pairs=(
                "tooth_overlap_relation",
                lambda x: (
                    x == "partial_overlap"
                ).sum(),
            ),
            num_no_tooth_overlap_pairs=(
                "tooth_overlap_relation",
                lambda x: (
                    x == "no_overlap"
                ).sum(),
            ),
            num_likely_same_episode_pairs=(
                "episode_relation",
                lambda x: (
                    x == "likely_same_episode"
                ).sum(),
            ),
            num_possible_same_episode_pairs=(
                "episode_relation",
                lambda x: (
                    x == "possible_same_episode"
                ).sum(),
            ),
            num_likely_different_episode_pairs=(
                "episode_relation",
                lambda x: (
                    x == "likely_different_episode"
                ).sum(),
            ),
            num_same_day_pairs=(
                "temporal_gap_category",
                lambda x: (
                    x == "same_day"
                ).sum(),
            ),
            num_long_term_pairs=(
                "temporal_gap_category",
                lambda x: x.isin(
                    [
                        "91_365_days",
                        "over_1_year",
                    ]
                ).sum(),
            ),
        )
        .reset_index()
    )

    summary["patient_id"] = (
        summary["patient_id"]
        .astype("string")
        .str.strip()
        .str.zfill(4)
    )

    summary[
        "has_different_treatment_reuse"
    ] = (
        summary[
            "num_different_treatment_pairs"
        ]
        > 0
    )

    summary[
        "has_same_treatment_reuse"
    ] = (
        summary[
            "num_same_treatment_pairs"
        ]
        > 0
    )

    return summary


def build_keyword_frequency(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count treatment keywords across reused visit pairs."""

    rows = []

    for column, visit_label in [
        (
            "treatment_keywords_1",
            "visit_1",
        ),
        (
            "treatment_keywords_2",
            "visit_2",
        ),
    ]:

        for value in pair_df[
            column
        ].dropna():

            if not value:
                continue

            for keyword in str(
                value
            ).split("|"):

                if not keyword:
                    continue

                rows.append(
                    {
                        "visit": visit_label,
                        "keyword": keyword,
                    }
                )

    if not rows:

        return pd.DataFrame(
            columns=[
                "visit",
                "keyword",
                "frequency",
            ]
        )

    return (
        pd.DataFrame(rows)
        .groupby(
            [
                "visit",
                "keyword",
            ]
        )
        .size()
        .reset_index(
            name="frequency"
        )
        .sort_values(
            "frequency",
            ascending=False,
        )
    )


# ============================================================
# Audit Summary
# ============================================================

def build_audit_summary(
    pair_df: pd.DataFrame,
    patient_summary: pd.DataFrame,
) -> dict:
    """Build high-level audit statistics."""

    return {
        "num_reused_visit_pair_rows": int(
            len(pair_df)
        ),
        "num_unique_patients_with_reuse": int(
            pair_df[
                "patient_id"
            ].nunique()
        ),
        "num_photograph_reuse_pairs": int(
            (
                pair_df[
                    "modality"
                ]
                == "photographs"
            ).sum()
        ),
        "num_radiograph_reuse_pairs": int(
            (
                pair_df[
                    "modality"
                ]
                == "radiographs"
            ).sum()
        ),
        "num_same_treatment_pairs": int(
            (
                pair_df[
                    "treatment_relation"
                ]
                == "same_exact"
            ).sum()
        ),
        "num_different_treatment_pairs": int(
            (
                pair_df[
                    "treatment_relation"
                ]
                == "different"
            ).sum()
        ),
        "num_exact_tooth_overlap_pairs": int(
            (
                pair_df[
                    "tooth_overlap_relation"
                ]
                == "exact_overlap"
            ).sum()
        ),
        "num_partial_tooth_overlap_pairs": int(
            (
                pair_df[
                    "tooth_overlap_relation"
                ]
                == "partial_overlap"
            ).sum()
        ),
        "num_no_tooth_overlap_pairs": int(
            (
                pair_df[
                    "tooth_overlap_relation"
                ]
                == "no_overlap"
            ).sum()
        ),
        "num_likely_same_episode_pairs": int(
            (
                pair_df[
                    "episode_relation"
                ]
                == "likely_same_episode"
            ).sum()
        ),
        "num_possible_same_episode_pairs": int(
            (
                pair_df[
                    "episode_relation"
                ]
                == "possible_same_episode"
            ).sum()
        ),
        "num_likely_different_episode_pairs": int(
            (
                pair_df[
                    "episode_relation"
                ]
                == "likely_different_episode"
            ).sum()
        ),
        "num_same_day_reuse_pairs": int(
            (
                pair_df[
                    "temporal_gap_category"
                ]
                == "same_day"
            ).sum()
        ),
        "num_long_term_reuse_pairs": int(
            pair_df[
                "temporal_gap_category"
            ].isin(
                [
                    "91_365_days",
                    "over_1_year",
                ]
            ).sum()
        ),
        "num_patients_with_different_treatment_reuse": int(
            patient_summary[
                "has_different_treatment_reuse"
            ].sum()
        ),
        "num_patients_with_same_treatment_reuse": int(
            patient_summary[
                "has_same_treatment_reuse"
            ].sum()
        ),
    }


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Characterize clinical, treatment, "
            "tooth-level, and episode-level context "
            "of intra-patient cross-visit image reuse."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to audit configuration YAML.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    metadata_path = Path(
        config["paths"][
            "metadata_file"
        ]
    )

    cross_visit_dir = Path(
        config["output"][
            "patient_cross_visit_characterization_dir"
        ]
    )

    output_dir = Path(
        "results/patient_treatment_characterization"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_files = [
        output_dir
        / "duplicated_visit_treatment.csv",
        output_dir
        / "treatment_pair_summary.csv",
        output_dir
        / "episode_pair_summary.csv",
        output_dir
        / "patient_treatment_longitudinal.csv",
        output_dir
        / "treatment_keyword_frequency.csv",
        output_dir
        / "audit_summary.json",
    ]

    if not args.force:

        existing = [
            path
            for path in output_files
            if path.exists()
        ]

        if existing:

            raise FileExistsError(
                "Output files already exist. "
                "Use --force to overwrite."
            )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    reuse_df = load_duplication_data(
        cross_visit_dir
    )

    metadata_df = load_visit_metadata(
        metadata_path
    )

    metadata_df = prepare_visit_metadata(
        metadata_df
    )

    metadata_df = build_clinical_text(
        metadata_df
    )

    # --------------------------------------------------------
    # Treatment keyword groups
    # --------------------------------------------------------

    treatment_keywords = {
        "endodontic": [
            "root canal",
            "endodontic",
            "pulpotomy",
            "pulpectomy",
        ],
        "restoration": [
            "filling",
            "restoration",
            "composite",
            "amalgam",
        ],
        "crown": [
            "crown",
            "prosthodontic",
        ],
        "extraction": [
            "extraction",
            "extract",
            "remove tooth",
        ],
        "implant": [
            "implant",
        ],
        "periodontal": [
            "periodontal",
            "scaling",
            "root planing",
        ],
        "orthodontic": [
            "orthodontic",
            "braces",
        ],
        "surgery": [
            "surgery",
            "surgical",
            "operation",
        ],
        "prosthesis": [
            "prosthesis",
            "denture",
            "bridge",
        ],
    }

    # --------------------------------------------------------
    # Characterize reused visit pairs
    # --------------------------------------------------------

    print(
        "[INFO] Characterizing treatment, clinical, "
        "tooth, and episode context..."
    )

    pair_df = characterize_visit_pairs(
        reuse_df,
        metadata_df,
        treatment_keywords,
    )

    # --------------------------------------------------------
    # Build summaries
    # --------------------------------------------------------

    print(
        "[INFO] Building treatment and tooth "
        "pair summary..."
    )

    pair_summary = build_pair_summary(
        pair_df
    )

    print(
        "[INFO] Building episode-level summary..."
    )

    episode_summary = build_episode_summary(
        pair_df
    )

    print(
        "[INFO] Building patient longitudinal "
        "treatment summary..."
    )

    patient_summary = build_patient_summary(
        pair_df
    )

    print(
        "[INFO] Building treatment keyword "
        "frequency..."
    )

    keyword_frequency = build_keyword_frequency(
        pair_df
    )

    print(
        "[INFO] Building audit summary..."
    )

    audit_summary = build_audit_summary(
        pair_df,
        patient_summary,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    pair_df.to_csv(
        output_dir
        / "duplicated_visit_treatment.csv",
        index=False,
    )

    pair_summary.to_csv(
        output_dir
        / "treatment_pair_summary.csv",
        index=False,
    )

    episode_summary.to_csv(
        output_dir
        / "episode_pair_summary.csv",
        index=False,
    )

    patient_summary.to_csv(
        output_dir
        / "patient_treatment_longitudinal.csv",
        index=False,
    )

    keyword_frequency.to_csv(
        output_dir
        / "treatment_keyword_frequency.csv",
        index=False,
    )

    with (
        output_dir
        / "audit_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit_summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "Patient Treatment Characterization "
        "completed successfully."
    )
    print("=" * 60)

    print(
        f"Reused visit-pair rows: "
        f"{len(pair_df):,}"
    )

    print(
        f"Patients with reuse: "
        f"{pair_df['patient_id'].nunique():,}"
    )

    print(
        f"Same treatment pairs: "
        f"{audit_summary['num_same_treatment_pairs']:,}"
    )

    print(
        f"Different treatment pairs: "
        f"{audit_summary['num_different_treatment_pairs']:,}"
    )

    print(
        f"Exact tooth-overlap pairs: "
        f"{audit_summary['num_exact_tooth_overlap_pairs']:,}"
    )

    print(
        f"Partial tooth-overlap pairs: "
        f"{audit_summary['num_partial_tooth_overlap_pairs']:,}"
    )

    print(
        f"No tooth-overlap pairs: "
        f"{audit_summary['num_no_tooth_overlap_pairs']:,}"
    )

    print(
        f"Likely same treatment episode pairs: "
        f"{audit_summary['num_likely_same_episode_pairs']:,}"
    )

    print(
        f"Possible same treatment episode pairs: "
        f"{audit_summary['num_possible_same_episode_pairs']:,}"
    )

    print(
        f"Likely different treatment episode pairs: "
        f"{audit_summary['num_likely_different_episode_pairs']:,}"
    )

    print(
        f"Same-day reuse pairs: "
        f"{audit_summary['num_same_day_reuse_pairs']:,}"
    )

    print(
        f"Long-term reuse pairs: "
        f"{audit_summary['num_long_term_reuse_pairs']:,}"
    )

    print(
        "Results saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()