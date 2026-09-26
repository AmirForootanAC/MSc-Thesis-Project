"""
Clinical Text -> Six-Label Leakage Audit on the FINAL Complete-Case Cohort.

Purpose
-------
Audit whether the six target labels used by the final supervised baseline
are explicitly present in the clinical text fields used by the text encoder.

IMPORTANT
---------
This audit uses EXACTLY the same complete-case filtering rule as:

    src/baseline/final/dataset.py

A row belongs to the complete-case cohort iff:

    photographs is non-empty
    AND radiographs is non-empty
    AND at least one of the four clinical text fields is non-empty

The audit never modifies the dataset.

Authoritative dataset
---------------------
results/six_label_patient_level_dataset/labeled_dataset.csv

Allowed clinical text columns
-----------------------------
- chief_complaint
- present_illness
- past_medical_record
- examination

Six target labels
-----------------
- label_caries
- label_gingivitis
- label_malocclusion
- label_pulpitis
- label_tooth_loss
- label_tooth_structure_loss

The audit distinguishes:

1. DIRECT LABEL TERMS
   Explicit label names or unambiguous textual equivalents.

2. POTENTIAL PROXY TERMS
   Clinical findings that may correlate with a label but are not
   automatically classified as leakage.

The audit is diagnostic only.
It does NOT remove rows, modify text, or alter labels.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "clinical_text_label_leakage_audit_complete_case"
)

TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]

LABEL_COLUMNS = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]

EXPECTED_COMPLETE_CASE_COUNTS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}


# =============================================================================
# Leakage Vocabulary
# =============================================================================

DIRECT_TERMS = {
    "label_caries": [
        "caries",
        "dental caries",
        "tooth caries",
        "dental decay",
        "tooth decay",
    ],
    "label_gingivitis": [
        "gingivitis",
        "gingival inflammation",
    ],
    "label_malocclusion": [
        "malocclusion",
        "malocclusion class i",
        "malocclusion class ii",
        "malocclusion class iii",
    ],
    "label_pulpitis": [
        "pulpitis",
        "reversible pulpitis",
        "irreversible pulpitis",
        "pulpal inflammation",
    ],
    "label_tooth_loss": [
        "tooth loss",
        "tooth lost",
        "missing tooth",
        "missing teeth",
        "tooth missing",
        "teeth missing",
        "edentulous",
        "edentulism",
    ],
    "label_tooth_structure_loss": [
        "tooth structure loss",
        "loss of tooth structure",
        "tooth structural loss",
        "loss of dental structure",
    ],
}


PROXY_TERMS = {
    "label_caries": [
        "cavity",
        "cavities",
        "decayed tooth",
        "decayed teeth",
        "decay",
        "carious",
    ],
    "label_gingivitis": [
        "gingival bleeding",
        "bleeding gums",
        "gum bleeding",
        "gingival swelling",
        "swollen gums",
        "gingival redness",
        "red gums",
    ],
    "label_malocclusion": [
        "crossbite",
        "overbite",
        "underbite",
        "open bite",
        "deep bite",
        "crowding",
        "spacing",
        "malaligned",
        "malalignment",
        "class i",
        "class ii",
        "class iii",
    ],
    "label_pulpitis": [
        "pulpal pain",
        "pulp pain",
        "pulp exposure",
        "spontaneous toothache",
    ],
    "label_tooth_loss": [
        "missing",
        "absent tooth",
        "absent teeth",
        "edentulous area",
    ],
    "label_tooth_structure_loss": [
        "attrition",
        "abrasion",
        "erosion",
        "fracture",
        "fractured tooth",
        "enamel loss",
        "dentin loss",
        "wear",
        "tooth wear",
        "chipped tooth",
    ],
}


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(value) -> str:
    """
    Normalize text for audit-only keyword matching.

    The original dataset is never modified.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.casefold()

    text = re.sub(
        r"[\u2010-\u2015\u2212]",
        "-",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def contains_term(
    text: str,
    term: str,
) -> bool:
    """
    Check whether a normalized term occurs as a complete token/phrase.
    """

    normalized_term = normalize_text(term)

    if not normalized_term:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_term)
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def matched_terms(
    text: str,
    vocabulary: list[str],
) -> list[str]:
    """
    Return all vocabulary terms found in text.
    """

    return [
        term
        for term in vocabulary
        if contains_term(text, term)
    ]


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    """
    Load the authoritative six-label dataset.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    required_columns = (
        [
            "split",
            "photographs",
            "radiographs",
            "patient_id",
            "checkup_id",
        ]
        + TEXT_COLUMNS
        + LABEL_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    return df


# =============================================================================
# Complete-Case Filtering
# =============================================================================

def has_value(value) -> bool:
    """
    Match the exact semantics used by CompleteCaseDataset.has_value().
    """

    return (
        pd.notna(value)
        and str(value).strip() != ""
    )


def build_complete_case_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply EXACTLY the complete-case rule used by
    src/baseline/final/dataset.py.

    Rule:

        photographs has value
        AND radiographs has value
        AND at least one text column is non-null
    """

    complete_case_mask = (
        df["photographs"].apply(has_value)
        & df["radiographs"].apply(has_value)
        & df[TEXT_COLUMNS].notna().any(axis=1)
    )

    complete_case_df = (
        df.loc[complete_case_mask]
        .copy()
        .reset_index(drop=True)
    )

    return complete_case_df


def validate_complete_case_counts(
    df: pd.DataFrame,
) -> None:
    """
    Verify that complete-case counts match the final baseline.
    """

    counts = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    print()
    print("=" * 78)
    print("COMPLETE-CASE SANITY CHECK")
    print("=" * 78)

    for split in [
        "train",
        "validation",
        "test",
    ]:
        actual = int(counts.get(split, 0))
        expected = EXPECTED_COMPLETE_CASE_COUNTS[split]

        status = "OK" if actual == expected else "MISMATCH"

        print(
            f"{split:<12} "
            f"actual={actual:>5,} "
            f"expected={expected:>5,} "
            f"[{status}]"
        )

    total_expected = sum(
        EXPECTED_COMPLETE_CASE_COUNTS.values()
    )

    total_actual = len(df)

    print(
        f"{'TOTAL':<12} "
        f"actual={total_actual:>5,} "
        f"expected={total_expected:>5,}"
    )

    if (
        total_actual != total_expected
        or any(
            int(counts.get(split, 0))
            != expected
            for split, expected
            in EXPECTED_COMPLETE_CASE_COUNTS.items()
        )
    ):
        raise RuntimeError(
            "Complete-case counts do not match the final baseline."
        )


# =============================================================================
# Row-Level Audit
# =============================================================================

def build_row_level_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create row-level audit results for the complete-case cohort.
    """

    identifier_columns = [
        "id",
        "patient_id",
        "checkup_id",
        "split",
    ]

    result = df[
        [
            column
            for column in identifier_columns
            if column in df.columns
        ]
    ].copy()

    # Keep the true target values in the row-level audit.
    for label in LABEL_COLUMNS:
        result[label] = (
            df[label]
            .astype(int)
            .values
        )

    # Keep original text so exact suspicious cases can be inspected later.
    for column in TEXT_COLUMNS:
        result[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .values
        )

    normalized_columns = {
        column: (
            df[column]
            .fillna("")
            .astype(str)
            .apply(normalize_text)
        )
        for column in TEXT_COLUMNS
    }

    for label in LABEL_COLUMNS:

        direct_matches = []
        direct_columns = []
        proxy_matches = []
        proxy_columns = []

        for idx in df.index:

            row_direct_terms = []
            row_direct_columns = []

            row_proxy_terms = []
            row_proxy_columns = []

            for column in TEXT_COLUMNS:

                text = normalized_columns[
                    column
                ].loc[idx]

                direct = matched_terms(
                    text,
                    DIRECT_TERMS[label],
                )

                proxy = matched_terms(
                    text,
                    PROXY_TERMS[label],
                )

                row_direct_terms.extend(direct)
                row_proxy_terms.extend(proxy)

                if direct:
                    row_direct_columns.append(
                        column
                    )

                if proxy:
                    row_proxy_columns.append(
                        column
                    )

            direct_matches.append(
                sorted(set(row_direct_terms))
            )

            direct_columns.append(
                sorted(set(row_direct_columns))
            )

            proxy_matches.append(
                sorted(set(row_proxy_terms))
            )

            proxy_columns.append(
                sorted(set(row_proxy_columns))
            )

        result[
            f"{label}__direct_terms"
        ] = [
            " | ".join(values)
            for values in direct_matches
        ]

        result[
            f"{label}__direct_columns"
        ] = [
            " | ".join(values)
            for values in direct_columns
        ]

        result[
            f"{label}__direct_match"
        ] = [
            bool(values)
            for values in direct_matches
        ]

        result[
            f"{label}__proxy_terms"
        ] = [
            " | ".join(values)
            for values in proxy_matches
        ]

        result[
            f"{label}__proxy_columns"
        ] = [
            " | ".join(values)
            for values in proxy_columns
        ]

        result[
            f"{label}__proxy_match"
        ] = [
            bool(values)
            for values in proxy_matches
        ]

    result["any_direct_label_match"] = False
    result["any_proxy_match"] = False

    for label in LABEL_COLUMNS:

        result["any_direct_label_match"] |= (
            result[
                f"{label}__direct_match"
            ]
        )

        result["any_proxy_match"] |= (
            result[
                f"{label}__proxy_match"
            ]
        )

    return result


# =============================================================================
# Column-Level Audit
# =============================================================================

def build_column_level_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit every text column separately.

    This answers the important question:
    "Is the leakage concentrated in one text field?"
    """

    rows = []

    for column in TEXT_COLUMNS:

        texts = (
            df[column]
            .fillna("")
            .astype(str)
            .apply(normalize_text)
        )

        for label in LABEL_COLUMNS:

            direct_count = 0
            proxy_count = 0

            direct_positive_count = 0
            direct_negative_count = 0

            proxy_positive_count = 0
            proxy_negative_count = 0

            direct_examples = []
            proxy_examples = []

            for idx, text in texts.items():

                direct = matched_terms(
                    text,
                    DIRECT_TERMS[label],
                )

                proxy = matched_terms(
                    text,
                    PROXY_TERMS[label],
                )

                if direct:

                    direct_count += 1

                    if int(df.loc[idx, label]) == 1:
                        direct_positive_count += 1
                    else:
                        direct_negative_count += 1

                    if len(direct_examples) < 5:
                        direct_examples.append(
                            {
                                "index": int(idx),
                                "label_value": int(
                                    df.loc[idx, label]
                                ),
                                "terms": direct,
                                "text": str(
                                    df.loc[
                                        idx,
                                        column,
                                    ]
                                )[:500],
                            }
                        )

                if proxy:

                    proxy_count += 1

                    if int(df.loc[idx, label]) == 1:
                        proxy_positive_count += 1
                    else:
                        proxy_negative_count += 1

                    if len(proxy_examples) < 5:
                        proxy_examples.append(
                            {
                                "index": int(idx),
                                "label_value": int(
                                    df.loc[idx, label]
                                ),
                                "terms": proxy,
                                "text": str(
                                    df.loc[
                                        idx,
                                        column,
                                    ]
                                )[:500],
                            }
                        )

            rows.append(
                {
                    "text_column": column,
                    "label": label,
                    "total_rows": int(len(df)),
                    "direct_match_rows": int(
                        direct_count
                    ),
                    "direct_match_rate": (
                        direct_count / len(df)
                        if len(df)
                        else 0.0
                    ),
                    "direct_match_positive_label": int(
                        direct_positive_count
                    ),
                    "direct_match_negative_label": int(
                        direct_negative_count
                    ),
                    "proxy_match_rows": int(
                        proxy_count
                    ),
                    "proxy_match_rate": (
                        proxy_count / len(df)
                        if len(df)
                        else 0.0
                    ),
                    "proxy_match_positive_label": int(
                        proxy_positive_count
                    ),
                    "proxy_match_negative_label": int(
                        proxy_negative_count
                    ),
                    "direct_examples": json.dumps(
                        direct_examples,
                        ensure_ascii=False,
                    ),
                    "proxy_examples": json.dumps(
                        proxy_examples,
                        ensure_ascii=False,
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Label-Level Audit
# =============================================================================

def build_label_level_audit(
    df: pd.DataFrame,
    row_audit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize direct/proxy matches for every target label.
    """

    rows = []

    for label in LABEL_COLUMNS:

        positive = (
            df[label].astype(int) == 1
        )

        negative = (
            df[label].astype(int) == 0
        )

        direct = row_audit[
            f"{label}__direct_match"
        ]

        proxy = row_audit[
            f"{label}__proxy_match"
        ]

        direct_positive = int(
            (direct & positive).sum()
        )

        direct_negative = int(
            (direct & negative).sum()
        )

        proxy_positive = int(
            (proxy & positive).sum()
        )

        proxy_negative = int(
            (proxy & negative).sum()
        )

        rows.append(
            {
                "label": label,
                "total_rows": int(len(df)),
                "positive_rows": int(
                    positive.sum()
                ),
                "negative_rows": int(
                    negative.sum()
                ),
                "direct_match_rows": int(
                    direct.sum()
                ),
                "direct_match_rate": (
                    float(direct.mean())
                    if len(direct)
                    else 0.0
                ),
                "direct_match_positive_rows": (
                    direct_positive
                ),
                "direct_match_negative_rows": (
                    direct_negative
                ),
                "direct_match_rate_among_positive": (
                    direct_positive
                    / positive.sum()
                    if positive.sum()
                    else 0.0
                ),
                "direct_match_rate_among_negative": (
                    direct_negative
                    / negative.sum()
                    if negative.sum()
                    else 0.0
                ),
                "proxy_match_rows": int(
                    proxy.sum()
                ),
                "proxy_match_rate": (
                    float(proxy.mean())
                    if len(proxy)
                    else 0.0
                ),
                "proxy_match_positive_rows": (
                    proxy_positive
                ),
                "proxy_match_negative_rows": (
                    proxy_negative
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Split-Level Audit
# =============================================================================

def build_split_level_audit(
    df: pd.DataFrame,
    row_audit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize direct/proxy matches separately for train/validation/test.
    """

    rows = []

    for split, split_df in df.groupby(
        "split",
        sort=False,
    ):

        indices = split_df.index

        row_subset = row_audit.loc[
            indices
        ]

        for label in LABEL_COLUMNS:

            direct = row_subset[
                f"{label}__direct_match"
            ]

            proxy = row_subset[
                f"{label}__proxy_match"
            ]

            rows.append(
                {
                    "split": split,
                    "label": label,
                    "rows": int(
                        len(split_df)
                    ),
                    "positive_rows": int(
                        split_df[label]
                        .astype(int)
                        .sum()
                    ),
                    "direct_match_rows": int(
                        direct.sum()
                    ),
                    "direct_match_rate": (
                        float(direct.mean())
                        if len(direct)
                        else 0.0
                    ),
                    "proxy_match_rows": int(
                        proxy.sum()
                    ),
                    "proxy_match_rate": (
                        float(proxy.mean())
                        if len(proxy)
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Global Summary
# =============================================================================

def build_summary(
    df: pd.DataFrame,
    row_audit: pd.DataFrame,
) -> dict:
    """
    Build compact JSON summary.
    """

    direct_columns = [
        f"{label}__direct_match"
        for label in LABEL_COLUMNS
    ]

    proxy_columns = [
        f"{label}__proxy_match"
        for label in LABEL_COLUMNS
    ]

    any_direct = row_audit[
        direct_columns
    ].any(axis=1)

    any_proxy = row_audit[
        proxy_columns
    ].any(axis=1)

    direct_by_label = {}

    for label in LABEL_COLUMNS:

        direct = row_audit[
            f"{label}__direct_match"
        ]

        positive = (
            df[label].astype(int) == 1
        )

        negative = (
            df[label].astype(int) == 0
        )

        direct_positive = (
            direct & positive
        )

        direct_negative = (
            direct & negative
        )

        direct_by_label[label] = {
            "rows_with_direct_match": int(
                direct.sum()
            ),
            "rate": float(
                direct.mean()
            ),
            "positive_rows_with_direct_match": int(
                direct_positive.sum()
            ),
            "negative_rows_with_direct_match": int(
                direct_negative.sum()
            ),
            "rate_among_positive": (
                float(
                    direct_positive.sum()
                    / positive.sum()
                )
                if positive.sum()
                else 0.0
            ),
            "rate_among_negative": (
                float(
                    direct_negative.sum()
                    / negative.sum()
                )
                if negative.sum()
                else 0.0
            ),
        }

    return {
        "dataset": str(DATASET_PATH),
        "cohort": "final_complete_case",
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "complete_case_definition": {
            "photographs_non_empty": True,
            "radiographs_non_empty": True,
            "at_least_one_text_column_non_null": True,
        },
        "expected_complete_case_counts": (
            EXPECTED_COMPLETE_CASE_COUNTS
        ),
        "actual_complete_case_counts": {
            split: int(
                (df["split"] == split).sum()
            )
            for split in EXPECTED_COMPLETE_CASE_COUNTS
        },
        "text_columns_audited": TEXT_COLUMNS,
        "label_columns_audited": LABEL_COLUMNS,
        "rows_with_any_direct_label_match": int(
            any_direct.sum()
        ),
        "rate_with_any_direct_label_match": float(
            any_direct.mean()
        ),
        "rows_with_any_proxy_match": int(
            any_proxy.sum()
        ),
        "rate_with_any_proxy_match": float(
            any_proxy.mean()
        ),
        "direct_matches_by_label": direct_by_label,
        "interpretation": (
            "Direct matches are explicit target-label terms or "
            "unambiguous textual equivalents. They are leakage "
            "candidates and require contextual inspection. "
            "Proxy matches are clinical findings and are not "
            "automatically classified as leakage."
        ),
    }


# =============================================================================
# Main
# =============================================================================

def run_audit(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "CLINICAL TEXT -> SIX-LABEL "
        "LEAKAGE AUDIT"
    )
    print(
        "FINAL COMPLETE-CASE COHORT"
    )
    print("=" * 78)

    print()
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")

    # -------------------------------------------------------------------------
    # Load full authoritative dataset
    # -------------------------------------------------------------------------

    print()
    print("Loading authoritative dataset...")

    full_df = load_dataset(
        dataset_path
    )

    print(
        f"Loaded full dataset: "
        f"{len(full_df):,} rows x "
        f"{len(full_df.columns):,} columns"
    )

    # -------------------------------------------------------------------------
    # Build EXACT final complete-case cohort
    # -------------------------------------------------------------------------

    print()
    print(
        "Applying EXACT final baseline "
        "complete-case filtering..."
    )

    df = build_complete_case_dataset(
        full_df
    )

    print(
        f"Complete-case cohort: "
        f"{len(df):,} rows"
    )

    validate_complete_case_counts(
        df
    )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    print()
    print("Building row-level audit...")
    row_audit = build_row_level_audit(
        df
    )

    print(
        "Building column-level audit..."
    )
    column_audit = build_column_level_audit(
        df
    )

    print(
        "Building label-level audit..."
    )
    label_audit = build_label_level_audit(
        df,
        row_audit,
    )

    print(
        "Building split-level audit..."
    )
    split_audit = build_split_level_audit(
        df,
        row_audit,
    )

    print(
        "Building global summary..."
    )
    summary = build_summary(
        df,
        row_audit,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    row_audit.to_csv(
        output_dir
        / "row_level_audit.csv",
        index=False,
    )

    column_audit.to_csv(
        output_dir
        / "column_level_audit.csv",
        index=False,
    )

    label_audit.to_csv(
        output_dir
        / "label_level_audit.csv",
        index=False,
    )

    split_audit.to_csv(
        output_dir
        / "split_level_audit.csv",
        index=False,
    )

    with open(
        output_dir
        / "audit_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("COMPLETE-CASE AUDIT SUMMARY")
    print("=" * 78)

    print(
        f"Complete-case rows: "
        f"{len(df):,}"
    )

    print(
        f"Rows with ANY direct label match: "
        f"{summary['rows_with_any_direct_label_match']:,} "
        f"({summary['rate_with_any_direct_label_match']:.4%})"
    )

    print(
        f"Rows with ANY proxy match: "
        f"{summary['rows_with_any_proxy_match']:,} "
        f"({summary['rate_with_any_proxy_match']:.4%})"
    )

    print()
    print("Direct matches by label:")

    for label, values in summary[
        "direct_matches_by_label"
    ].items():

        print(
            f"  {label:<32} "
            f"{values['rows_with_direct_match']:>6,} "
            f"({values['rate']:.4%}) "
            f"| positive="
            f"{values['positive_rows_with_direct_match']:>5,} "
            f"({values['rate_among_positive']:.2%}) "
            f"| negative="
            f"{values['negative_rows_with_direct_match']:>5,} "
            f"({values['rate_among_negative']:.2%})"
        )

    print()
    print(
        "Results saved to:"
    )
    print(
        f"  {output_dir}"
    )

    print()
    print("Next inspection files:")
    print(
        "  - column_level_audit.csv"
    )
    print(
        "  - label_level_audit.csv"
    )
    print(
        "  - row_level_audit.csv"
    )
    print(
        "  - split_level_audit.csv"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Direct matches are leakage candidates, "
        "not automatic proof of invalid data."
    )
    print(
        "Context and label value must be inspected "
        "before changing the baseline."
    )

    return summary


if __name__ == "__main__":
    run_audit()