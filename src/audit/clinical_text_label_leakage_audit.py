"""
Clinical Text -> Six-Label Leakage Audit for the COde Dataset.

Purpose
-------
Audit whether the six target labels used by the final supervised baseline
are explicitly present in the clinical text fields used by the text encoder.

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
   themselves sufficient to establish direct label leakage.

The audit never modifies the dataset.
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
    / "clinical_text_label_leakage_audit"
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


# =============================================================================
# Leakage Vocabulary
# =============================================================================

# These are treated as DIRECT leakage candidates.
#
# We intentionally keep this list conservative.
# A term is considered direct only when it explicitly names the
# disease/condition represented by the target label.

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


# These are NOT automatically considered leakage.
#
# They are reported separately because they may be legitimate clinical
# findings or potential proxies for a target label.
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
    Normalize clinical text for robust keyword matching.

    This is used only for the audit.
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

    # Normalize common punctuation to spaces.
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


def contains_term(text: str, term: str) -> bool:
    """
    Check whether a normalized term occurs as a word/phrase.

    Word boundaries are used for English terms to avoid false matches
    such as 'decay' matching an unrelated longer token.
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
    Return all vocabulary terms found in normalized text.
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
        TEXT_COLUMNS
        + LABEL_COLUMNS
        + ["split"]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing from the six-label dataset:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    return df


# =============================================================================
# Combined Clinical Text
# =============================================================================

def build_combined_text(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Build exactly the clinical text modality used by the audit.

    Only the four approved text columns are included.
    """

    return (
        df[TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .apply(normalize_text)
    )


# =============================================================================
# Row-Level Audit
# =============================================================================

def build_row_level_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a row-level leakage audit.

    For every visit:
    - identify direct label terms
    - identify proxy terms
    - record the columns where each match occurred
    """

    result = df[
        [
            column
            for column in [
                "id",
                "patient_id",
                "checkup_id",
                "split",
            ]
            if column in df.columns
        ]
    ].copy()

    normalized_columns = {
        column: df[column]
        .fillna("")
        .astype(str)
        .apply(normalize_text)
        for column in TEXT_COLUMNS
    }

    combined_text = pd.Series(
        "",
        index=df.index,
    )

    for column in TEXT_COLUMNS:
        combined_text = (
            combined_text
            + " "
            + normalized_columns[column]
        )

    combined_text = combined_text.str.strip()

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
                text = normalized_columns[column].loc[idx]

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
                    row_direct_columns.append(column)

                if proxy:
                    row_proxy_columns.append(column)

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

        result[f"{label}__direct_terms"] = [
            " | ".join(values)
            for values in direct_matches
        ]

        result[f"{label}__direct_columns"] = [
            " | ".join(values)
            for values in direct_columns
        ]

        result[f"{label}__direct_match"] = [
            bool(values)
            for values in direct_matches
        ]

        result[f"{label}__proxy_terms"] = [
            " | ".join(values)
            for values in proxy_matches
        ]

        result[f"{label}__proxy_columns"] = [
            " | ".join(values)
            for values in proxy_columns
        ]

        result[f"{label}__proxy_match"] = [
            bool(values)
            for values in proxy_matches
        ]

    result["any_direct_label_match"] = False
    result["any_proxy_match"] = False

    for label in LABEL_COLUMNS:
        result["any_direct_label_match"] |= result[
            f"{label}__direct_match"
        ]

        result["any_proxy_match"] |= result[
            f"{label}__proxy_match"
        ]

    return result


# =============================================================================
# Per-Column Audit
# =============================================================================

def build_column_level_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure direct and proxy matches separately for every
    clinical text column and every label.
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
                                "terms": direct,
                                "text": str(
                                    df.loc[idx, column]
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
                                "terms": proxy,
                                "text": str(
                                    df.loc[idx, column]
                                )[:500],
                            }
                        )

            rows.append(
                {
                    "text_column": column,
                    "label": label,
                    "total_rows": int(len(df)),
                    "direct_match_rows": int(direct_count),
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
                    "proxy_match_rows": int(proxy_count),
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
    Summarize direct leakage and proxy evidence per label.
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
                "positive_rows": int(positive.sum()),
                "negative_rows": int(negative.sum()),
                "direct_match_rows": int(direct.sum()),
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
                    direct_positive / positive.sum()
                    if positive.sum()
                    else 0.0
                ),
                "direct_match_rate_among_negative": (
                    direct_negative / negative.sum()
                    if negative.sum()
                    else 0.0
                ),
                "proxy_match_rows": int(proxy.sum()),
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
    Summarize direct matches separately for train/val/test.

    This helps verify that explicit target names are not concentrated
    only in one split.
    """

    rows = []

    for split, split_df in df.groupby("split"):
        indices = split_df.index

        row_subset = row_audit.loc[indices]

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
                    "rows": int(len(split_df)),
                    "positive_rows": int(
                        split_df[label].astype(int).sum()
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
    Build a compact JSON summary.
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

        direct_by_label[label] = {
            "rows_with_direct_match": int(
                direct.sum()
            ),
            "rate": float(direct.mean()),
        }

    return {
        "dataset": str(DATASET_PATH),
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "text_columns_audited": TEXT_COLUMNS,
        "label_columns_audited": LABEL_COLUMNS,
        "splits": sorted(
            df["split"].dropna().astype(str).unique().tolist()
        ),
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
            "unambiguous textual equivalents. Proxy matches are "
            "clinical findings and are not automatically classified "
            "as leakage."
        ),
    }


# =============================================================================
# Main
# =============================================================================

def run_audit(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Run the complete clinical text label leakage audit.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("CLINICAL TEXT -> SIX-LABEL LEAKAGE AUDIT")
    print("=" * 78)

    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")
    print()

    print("Loading dataset...")
    df = load_dataset(dataset_path)

    print(
        f"Loaded {len(df):,} rows x {len(df.columns):,} columns"
    )

    print()
    print("Audited clinical text columns:")
    for column in TEXT_COLUMNS:
        print(f"  - {column}")

    print()
    print("Audited target labels:")
    for label in LABEL_COLUMNS:
        print(f"  - {label}")

    print()
    print("Building row-level audit...")
    row_audit = build_row_level_audit(df)

    print("Building column-level audit...")
    column_audit = build_column_level_audit(df)

    print("Building label-level audit...")
    label_audit = build_label_level_audit(
        df,
        row_audit,
    )

    print("Building split-level audit...")
    split_audit = build_split_level_audit(
        df,
        row_audit,
    )

    print("Building summary...")
    summary = build_summary(
        df,
        row_audit,
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    row_audit.to_csv(
        output_dir / "row_level_audit.csv",
        index=False,
    )

    column_audit.to_csv(
        output_dir / "column_level_audit.csv",
        index=False,
    )

    label_audit.to_csv(
        output_dir / "label_level_audit.csv",
        index=False,
    )

    split_audit.to_csv(
        output_dir / "split_level_audit.csv",
        index=False,
    )

    with open(
        output_dir / "audit_summary.json",
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
    print("AUDIT SUMMARY")
    print("=" * 78)

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
            f"({values['rate']:.4%})"
        )

    print()
    print("Results saved to:")
    print(f"  {output_dir}")

    print()
    print("Important:")
    print(
        "A direct match is a leakage candidate, not by itself proof "
        "that the entire text modality is invalid."
    )
    print(
        "Proxy terms are reported separately because they may represent "
        "legitimate clinical evidence."
    )

    return summary


if __name__ == "__main__":
    run_audit()