"""
A2 Text Leakage Audit for the COde Dataset.

This audit evaluates whether the sanitized clinical text retains
substantial predictive information using a simple classical model:

    TF-IDF + One-vs-Rest Logistic Regression

The audit uses the same complete-case cohort and patient-level split
as the final supervised multimodal benchmark.

Experiments:
    1. All four model-eligible clinical text fields.
    2. Each field individually.

The TF-IDF vocabulary is fitted on the training split only.
No test-set information is used for fitting or model selection.

The six diagnostic targets are evaluated independently.

Outputs:
    results/audit/a2_text_leakage/
        summary.csv
        per_label_results.csv
        run_config.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "audit"
    / "a2_text_leakage"
)


# =============================================================================
# Configuration
# =============================================================================

TEXT_FIELDS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]

LABEL_NAMES = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]

SPLITS = [
    "train",
    "validation",
    "test",
]

EXPECTED_COUNTS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}

DEFAULT_THRESHOLD = 0.5

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 50000,
}

LOGISTIC_REGRESSION_CONFIG = {
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": 42,
}


# =============================================================================
# Dataset
# =============================================================================

def is_non_empty(value) -> bool:
    """Return True when a value contains usable text."""
    if pd.isna(value):
        return False

    text = str(value).strip()

    if text == "":
        return False

    if text.lower() in {"nan", "none", "null"}:
        return False

    return True


def load_complete_case_dataset() -> pd.DataFrame:
    """
    Load the canonical labeled dataset and construct the exact
    complete-case cohort used by the supervised benchmark.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "split",
        "photographs",
        "radiographs",
        *TEXT_FIELDS,
        *LABEL_NAMES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Same complete-case definition as final_linear_v3.
    complete_case_mask = (
        df["photographs"].apply(is_non_empty)
        & df["radiographs"].apply(is_non_empty)
        & df[TEXT_FIELDS].apply(
            lambda row: any(
                is_non_empty(value)
                for value in row
            ),
            axis=1,
        )
    )

    df = df.loc[
        complete_case_mask
    ].reset_index(drop=True)

    for split, expected in EXPECTED_COUNTS.items():
        actual = int(
            (df["split"] == split).sum()
        )

        if actual != expected:
            raise RuntimeError(
                f"Complete-case mismatch for {split}: "
                f"expected {expected}, found {actual}"
            )

    return df


# =============================================================================
# Text construction
# =============================================================================

def build_text(
    df: pd.DataFrame,
    fields: list[str],
) -> pd.Series:
    """
    Concatenate selected clinical text fields in the same order
    used by the supervised text pipeline.
    """

    return (
        df[fields]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: " ".join(
                value.strip()
                for value in row
                if value.strip()
            ),
            axis=1,
        )
    )


# =============================================================================
# Metrics
# =============================================================================

def safe_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute AUROC safely.

    A value of NaN is returned if a target contains only one class.
    """

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            y_score,
        )
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[dict, list[dict]]:
    """
    Calculate overall and per-label metrics at threshold 0.5.
    """

    y_pred = (
        y_score >= DEFAULT_THRESHOLD
    ).astype(int)

    per_label = []

    for index, label_name in enumerate(LABEL_NAMES):
        label_true = y_true[:, index]
        label_pred = y_pred[:, index]
        label_score = y_score[:, index]

        per_label.append(
            {
                "label": label_name,
                "f1": float(
                    f1_score(
                        label_true,
                        label_pred,
                        zero_division=0,
                    )
                ),
                "auroc": safe_roc_auc(
                    label_true,
                    label_score,
                ),
                "positive_count": int(
                    label_true.sum()
                ),
            }
        )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    micro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )

    label_aurocs = [
        item["auroc"]
        for item in per_label
        if not np.isnan(item["auroc"])
    ]

    macro_auroc = (
        float(np.mean(label_aurocs))
        if label_aurocs
        else float("nan")
    )

    summary = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_auroc": macro_auroc,
    }

    return summary, per_label


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    fields: list[str],
) -> tuple[dict, list[dict]]:
    """
    Run one TF-IDF + Logistic Regression experiment.

    TF-IDF is fitted exclusively on the training split.
    """

    print()
    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)
    print(f"Fields: {fields}")

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "validation"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    train_text = build_text(
        train_df,
        fields,
    )

    val_text = build_text(
        val_df,
        fields,
    )

    test_text = build_text(
        test_df,
        fields,
    )

    y_train = train_df[
        LABEL_NAMES
    ].astype(int).to_numpy()

    y_val = val_df[
        LABEL_NAMES
    ].astype(int).to_numpy()

    y_test = test_df[
        LABEL_NAMES
    ].astype(int).to_numpy()

    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )

    print("Fitting TF-IDF on training text only...")

    vectorizer = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    x_train = vectorizer.fit_transform(
        train_text
    )

    x_val = vectorizer.transform(
        val_text
    )

    x_test = vectorizer.transform(
        test_text
    )

    print(
        f"TF-IDF matrix: "
        f"{x_train.shape[0]} samples × "
        f"{x_train.shape[1]} features"
    )

    print("Training One-vs-Rest Logistic Regression...")

    base_classifier = LogisticRegression(
        **LOGISTIC_REGRESSION_CONFIG
    )

    classifier = OneVsRestClassifier(
        base_classifier
    )

    classifier.fit(
        x_train,
        y_train,
    )

    print("Generating predictions...")

    val_scores = classifier.predict_proba(
        x_val
    )

    test_scores = classifier.predict_proba(
        x_test
    )

    val_summary, val_per_label = evaluate_predictions(
        y_val,
        val_scores,
    )

    test_summary, test_per_label = evaluate_predictions(
        y_test,
        test_scores,
    )

    print()
    print(
        f"Validation Macro F1: "
        f"{val_summary['macro_f1']:.4f}"
    )
    print(
        f"Validation Macro AUROC: "
        f"{val_summary['macro_auroc']:.4f}"
    )
    print(
        f"Test Macro F1: "
        f"{test_summary['macro_f1']:.4f}"
    )
    print(
        f"Test Micro F1: "
        f"{test_summary['micro_f1']:.4f}"
    )
    print(
        f"Test Macro AUROC: "
        f"{test_summary['macro_auroc']:.4f}"
    )

    summary_row = {
        "experiment": experiment_name,
        "fields": " + ".join(fields),
        "num_fields": len(fields),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "tfidf_features": int(
            x_train.shape[1]
        ),
        "validation_macro_f1": val_summary[
            "macro_f1"
        ],
        "validation_micro_f1": val_summary[
            "micro_f1"
        ],
        "validation_macro_auroc": val_summary[
            "macro_auroc"
        ],
        "test_macro_f1": test_summary[
            "macro_f1"
        ],
        "test_micro_f1": test_summary[
            "micro_f1"
        ],
        "test_macro_auroc": test_summary[
            "macro_auroc"
        ],
    }

    per_label_rows = []

    for val_item, test_item in zip(
        val_per_label,
        test_per_label,
    ):
        per_label_rows.append(
            {
                "experiment": experiment_name,
                "fields": " + ".join(fields),
                "label": val_item["label"],
                "validation_f1": val_item["f1"],
                "validation_auroc": val_item[
                    "auroc"
                ],
                "test_f1": test_item["f1"],
                "test_auroc": test_item[
                    "auroc"
                ],
                "test_positive_count": test_item[
                    "positive_count"
                ],
            }
        )

    return summary_row, per_label_rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("A2 TEXT LEAKAGE AUDIT")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_PATH}")

    df = load_complete_case_dataset()

    print()
    print(
        f"Complete-case cohort: {len(df):,} visits"
    )

    print(
        df["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    experiments = {
        "all_four_fields": TEXT_FIELDS,
    }

    for field in TEXT_FIELDS:
        experiments[
            f"single_{field}"
        ] = [field]

    all_summary_rows = []
    all_per_label_rows = []

    for experiment_name, fields in experiments.items():
        summary_row, per_label_rows = run_experiment(
            df=df,
            experiment_name=experiment_name,
            fields=fields,
        )

        all_summary_rows.append(
            summary_row
        )

        all_per_label_rows.extend(
            per_label_rows
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        all_summary_rows
    )

    per_label_df = pd.DataFrame(
        all_per_label_rows
    )

    summary_path = (
        OUTPUT_DIR / "summary.csv"
    )

    per_label_path = (
        OUTPUT_DIR / "per_label_results.csv"
    )

    config_path = (
        OUTPUT_DIR / "run_config.json"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    per_label_df.to_csv(
        per_label_path,
        index=False,
    )

    run_config = {
        "dataset_path": str(
            DATASET_PATH
        ),
        "dataset_rows": int(len(df)),
        "split_counts": {
            split: int(
                (df["split"] == split).sum()
            )
            for split in SPLITS
        },
        "text_fields": TEXT_FIELDS,
        "label_names": LABEL_NAMES,
        "experiments": experiments,
        "tfidf": TFIDF_CONFIG,
        "logistic_regression": (
            LOGISTIC_REGRESSION_CONFIG
        ),
        "threshold": DEFAULT_THRESHOLD,
        "fit_policy": (
            "TF-IDF and classifiers are fitted "
            "using training data only."
        ),
        "model_selection_policy": (
            "No test-set model selection or "
            "threshold optimization."
        ),
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_config,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("A2 COMPLETE")
    print("=" * 70)

    print()
    print("Summary:")
    print(
        summary_df[
            [
                "experiment",
                "test_macro_f1",
                "test_micro_f1",
                "test_macro_auroc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
