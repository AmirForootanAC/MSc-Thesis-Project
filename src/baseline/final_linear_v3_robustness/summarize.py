"""Aggregate final_linear_v3 random-seed robustness results.

The summary uses:
    seed 42  -> existing final_linear_v3 reference result
    seed 123 -> robustness run, if available
    seed 456 -> robustness run, if available

Mean and SD are calculated only when results are available.
No missing robustness run is treated as zero.
"""

import json
from pathlib import Path

import pandas as pd


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)


REFERENCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_linear_v3"
)


ROBUSTNESS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_linear_v3_robustness"
)


# ======================================================================
# All seven final V3 configurations
# ======================================================================

CONFIGURATIONS = {
    "text": {
        "display_name":
            "Clinical Text",

        "reference_scenario":
            "text_only",

        "reference_path":
            REFERENCE_ROOT
            / "text_only"
            / "final_results.json",
    },

    "photo": {
        "display_name":
            "Photograph",

        "reference_scenario":
            "image_only",

        "reference_path":
            REFERENCE_ROOT
            / "image_only"
            / "final_results.json",
    },

    "radiograph": {
        "display_name":
            "Radiograph",

        "reference_scenario":
            "xray_only",

        "reference_path":
            REFERENCE_ROOT
            / "xray_only"
            / "final_results.json",
    },

    "photo_text": {
        "display_name":
            "Photograph + Text",

        "reference_scenario":
            "image_text",

        "reference_path":
            REFERENCE_ROOT
            / "image_text"
            / "final_results.json",
    },

    "photo_radiograph": {
        "display_name":
            "Photograph + Radiograph",

        "reference_scenario":
            "image_xray",

        "reference_path":
            REFERENCE_ROOT
            / "image_xray"
            / "final_results.json",
    },

    "radiograph_text": {
        "display_name":
            "Radiograph + Text",

        "reference_scenario":
            "text_xray",

        "reference_path":
            REFERENCE_ROOT
            / "text_xray"
            / "final_results.json",
    },

    "full_multimodal": {
        "display_name":
            "Full Multimodal",

        "reference_scenario":
            "full_multimodal",

        "reference_path":
            REFERENCE_ROOT
            / "full_multimodal"
            / "final_results.json",
    },
}


NEW_SEEDS = [
    123,
    456,
]


METRICS = [
    "macro_f1",
    "micro_f1",
    "auroc",
    "accuracy",
]


# ======================================================================
# Helpers
# ======================================================================

def robustness_path(
    configuration,
    seed,
):
    return (
        ROBUSTNESS_ROOT
        / f"{configuration}_seed{seed}"
        / "final_results.json"
    )


def load_json(path):
    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def extract_test_metrics(result):
    if result is None:
        return None

    return result[
        "test_default"
    ]


# ======================================================================
# Main
# ======================================================================

def main():
    ROBUSTNESS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_rows = []

    # ================================================================
    # Load seed 42 reference + any available robustness seeds.
    # ================================================================

    for configuration, info in (
        CONFIGURATIONS.items()
    ):

        # ------------------------------------------------------------
        # Seed 42
        # ------------------------------------------------------------

        reference_result = load_json(
            info[
                "reference_path"
            ]
        )

        if reference_result is None:
            raise FileNotFoundError(
                "\nMissing existing seed=42 reference result:\n"
                f"  {info['reference_path']}"
            )

        reference_metrics = (
            extract_test_metrics(
                reference_result
            )
        )

        for metric in METRICS:
            long_rows.append(
                {
                    "configuration":
                        configuration,

                    "display_name":
                        info[
                            "display_name"
                        ],

                    "seed":
                        42,

                    "seed_role":
                        "reference",

                    "metric":
                        metric,

                    "value":
                        float(
                            reference_metrics[
                                metric
                            ]
                        ),
                }
            )

        # ------------------------------------------------------------
        # Seeds 123 and 456
        # ------------------------------------------------------------

        for seed in NEW_SEEDS:

            path = robustness_path(
                configuration,
                seed,
            )

            result = load_json(
                path
            )

            if result is None:
                print(
                    "[SKIP] Missing robustness result:"
                )

                print(
                    f"       {path}"
                )

                continue

            metrics = extract_test_metrics(
                result
            )

            for metric in METRICS:
                long_rows.append(
                    {
                        "configuration":
                            configuration,

                        "display_name":
                            info[
                                "display_name"
                            ],

                        "seed":
                            seed,

                        "seed_role":
                            "robustness",

                        "metric":
                            metric,

                        "value":
                            float(
                                metrics[
                                    metric
                                ]
                            ),
                    }
                )

    long_df = pd.DataFrame(
        long_rows
    )

    if long_df.empty:
        raise RuntimeError(
            "No robustness/reference results were found."
        )

    # ================================================================
    # Seed-level wide table
    # ================================================================

    seed_wide = (
        long_df
        .pivot_table(
            index=[
                "configuration",
                "display_name",
                "metric",
            ],
            columns="seed",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    seed_wide = seed_wide.rename(
        columns={
            42:
                "seed_42",

            123:
                "seed_123",

            456:
                "seed_456",
        }
    )

    seed_wide_path = (
        ROBUSTNESS_ROOT
        / "seed_results_by_seed.csv"
    )

    seed_wide.to_csv(
        seed_wide_path,
        index=False,
    )

    # ================================================================
    # Mean ± SD across available seeds
    #
    # IMPORTANT:
    # For a configuration to be used as a three-seed robustness
    # estimate, all three seeds must be present.
    #
    # We do not calculate a "mean ± SD" from 1 or 2 seeds and label
    # it as the requested three-seed result.
    # ================================================================

    summary_rows = []

    for (
        configuration,
        display_name,
        metric,
    ), group in long_df.groupby(
        [
            "configuration",
            "display_name",
            "metric",
        ],
        sort=False,
    ):

        available_seeds = sorted(
            group["seed"]
            .astype(int)
            .unique()
            .tolist()
        )

        values = (
            group["value"]
            .to_numpy(
                dtype=float
            )
        )

        has_three_seeds = (
            available_seeds
            == [42, 123, 456]
        )

        if has_three_seeds:
            mean = values.mean()

            sd = values.std(
                ddof=1
            )

            mean_sd = (
                f"{mean:.4f} ± {sd:.4f}"
            )

        else:
            mean = None
            sd = None
            mean_sd = None

        summary_rows.append(
            {
                "configuration":
                    configuration,

                "display_name":
                    display_name,

                "metric":
                    metric,

                "available_seeds":
                    ",".join(
                        map(
                            str,
                            available_seeds,
                        )
                    ),

                "n_seeds":
                    len(
                        available_seeds
                    ),

                "three_seed_complete":
                    has_three_seeds,

                "mean":
                    mean,

                "sd":
                    sd,

                "mean_plus_minus_sd":
                    mean_sd,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        ROBUSTNESS_ROOT
        / "seed_results_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ================================================================
    # Publication-friendly table.
    #
    # Only configurations with all 3 seeds are included.
    # ================================================================

    complete_df = summary_df[
        summary_df[
            "three_seed_complete"
        ]
        == True
    ].copy()

    publication_rows = []

    for configuration in (
        CONFIGURATIONS.keys()
    ):

        subset = complete_df[
            complete_df[
                "configuration"
            ]
            == configuration
        ]

        if subset.empty:
            continue

        first = subset.iloc[0]

        row = {
            "configuration":
                configuration,

            "display_name":
                first[
                    "display_name"
                ],
        }

        for metric in METRICS:
            metric_rows = subset[
                subset[
                    "metric"
                ]
                == metric
            ]

            if metric_rows.empty:
                row[metric] = None
            else:
                row[metric] = metric_rows.iloc[0][
                    "mean_plus_minus_sd"
                ]

        publication_rows.append(
            row
        )

    publication_df = pd.DataFrame(
        publication_rows
    )

    publication_path = (
        ROBUSTNESS_ROOT
        / "seed_results_mean_sd.csv"
    )

    publication_df.to_csv(
        publication_path,
        index=False,
    )

    # ================================================================
    # JSON output
    # ================================================================

    json_path = (
        ROBUSTNESS_ROOT
        / "seed_results_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary_rows,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ================================================================
    # Console
    # ================================================================

    print("\n" + "=" * 80)
    print(
        "FINAL LINEAR V3 RANDOM-SEED ROBUSTNESS SUMMARY"
    )
    print("=" * 80)

    print(
        "\nAvailable seed results:"
    )

    print(
        seed_wide.to_string(
            index=False
        )
    )

    print(
        "\nThree-seed mean ± SD:"
    )

    if publication_df.empty:
        print(
            "No configuration has all three seeds yet."
        )
    else:
        print(
            publication_df.to_string(
                index=False
            )
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {seed_wide_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {publication_path}"
    )

    print(
        f"  {json_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()