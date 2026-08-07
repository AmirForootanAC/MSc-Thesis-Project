"""
End-to-end validation for the COde multimodal data pipeline.

Checks:
- sample count
- modality availability
- missing modality consistency
- patient split integrity
- deterministic behavior
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.code_dataset import COdeDataset


CONFIG_PATH = "configs/base.yaml"


def load_config():

    with open(
        CONFIG_PATH,
        "r",
    ) as f:

        return yaml.safe_load(f)



def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



def validate_sample_count(dataset):

    return {
        "total_samples": len(dataset)
    }



def validate_modalities(dataset):

    photographs = 0
    radiographs = 0
    clinical_text = 0


    for idx in range(len(dataset)):

        sample = dataset[idx]


        if not sample.missing_flags[
            "photographs_missing"
        ]:
            photographs += 1


        if not sample.missing_flags[
            "radiographs_missing"
        ]:
            radiographs += 1


        if not sample.missing_flags[
            "clinical_text_missing"
        ]:
            clinical_text += 1


    return {
        "samples_with_photographs": photographs,
        "samples_with_radiographs": radiographs,
        "samples_with_clinical_text": clinical_text,
    }



def validate_missingness(dataset):

    result = {
        "photographs_missing": 0,
        "radiographs_missing": 0,
        "clinical_text_missing": 0,
    }


    for idx in range(len(dataset)):

        flags = dataset[idx].missing_flags


        for key in result:

            if flags[key]:

                result[key] += 1


    return result



def validate_patient_split(csv_path):

    df = pd.read_csv(
        csv_path
    )


    leakage = (
        df.groupby("patient_id")["split"]
        .nunique()
    )


    leaked_patients = int(
        (leakage > 1).sum()
    )


    return {
        "patients_with_split_leakage": leaked_patients
    }



def validate_deterministic(dataset, seed):

    set_seed(seed)

    first_run = [
        dataset[i].visit_id
        for i in range(10)
    ]


    set_seed(seed)

    second_run = [
        dataset[i].visit_id
        for i in range(10)
    ]


    return {
        "deterministic": (
            first_run == second_run
        ),

        "checked_samples": 10,
    }



def run_validation():

    config = load_config()


    set_seed(
        config["project"]["seed"]
    )


    dataset = COdeDataset(
        config["paths"]["dataset_csv"]
    )


    report = {

        "sample_count":
            validate_sample_count(
                dataset
            ),

        "modalities":
            validate_modalities(
                dataset
            ),

        "missingness":
            validate_missingness(
                dataset
            ),

        "patient_assignment":
            validate_patient_split(
                config["paths"]["patient_split"]
            ),

        "deterministic":
            validate_deterministic(
                dataset,
                config["project"]["seed"],
            ),
    }


    return report



def main():

    output_dir = Path(
        "results/pipeline_validation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    report = run_validation()


    output_file = (
        output_dir
        /
        "pipeline_validation_report.json"
    )


    with open(
        output_file,
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )


    print()
    print("=" * 60)
    print(
        "Pipeline validation completed."
    )
    print("=" * 60)

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print()

    print(
        f"Results saved to: {output_dir}"
    )



if __name__ == "__main__":

    main()