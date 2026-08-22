"""
Evaluate text-only baseline.
"""

import json
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
)

from src.baseline import config

from src.baseline.text_dataset import (
    COdeTextDataset,
)

from src.baseline.text_collate import (
    text_collate,
)

from src.baseline.text_model import (
    TextOnlyBaseline,
)



def main():

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    dataset = COdeTextDataset(
        csv_path=config.DATASET_PATH,
        split=config.TEST_SPLIT,
    )


    loader=DataLoader(
        dataset,
        batch_size=config.TEXT_BATCH_SIZE,
        shuffle=False,
        collate_fn=text_collate,
    )


    model = TextOnlyBaseline(
        model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )


    ckpt=torch.load(
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
        /
        "best_model.pt",
        map_location=device,
    )


    model.load_state_dict(
        ckpt["model_state"]
    )

    model.to(device)

    model.eval()


    logits_all=[]
    labels_all=[]


    with torch.no_grad():

        for batch in loader:

            ids=batch["input_ids"].to(device)

            mask=batch["attention_mask"].to(device)

            labels=batch["labels"]


            logits=model(
                ids,
                mask
            )


            logits_all.append(
                logits.cpu()
            )

            labels_all.append(
                labels
            )


    logits=torch.cat(logits_all)

    labels=torch.cat(labels_all)


    probs=torch.sigmoid(logits).numpy()

    labels=labels.numpy()


    preds=(probs>=0.5)


    metrics={

        "macro_f1":
        f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0,
        ),

        "micro_f1":
        f1_score(
            labels,
            preds,
            average="micro",
            zero_division=0,
        ),

        "accuracy":
        accuracy_score(
            labels,
            preds,
        ),

        "auroc":
        roc_auc_score(
            labels,
            probs,
            average="macro",
        )
    }

    output_dir = (
    config.RESULT_ROOT
    /
    config.TEXT_EXPERIMENT_NAME
)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "test_metrics.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(
        "Saved:",
        output_dir / "test_metrics.json"
    )

    print(metrics)


    with open(
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
        /
        "test_evaluation.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


if __name__=="__main__":
    main()