"""
SSL embedding similarity evaluation.

Positive:
same visit modality pairs

Negative:
random shuffled modality pairs
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.ssl.dataset import COdeSSLDataset
from src.ssl.collate import ssl_collate
from src.ssl.model import MultimodalSSLModel
from src.ssl.tokenizer import ClinicalTokenizer
from src.ssl.utils import load_modality_batch

from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


CSV = (
    "results/"
    "labeled_patient_level_dataset/"
    "labeled_dataset.csv"
)


CHECKPOINT = (
    "results/"
    "ssl_pretraining/"
    "multimodal_dynamic/"
    "best_ssl_model.pt"
)


OUT = Path(
    "results/"
    "ssl_pretraining/"
    "similarity_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


BATCH_SIZE = 16


def cosine(a, b):
    return F.cosine_similarity(
        a,
        b
    ).mean().item()



def main():

    print(
        f"Device: {DEVICE}"
    )


    dataset = COdeSSLDataset(
        CSV,
        split="validation"
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=ssl_collate,
        num_workers=2,
        pin_memory=True,
    )


    model = MultimodalSSLModel().to(
        DEVICE
    )


    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        state
    )

    model.eval()


    tokenizer = ClinicalTokenizer()

    transform = get_image_transform()

    image_loader = COdeImageLoader(
        "data/raw/COde-Dataset/Images"
    )


    results = {

        "image_text": {
            "positive": [],
            "negative": []
        },

        "image_radiograph": {
            "positive": [],
            "negative": []
        },

        "radiograph_text": {
            "positive": [],
            "negative": []
        }
    }


    with torch.no_grad():

        for batch in tqdm(loader):


            images, image_mask = load_modality_batch(
                batch["images"],
                "photograph",
                image_loader,
                transform,
                DEVICE
            )


            radiographs, radiograph_mask = load_modality_batch(
                batch["radiographs"],
                "radiograph",
                image_loader,
                transform,
                DEVICE
            )


            tokens = tokenizer(
                batch["text"]
            )


            input_ids = tokens["input_ids"].to(
                DEVICE
            )

            attention = tokens["attention_mask"].to(
                DEVICE
            )


            image_z = F.normalize(
                model.project_image(images),
                dim=1
            )


            radiograph_z = F.normalize(
                model.project_radiograph(radiographs),
                dim=1
            )


            text_z = F.normalize(
                model.project_text(
                    input_ids,
                    attention
                ),
                dim=1
            )


            B = image_z.size(0)


            # positive pairs

            for i in range(B):

                if image_mask[i] and batch["has_text"][i]:

                    results["image_text"]["positive"].append(
                        cosine(
                            image_z[i:i+1],
                            text_z[i:i+1]
                        )
                    )


                if image_mask[i] and radiograph_mask[i]:

                    results["image_radiograph"]["positive"].append(
                        cosine(
                            image_z[i:i+1],
                            radiograph_z[i:i+1]
                        )
                    )


                if radiograph_mask[i] and batch["has_text"][i]:

                    results["radiograph_text"]["positive"].append(
                        cosine(
                            radiograph_z[i:i+1],
                            text_z[i:i+1]
                        )
                    )


            # negative by shuffle

            perm = torch.randperm(B)


            results["image_text"]["negative"].extend(
                F.cosine_similarity(
                    image_z,
                    text_z[perm]
                ).cpu().tolist()
            )


            results["image_radiograph"]["negative"].extend(
                F.cosine_similarity(
                    image_z,
                    radiograph_z[perm]
                ).cpu().tolist()
            )


            results["radiograph_text"]["negative"].extend(
                F.cosine_similarity(
                    radiograph_z,
                    text_z[perm]
                ).cpu().tolist()
            )



    summary = {}


    for pair, values in results.items():

        summary[pair] = {

            "positive_mean":
            sum(values["positive"])
            /
            len(values["positive"])
            if values["positive"]
            else 0,


            "negative_mean":
            sum(values["negative"])
            /
            len(values["negative"])
            if values["negative"]
            else 0,


            "positive_count":
            len(values["positive"]),

            "negative_count":
            len(values["negative"])

        }



    with open(
        OUT/"similarity_results.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":
    main()