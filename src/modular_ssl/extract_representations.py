"""
Multi-View Representation Extractor for Modular SSL.

Extracts both visit-level Gated-MIL embeddings AND individual view embeddings
for all 4,195 complete-case visits using:
    1. Best Photograph SSL Encoder (best_photograph_ssl.pt)
    2. Best Radiograph SSL Encoder (best_radiograph_ssl.pt)
    3. Clinical Text SSL Encoder
"""

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from src.baseline.image_loader import COdeImageLoader
from src.ssl_v2.transforms import get_eval_transform
from src.ssl_v2.cloud_sync import KagglePersistenceManager
from . import config
from .dataset import has_value, parse_filenames, build_text
from .models import SingleVisualModalitySSLModel


class CompleteCaseExtractionDataset(Dataset):
    def __init__(self, split: str):
        df = pd.read_csv(config.DATASET_PATH)
        df = df[df["split"] == split].copy()
        df = df[
            df["photographs"].apply(has_value)
            & df["radiographs"].apply(has_value)
            & df[config.TEXT_COLUMNS].notna().any(axis=1)
        ].reset_index(drop=True)

        self.df = df
        self.photo_loader = COdeImageLoader(
            config.IMAGE_ROOT,
            get_eval_transform(config.IMAGE_SIZE, "photograph", use_clahe=False),
        )
        self.radio_loader = COdeImageLoader(
            config.IMAGE_ROOT,
            get_eval_transform(config.IMAGE_SIZE, "radiograph", use_clahe=True),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    def __len__(self) -> int:
        return len(self.df)

    def _load_fixed_views(
        self, fnames: list[str], modality: str, loader: COdeImageLoader, max_v: int
    ):
        selected = fnames[:max_v]
        padded = torch.zeros(
            (max_v, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=torch.float32
        )
        mask = torch.zeros((max_v,), dtype=torch.bool)
        valid_idx = 0
        for fn in selected:
            try:
                img = loader.load(fn, modality)
                padded[valid_idx] = img
                mask[valid_idx] = True
                valid_idx += 1
            except Exception:
                continue
        if valid_idx == 0:
            mask[0] = True
        return padded, mask

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        p_imgs, p_mask = self._load_fixed_views(
            parse_filenames(row["photographs"]),
            "photograph",
            self.photo_loader,
            config.MAX_PHOTO_VIEWS_EXTRACT,
        )
        r_imgs, r_mask = self._load_fixed_views(
            parse_filenames(row["radiographs"]),
            "radiograph",
            self.radio_loader,
            config.MAX_RADIO_VIEWS_EXTRACT,
        )
        enc = self.tokenizer(
            build_text(row),
            padding="max_length",
            truncation=True,
            max_length=config.TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "checkup_id": str(row["checkup_id"]),
            "patient_id": str(row["patient_id"]),
            "photo_images": p_imgs,
            "photo_mask": p_mask,
            "radio_images": r_imgs,
            "radio_mask": r_mask,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                row[config.LABEL_NAMES].astype(float).values,
                dtype=torch.float32,
            ),
        }


@torch.no_grad()
def main():
    device = torch.device(config.DEVICE)
    config.REPRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

    sync_mgr = KagglePersistenceManager(config.PROJECT_ROOT)
    sync_mgr.results_dir = config.OUTPUT_ROOT
    sync_mgr.restore_all_from_backup()

    photo_ckpt_path = config.PHOTO_SSL_DIR / "best_photograph_ssl.pt"
    radio_ckpt_path = config.RADIO_SSL_DIR / "best_radiograph_ssl.pt"

    photo_ssl = SingleVisualModalitySSLModel("photograph").to(device)
    radio_ssl = SingleVisualModalitySSLModel("radiograph").to(device)

    if photo_ckpt_path.exists():
        ckpt_p = torch.load(photo_ckpt_path, map_location="cpu", weights_only=False)
        photo_ssl.load_state_dict(ckpt_p["model_state_dict"])
        print(f"[Loaded] Photograph SSL checkpoint: {photo_ckpt_path}")
    else:
        print(f"[Warning] {photo_ckpt_path} not found! Using ImageNet/ClinicalBERT weights.")

    if radio_ckpt_path.exists():
        ckpt_r = torch.load(radio_ckpt_path, map_location="cpu", weights_only=False)
        radio_ssl.load_state_dict(ckpt_r["model_state_dict"])
        print(f"[Loaded] Radiograph SSL checkpoint: {radio_ckpt_path}")
    else:
        print(f"[Warning] {radio_ckpt_path} not found! Using ImageNet/ClinicalBERT weights.")

    photo_ssl.eval()
    radio_ssl.eval()

    for split in ["train", "validation", "test"]:
        ds = CompleteCaseExtractionDataset(split=split)
        loader = DataLoader(
            ds,
            batch_size=16,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        checkup_ids, patient_ids = [], []
        photo_visits, photo_views_list, photo_masks = [], [], []
        radio_visits, radio_views_list, radio_masks = [], [], []
        text_visits, text_raws, labels_list = [], [], []

        for batch in tqdm(loader, desc=f"Extracting [{split}]"):
            p_imgs = batch["photo_images"].to(device, non_blocking=True)
            p_mask = batch["photo_mask"].to(device, non_blocking=True)
            r_imgs = batch["radio_images"].to(device, non_blocking=True)
            r_mask = batch["radio_mask"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            amask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                p_out = photo_ssl.vision_encoder(p_imgs, p_mask)
                r_out = radio_ssl.vision_encoder(r_imgs, r_mask)
                # Combine text representations from both SSL branches
                t_out_p = photo_ssl.text_encoder(ids, amask)
                t_out_r = radio_ssl.text_encoder(ids, amask)

                t_visit = 0.5 * (
                    t_out_p["visit_embedding"] + t_out_r["visit_embedding"]
                )
                t_raw = 0.5 * (
                    t_out_p["raw_text_feat"] + t_out_r["raw_text_feat"]
                )

            checkup_ids.extend(batch["checkup_id"])
            patient_ids.extend(batch["patient_id"])
            photo_visits.append(p_out["visit_embedding"].float().cpu())
            photo_views_list.append(p_out["projected_views"].float().cpu())
            photo_masks.append(p_mask.cpu())

            radio_visits.append(r_out["visit_embedding"].float().cpu())
            radio_views_list.append(r_out["projected_views"].float().cpu())
            radio_masks.append(r_mask.cpu())

            text_visits.append(t_visit.float().cpu())
            text_raws.append(t_raw.float().cpu())
            labels_list.append(batch["labels"].float().cpu())

        artifact = {
            "checkup_id": checkup_ids,
            "patient_id": patient_ids,
            "photo_visit": torch.cat(photo_visits, dim=0),
            "photo_views": torch.cat(photo_views_list, dim=0),
            "photo_mask": torch.cat(photo_masks, dim=0),
            "radio_visit": torch.cat(radio_visits, dim=0),
            "radio_views": torch.cat(radio_views_list, dim=0),
            "radio_mask": torch.cat(radio_masks, dim=0),
            "text_visit": torch.cat(text_visits, dim=0),
            "text_raw": torch.cat(text_raws, dim=0),
            "labels": torch.cat(labels_list, dim=0),
        }

        out_path = config.REPRESENTATION_DIR / f"{split}_multiview_rep.pt"
        torch.save(artifact, out_path)
        sync_mgr.backup_file(out_path)
        print(f"Saved {split} representations -> {out_path}")


if __name__ == "__main__":
    main()