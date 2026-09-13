import torch

def complete_case_collate(batch):
    out = {
        "checkup_id": [x["checkup_id"] for x in batch],
        "patient_id": [x["patient_id"] for x in batch],
        "labels": torch.stack([x["labels"] for x in batch]),
    }
    if "images" in batch[0]:
        out["images"] = [x["images"] for x in batch]
    if "radiographs" in batch[0]:
        out["radiographs"] = [x["radiographs"] for x in batch]
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([x["input_ids"] for x in batch])
        out["attention_mask"] = torch.stack([x["attention_mask"] for x in batch])
    return out
