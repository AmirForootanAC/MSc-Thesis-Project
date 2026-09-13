# Corrected Supervised Baselines

This directory contains the corrected apples-to-apples supervised benchmark.

Every scenario uses the same complete-case population (photograph + radiograph + at least one permitted clinical-text field) within the existing patient-level split. No new split is created.

Expected population:
- train: 2935
- validation: 627
- test: 633

Scenarios:
- text_only
- image_only
- xray_only
- image_text
- image_xray
- text_xray
- full_multimodal

Change only `ACTIVE_SCENARIO` in `config.py`, then run:

```bash
python -m src.baseline.final.train
```

Text uses only:
- chief_complaint
- present_illness
- past_medical_record
- examination

It never uses `anomalies_en`, `diagnosis`, `treatment_plan`, or `management`.

Outputs are written to `results/baseline/final/<scenario>/`.
