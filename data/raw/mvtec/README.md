# MVTec AD Dataset Setup Guide

FactoryIQ uses **MVTec AD** (Anomaly Detection) for visual defect inspection across 4 targeted industrial product categories:

1. `bottle/`
2. `screw/`
3. `metal_nut/`
4. `tile/`

## Directory Structure Strategy
Place downloaded category folders under `data/raw/mvtec/`:

```text
data/raw/mvtec/
├── bottle/
│   ├── train/          # Defect-free images for training anomaly model
│   ├── test/           # Normal + defect images (broken_large, contamination, etc.)
│   └── ground_truth/   # Pixel-level defect segmentation masks
├── screw/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── metal_nut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
└── tile/
    ├── train/
    ├── test/
    └── ground_truth/
```

## Synthetic / Fallback Mode
If raw MVTec AD image files are not downloaded locally, FactoryIQ includes synthetic image generators in `src/vision/dataset.py` and fallback feature extractors so the entire pipeline can execute out-of-the-box!
