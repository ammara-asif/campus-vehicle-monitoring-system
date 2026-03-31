# Campus Vehicle Monitoring System (CVMS)

A computer vision pipeline for real-time vehicle detection, license plate
recognition, cross-camera tracking, parking occupancy monitoring, and
violation detection on university campuses.

---

## Project Overview

The CVMS integrates five sub-systems into a unified pipeline:

1. **Entry Detection** — Vehicle detection + license plate OCR at entry points
2. **Per-Camera Tracking** — DeepSORT/ByteTrack assigns temporary track IDs
3. **Cross-Camera Matching** — License plate (primary) + time/location (secondary)
4. **Parking Detection** — Slot occupancy monitoring
5. **Violation Detection** — Illegal parking, restricted zone entry

---

## Repository Structure

```
CVMS/
├── data/
│   ├── parking-data/          # Single-slot empty/filled dataset
│   │   ├── empty/              # 3045 empty slot images
│   │   └── filled/             # 3045 filled slot images
│   ├── license_plates/         # Pakistani LP dataset
│
├── preprocessing/
│   └── parking-eda.ipynb    # EDA notebook
│
├── src/
|
├── requirements.txt
└── README.md
```
---

## Dataset Download

### 1. Parking Slot Dataset
We use the **Parking Lot Detection Counter Dataset** by A. Panwhar for parking occupancy detection.

- **Download:** [Parking Lot Detection Counter Dataset on Kaggle](https://www.kaggle.com/datasets/iasadpanwhar/parking-lot-detection-counter?select=parking)
- After downloading, extract and place as:
```
data/
└── parking-data/
    ├── empty/      # 3045 empty slot images
    └── filled/     # 3045 filled slot images
```

Or use the Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d iasadpanwhar/parking-lot-detection-counter
unzip parking-lot-detection-counter.zip -d data/parking-data/
```

---

### 2. Pakistani License Plate Dataset
We use the **Pakistani Car Number Plates Dataset** by Z. Aleemi for license plate recognition.

- **Download:** [Pakistani Car Number Plates Dataset on Kaggle](https://www.kaggle.com/datasets/zakirkhanaleemi/pakistani-car-number-plates-data?select=Pakistani+License+Number+Plates+Data)
- After downloading, extract and place as:
```
data/
└── license_plates/
    ├── images/
    └── annotations/
```

Or use the Kaggle CLI:
```bash
kaggle datasets download -d zakirkhanaleemi/pakistani-car-number-plates-data
unzip pakistani-car-number-plates-data.zip -d data/license_plates/
```

> **Note:** You need a Kaggle account and API token to use the CLI.  
> Place your `kaggle.json` at `~/.kaggle/kaggle.json` before running the commands.

---

## Installation

```bash
git clone https://github.com/ammara-asif/campus-vehicle-monitoring-system.git
cd campus-vehicle-monitoring-system
pip install -r requirements.txt
```
