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

### Dataset Setup

1. Place single-slot parking images under `data/parking_slots/empty/` and
   `data/parking_slots/filled/`
2. Place license plate images under `data/license_plates/`
3. Campus feed footage goes under `data/campus_feed/` once available

---
## Installation

```bash
git clone https://github.com/your-team/cvms.git
cd cvms
pip install -r requirements.txt
```
