# Floorplan Understanding Pipeline (OpenCV + SAM + Vectorization)

## Overview

This project implements a computer vision pipeline for **floorplan analysis and vectorization**, combining classical image processing with modern deep learning segmentation using Meta’s Segment Anything Model (SAM).

The goal is to transform raw floorplan images into structured geometric representations, including:

* Wall segmentation
* Region (room-like) masks
* Polygon extraction
* Graph-based structural representation
* JSON export for downstream applications

---

## Pipeline Overview

```
Input Floorplan Image
        ↓
Perspective Correction (Top-down normalization)
        ↓
Preprocessing (contrast, thresholding, morphology)
        ↓
Classical CV (edges → contours → boxes)
        ↓
Polygon Extraction
        ↓
Graph Construction (walls / connectivity)
        ↓
JSON Export
```

## Project Structure

```
project/
│
├── notebooks/
│   ├── 01_capture_to_topdown.ipynb
│   ├── 02_image_preprocessing.ipynb
│   ├── 03_segmentation_opencv.ipynb
│   ├── 04_polygon_vectorization.ipynb
│   └── 05_sam_segmentation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── io_utils.py
│   └── visualization.py
│
├── checkpoints/
│   └── sam_vit_b.pth   (not committed to git)
│
├── outputs/
│
├── requirements.txt
└── README.md
```

---

## Installation

### Clone repository

```bash
git clone https://github.com/your-username/floorplan-preprocessing-pipeline.git
cd floorplan-preprocessing-pipeline
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Tech Stack

* Python
* OpenCV
* NumPy
* PyTorch
* Segment Anything Model (SAM)
* Matplotlib