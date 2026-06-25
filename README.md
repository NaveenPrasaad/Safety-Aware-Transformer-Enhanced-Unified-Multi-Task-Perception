# Safety-Aware Multitask Perception for Autonomous Driving in Unstructured Indian Roads

## Overview

This repository contains the implementation of a **Safety-Aware Multitask Perception Network** designed for autonomous driving in unstructured road environments commonly found in developing countries such as India.

The proposed framework jointly performs:

- Object Detection
- Drivable Area Segmentation
- Lane Line Segmentation

using a shared feature extraction backbone and task-specific decoders. The network incorporates a Transformer-enhanced feature encoder and a safety-aware refinement module that improves drivable area estimation by explicitly accounting for road anomalies.

---

## Motivation

Autonomous driving systems developed for structured road environments often face significant challenges when deployed on Indian roads due to:

- Potholes
- Waterlogging
- Road cracks
- Mixed traffic participants
- Drivable  Area
- Lane markings

To address these challenges, we propose a multitask perception framework that simultaneously understands:

1. What objects are present?
2. Which regions are safe to drive on?
3. Where are the lane markings?

while integrating explicit safety constraints into drivable area estimation.

---

## Proposed Architecture

### Shared Encoder

The network utilizes:

- YOLOv8 backbone
- C3TR Transformer module
- Multi-scale feature extraction

The Transformer-enhanced encoder improves long-range dependency modeling and scene understanding in complex road environments.

### Task-Specific Heads

The shared features are processed through three dedicated decoders:

#### 1. Object Detection

Detects:

- Potholes
- Waterlogging
- Road Cracks
- Speed Bumps
- Vehicles
- Pedestrians
- Motorcycles
- Autorickshaws
- Other road participants

#### 2. Drivable Area Segmentation

Predicts safe navigable regions for autonomous vehicles.

#### 3. Lane Line Segmentation

Identifies lane boundaries and road guidance markings.

---

## Safety-Aware Refinement

A deterministic safety-aware refinement module is integrated during inference.

### Critical Hazards

The following hazards are removed from the drivable area:

- Potholes
- Waterlogging

### Navigable Hazards

The following remain inside the drivable area while being treated as alert regions:

- Speed bumps

This improves practical safety by preventing hazardous regions from being classified as drivable.

---

## Dataset

### CMIRD (Custom Multitask Indian Road Dataset)

The proposed model was trained and evaluated on a custom dataset collected from:

- Thiruvananthapuram, India
- Nagercoil, India

The dataset contains:

- Urban roads
- Semi-urban roads
- Structured roads
- Unstructured roads
- Diverse weather conditions
- Road anomalies

### Additional Dataset

- BDD100K

A dataset combination utility is provided for integrating custom YOLO datasets with BDD100K.

---

## Experimental Results

| Task | Metric | Performance |
|--------|--------|-------------|
| Object Detection | mAP@50 | 0.877 |
| Drivable Area Segmentation | mIoU | 0.962 |
| Lane Line Segmentation | IoU | 0.447 |

### Key Findings

- 1.9% improvement in anomaly detection over A-YOLOM(s)
- Improved safety-aware drivable area estimation
- Competitive lane detection performance
- Robust performance across structured and unstructured road environments
- Strong cross-region generalization capability

---

## Repository Structure

```text
├── predictor_multi.py          # Multitask inference pipeline
├── combine-dataset.py          # Dataset merging utility
├── requirements.txt
├── setup.py
├── LICENSE
├── README.md
└── ...
```

---

## Installation


### Create Environment

```bash
conda create -n multitask_perception python=3.9
conda activate multitask_perception
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Inference

Run multitask prediction:

```bash
python predictor_multi.py
```

The framework generates:

- Object detection results
- Drivable area masks
- Lane segmentation masks
- Safety-aware refined drivable regions

---

## Dataset Preparation

To combine a custom YOLO dataset with BDD100K:

```bash
python combine-dataset.py
```

Update dataset paths in:

```python
bdd100k_dataset_root = "path/to/bdd100k"
custom_yolo_dataset_root = "path/to/custom_dataset"
```

before execution.

---

## Acknowledgements

This work utilizes:

- YOLOv8
- BDD100K Dataset
- PyTorch

We thank the contributors and researchers whose open-source tools enabled this research.
