## ConDo: *Con*trastive *Do*main Generalization For Domain Shift In Histopathology Images
<img width="677" height="486" alt="Screenshot 2025-12-08 at 11 53 42 PM" src="https://github.com/user-attachments/assets/3af069ec-5142-4046-bfe4-ecc1602d8019" />

A PyTorch implementation of my Independent Study project.

This project uses the **CAMELYON16** dataset for lymph node metastasis detection in whole-slide histopathology images, which can be found [here](https://camelyon16.grand-challenge.org/Data/).
## Results

**Table: Precision (P), Recall (R), and F1-score (F1) for tumor and normal classes across RUMC and UMCU domains. Best results are in bold.**

### RUMC Domain

| Model | Tumor P | Tumor R | Tumor F1 | Normal P | Normal R | Normal F1 |
|-------|---------|---------|----------|----------|----------|-----------|
| ResNet50 | 0.950 | **0.830** | 0.890 | **0.990** | **1.000** | **0.995** |
| ConDo | **0.994** | 0.812 | **0.894** | 0.984 | 0.999 | 0.992 |

### UMCU Domain

| Model | Tumor P | Tumor R | Tumor F1 | Normal P | Normal R | Normal F1 |
|-------|---------|---------|----------|----------|----------|-----------|
| ResNet50 | 0.780 | 0.600 | 0.680 | 0.960 | 0.980 | 0.970 |
| ConDo | **0.820** | **0.642** | **0.814** | **0.968** | **0.987** | **0.978** |

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
