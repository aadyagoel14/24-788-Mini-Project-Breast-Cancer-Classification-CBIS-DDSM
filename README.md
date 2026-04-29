# Breast Cancer Classification on CBIS-DDSM
### 24-788 Introduction to Deep Learning — Mini-Project (Spring 2026)

**Team Members:** Aadya Goel (aadyag) · Amisha Dhawan (adhawan)

---

## Overview

This project compares three deep learning architectures for binary breast cancer classification (benign vs. malignant) on the CBIS-DDSM mammography dataset:

| Model | Type | Description |
|---|---|---|
| **ANN** | Baseline | Fully-connected network; no spatial inductive bias |
| **ResNet50** | Variant 1 | Transfer learning from ImageNet + fine-tuning |
| **EfficientNetB0** | Variant 2 | Transfer learning from ImageNet + fine-tuning |

---

## Dataset

**CBIS-DDSM** (Curated Breast Imaging Subset of DDSM)  
Source: [Kaggle — awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

The dataset contains full mammogram images from mass and calcification cases, labeled as `BENIGN` or `MALIGNANT`. `BENIGN_WITHOUT_CALLBACK` cases are merged into the `BENIGN` class. Images are matched to labels via `dicom_info.csv` and filtered to full mammogram series only.

**Split used:**
- Train: 90% of training CSV rows with matched images
- Validation: 10% of training CSV rows (stratified)
- Test: held-out test CSV rows

---

## Repository Structure

```
├── mini-project-intro-to-deep-learning.ipynb   # Main training notebook (Kaggle)
├── reproduce_results.ipynb                      # Loads saved checkpoints, regenerates figures
└── README.md
```

> **Model checkpoints** are hosted on Google Drive (too large for GitHub):  
> 📁 [DL_MiniProject — Google Drive](https://drive.google.com/drive/folders/1-GdCJcWcW6YGWTVQw91ehiA8rHLt9FiZ?usp=sharing)

---

## Environment Setup

The main notebook runs on **Kaggle** (GPU accelerator recommended). All dependencies are pre-installed in the Kaggle environment. No additional installation is required.

For the reproduction notebook on **Google Colab**, install dependencies with:

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
```

---

## Data Setup

1. Go to [Kaggle dataset page](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
2. Click **Download** (requires free Kaggle account)
3. The dataset will be automatically mounted at the following path when added to a Kaggle notebook:
   ```
   /kaggle/input/cbis-ddsm-breast-cancer-image-dataset/
       csv/
           calc_case_description_train_set.csv
           calc_case_description_test_set.csv
           mass_case_description_train_set.csv
           mass_case_description_test_set.csv
           dicom_info.csv
       jpeg/
           <UID folders>/
   ```
4. Set `DATA_ROOT` in the notebook config cell to match this path.

---

## Reproducing Key Results

You do **not** need to retrain from scratch. Follow these steps:

### Step 1 — Download model checkpoints

The three `.keras` checkpoint files are hosted on Google Drive due to their size:

📁 **[DL_MiniProject — Google Drive](https://drive.google.com/drive/folders/1-GdCJcWcW6YGWTVQw91ehiA8rHLt9FiZ?usp=sharing)**

Download all three files from the `DL_MiniProject/` folder:
- `ANN_Baseline_best.keras`
- `ResNet50_Transfer_ft_best.keras`
- `EfficientNetB0_Transfer_ft_best.keras`

### Step 2 — Open `reproduce_results.ipynb` in Google Colab

### Step 3 — Upload checkpoints when prompted

The notebook will display a file upload widget. Select all three `.keras` files.

### Step 4 — Run all cells

The notebook will:
1. Rebuild the test dataset from the CBIS-DDSM CSVs
2. Load each saved model
3. Evaluate on the test set (accuracy, AUC, precision, recall, F1)
4. Regenerate all figures: confusion matrices, ROC curves, metric comparison bar chart

> **Note:** You will need to upload your Kaggle API token (`kaggle.json`) or manually provide the CSV files when prompted in the reproduction notebook.

---

## Training From Scratch

To retrain all models, open `mini-project-intro-to-deep-learning.ipynb` on Kaggle and run all cells in order. Training time per model on a Kaggle GPU (T4):

| Model | Approx. Training Time |
|---|---|
| ANN Baseline | ~10–15 min |
| ResNet50 (head only + fine-tune) | ~25–35 min |
| EfficientNetB0 (head only + fine-tune) | ~25–35 min |

Key hyperparameters:

| Setting | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Max epochs | 30 (EarlyStopping, patience=5) |
| Optimizer | Adam |
| ANN learning rate | 1e-5 |
| Transfer learning LR (head) | 1e-4 |
| Fine-tuning LR | 1e-5 |
| Loss | Binary cross-entropy |

---

## Results Summary

| Model | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ANN Baseline | 0.576 | 0.634 | 0.486 | 0.517 | 0.501 |
| ResNet50 | **0.673** | **0.710** | **0.587** | **0.695** | **0.637** |
| EfficientNetB0 | 0.637 | 0.689 | 0.553 | 0.632 | 0.590 |

*All metrics computed on the held-out test set (422 samples: 248 benign, 174 malignant). Precision, recall, and F1 are reported for the malignant class.*

---

## AI Tool Usage

GitHub Copilot and Claude (Anthropic) were used to assist with debugging data pipeline issues and README formatting. **All model design decisions, experimental choices, and written analysis are our own.**

---

## Citation

```
Lee, R.S., Gimenez, F., Hoogi, A., Miyake, K.K., Gorovoy, M., & Rubin, D.L. (2017).
A curated mammography data set for use in computer-aided detection and diagnosis research.
Scientific Data, 4, 170177.
```
