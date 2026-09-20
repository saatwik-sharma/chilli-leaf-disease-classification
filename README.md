# Chilli Leaf Disease Classification Using CNN-ViT Models

A research implementation for six-class chilli leaf disease classification using ImageNet-pretrained CNN and Vision Transformer architectures, with duplicate-aware dataset auditing, controlled model comparison, and Grad-CAM-based visual interpretation.

---

## 1. Project Overview

Accurate identification of chilli leaf diseases from leaf images can support early disease recognition and reduce dependence on manual visual inspection.

This project investigates deep learning approaches for classifying chilli leaf images into six disease categories. The study follows a controlled experimental pipeline:

**Dataset Analysis → Duplicate/Leakage Audit → Duplicate-Aware Split → Model Training → Evaluation → XAI**

Three main model configurations were evaluated:

1. ResNet-50 baseline
2. Frozen CNN + ViT hybrid
3. Fully fine-tuned CNN + ViT hybrid

The final objective was to determine how CNN features, Vision Transformer features, and end-to-end fine-tuning affect classification performance under a duplicate-aware evaluation protocol.

---

## 2. Dataset

The dataset contains **8,817 chilli leaf images** belonging to six classes.

| Class                |    Images |
| -------------------- | --------: |
| Bacterial Spot       |     1,629 |
| Cercospora Leaf Spot |     1,898 |
| Curl Virus           |     1,590 |
| Healthy Leaf         |     1,647 |
| Nutrition Deficiency |     1,207 |
| Powdery Mildew       |       846 |
| **Total**            | **8,817** |

### Dataset location

The original dataset is intentionally **not included in this repository** because of repository size and reproducibility considerations.

The dataset folder and ZIP file are excluded using `.gitignore`.

---

## 3. Dataset Quality and Duplicate Audit

Before model training, the complete dataset was audited for visually similar and potentially duplicated images.

A **difference hash (dHash)** based similarity analysis was performed across all 8,817 images.

### Duplicate analysis

* Total images analyzed: **8,817**
* dHash groups identified: **6,397**
* Multi-image groups: **1,195**
* Largest similarity group: **146 images**

The duplicate analysis was used to reduce the possibility of visually similar images appearing across training, validation, and test sets.

This is important because random image-level splitting can produce overly optimistic results when near-duplicate images are distributed across different subsets.

---

## 4. Final Dataset Split

A duplicate-aware split was created and locked before the final experiments.

| Split      |    Images | Percentage |
| ---------- | --------: | ---------: |
| Training   |     6,172 |       ~70% |
| Validation |     1,323 |       ~15% |
| Test       |     1,322 |       ~15% |
| **Total**  | **8,817** |   **100%** |

The **test set was kept untouched during training and model selection**.

The final split is stored in:

```text
03_dataset_splitting/dataset_splits_final.json
```

---

## 5. Preprocessing

All models use ImageNet-compatible preprocessing.

### Training

The training pipeline applies:

* Random resized crop to 224 × 224
* Random horizontal flip
* Random rotation up to ±15°
* Mild ColorJitter
* ImageNet normalization

### Validation and Test

Validation and test images use:

* Resize to 224 × 224
* ImageNet normalization

The same locked split and evaluation protocol are used for the final model comparison.

---

## 6. Experimental Pipeline

The complete research pipeline is:

```text
Raw Dataset
     ↓
Dataset Inspection
     ↓
Class Distribution Analysis
     ↓
dHash Duplicate / Similarity Audit
     ↓
Duplicate-Aware Dataset Split
     ↓
Training / Validation / Test
     ↓
ResNet-50 Baseline
     ↓
Frozen CNN + ViT Hybrid
     ↓
Fully Fine-Tuned CNN + ViT Hybrid
     ↓
Final Test Evaluation
     ↓
Confusion Matrix + Classification Metrics
     ↓
Grad-CAM Visual Interpretation
```

---

# 7. Model Architectures

## 7.1 ResNet-50 Baseline

A standard ImageNet-pretrained **ResNet-50** was used as the CNN baseline.

The final classification layer was replaced with a six-class classifier corresponding to the chilli disease categories.

The purpose of this experiment was to establish a strong CNN baseline before introducing multimodal feature fusion.

---

## 7.2 Frozen CNN + ViT Hybrid

The second experiment combines:

* ImageNet-pretrained ResNet-50
* ImageNet-pretrained ViT-B/16

The feature representations from both branches are concatenated and passed to a trainable classification layer.

In this experiment, both backbone networks remain frozen and only the classifier is trained.

### Parameter configuration

```text
CNN parameters:          23,508,032
ViT parameters:          85,798,656
Classifier parameters:       16,902
Total parameters:       109,323,590
Trainable parameters:       16,902
```

This experiment tests whether simply combining pretrained CNN and transformer representations provides an advantage without fine-tuning the feature extractors.

---

## 7.3 Fully Fine-Tuned CNN + ViT Hybrid

The final hybrid model uses the same ResNet-50 + ViT-B/16 architecture, but both backbone networks are fully trainable.

The architecture consists of:

```text
Input Image
     │
     ├───────────────┐
     ↓               ↓
 ResNet-50         ViT-B/16
     │               │
 CNN Features    ViT Features
     │               │
     └───────┬───────┘
             ↓
       Feature Fusion
             ↓
        Classification
             ↓
        6 Classes
```

### Training configuration

```text
Batch size:              2
Gradient accumulation:   8
Effective batch size:    16
Maximum epochs:          10
Early stopping patience: 3
Backbone learning rate:  1e-5
Classifier learning rate: 1e-4
Weight decay:             1e-2
Dropout:                  0.3
Optimizer:                AdamW
Scheduler:                CosineAnnealingLR
Loss:                     CrossEntropyLoss
AMP:                      Disabled
```

The use of separate learning rates allows the pretrained backbones to be fine-tuned conservatively while allowing the newly initialized classifier to learn more rapidly.

---

# 8. Experimental Chronology

Initial experiments were performed before the final duplicate-aware dataset audit.

The preliminary results were retained separately because the later experiments use the stricter duplicate-aware split.

### Preliminary results

| Model          | Test Accuracy |
| -------------- | ------------: |
| ResNet-50      |        98.26% |
| ViT-B/16       |        98.64% |
| Initial Hybrid |        97.88% |

These results are treated as **preliminary** rather than as the final controlled comparison.

After identifying visually similar images and creating the final duplicate-aware split, the main experiments were repeated.

---

# 9. Final Results

The final controlled comparison is based on the locked duplicate-aware test set containing **1,322 images**.

| Model                | Test Accuracy |   Macro F1 |
| -------------------- | ------------: | ---------: |
| ResNet-50            |    **97.81%** | **97.88%** |
| Frozen CNN + ViT     |    **93.95%** | **94.29%** |
| Fine-Tuned CNN + ViT |    **98.26%** | **98.36%** |

The fully fine-tuned CNN + ViT hybrid achieved the highest test accuracy among the three final experiments.

Compared with the ResNet-50 baseline, the fine-tuned hybrid improved test accuracy by:

**0.45 percentage points**

Compared with the frozen CNN + ViT hybrid, fine-tuning improved test accuracy by:

**4.31 percentage points**

The results indicate that simply concatenating frozen representations was not sufficient for this dataset, while end-to-end adaptation of both branches produced stronger performance.

---

# 10. ResNet-50 Final Results

### Test performance

```text
Test Accuracy:       97.81%
Test Loss:            0.0653

Macro Precision:     97.98%
Macro Recall:        97.81%
Macro F1:            97.88%

Weighted Precision:  97.83%
Weighted Recall:     97.81%
Weighted F1:         97.80%
```

### Per-class performance

| Class                | Precision |  Recall |      F1 |
| -------------------- | --------: | ------: | ------: |
| Bacterial Spot       |    97.21% |  99.59% |  98.39% |
| Cercospora Leaf Spot |    99.65% |  98.60% |  99.12% |
| Curl Virus           |    95.40% |  96.20% |  95.80% |
| Healthy Leaf         |    96.80% |  97.98% |  97.38% |
| Nutrition Deficiency |    98.84% |  94.48% |  96.61% |
| Powdery Mildew       |   100.00% | 100.00% | 100.00% |

---

# 11. Frozen CNN + ViT Results

### Test performance

```text
Test Accuracy:       93.95%
Test Loss:             0.2533

Macro Precision:      94.58%
Macro Recall:         94.18%
Macro F1:             94.29%

Weighted Precision:   94.27%
Weighted Recall:      93.95%
Weighted F1:          94.01%
```

This experiment demonstrates that feature fusion alone does not necessarily improve performance when both pretrained feature extractors remain frozen.

---

# 12. Fine-Tuned CNN + ViT Results

### Test performance

```text
Test Accuracy:       98.26%
Test Loss:             0.0728

Macro Precision:      98.41%
Macro Recall:         98.33%
Macro F1:             98.36%

Weighted Precision:   98.27%
Weighted Recall:      98.26%
Weighted F1:          98.26%
```

### Per-class performance

| Class                | Precision |  Recall |      F1 |
| -------------------- | --------: | ------: | ------: |
| Bacterial Spot       |    96.39% |  97.96% |  97.17% |
| Cercospora Leaf Spot |    98.27% |  99.65% |  98.95% |
| Curl Virus           |    98.27% |  95.78% |  97.01% |
| Healthy Leaf         |    99.19% |  98.79% |  98.99% |
| Nutrition Deficiency |    98.33% |  97.79% |  98.06% |
| Powdery Mildew       |   100.00% | 100.00% | 100.00% |

The model correctly classified:

```text
1293 / 1322 test images
```

with **29 misclassified images**.

---

# 13. Confusion Analysis

The final fine-tuned hybrid confusion matrix is:

```text
[[240, 0, 3, 0, 2, 0],
 [  0,284, 0, 1, 0, 0],
 [  6, 2,227, 1, 1, 0],
 [  1, 1, 1,244, 0, 0],
 [  2, 2, 0, 0,177, 0],
 [  0, 0, 0, 0, 0,127]]
```

The main remaining errors occur between visually similar disease categories, particularly:

* Curl Virus and Bacterial Spot
* Nutrition Deficiency and Curl Virus
* Healthy Leaf and several disease categories

This indicates that some errors are associated with visually overlapping symptoms rather than broad class confusion.

---

# 14. Explainable AI

Grad-CAM was applied to the **final convolutional layer of the ResNet-50 branch** of the fully fine-tuned CNN + ViT hybrid.

The target layer was:

```text
model.cnn.layer4[-1]
```

Three correctly classified examples were selected from each of the six classes, producing:

```text
18 Grad-CAM visualizations
```

The XAI outputs are stored in:

```text
07_xai/xai_gradcam/
```

Grad-CAM is used here to provide a visual interpretation of the CNN branch's learned spatial attention.

It should not be interpreted as a complete explanation of the entire hybrid model decision because the final classifier also receives features from the ViT branch.

---

# 15. Repository Structure

```text
chilli-leaf-disease-classification/
│
├── README.md
│
├── 01_dataset_analysis/
│   └── inspect_dataset.py
│
├── 02_duplicate_leakage_analysis/
│   ├── check_leakage.py
│   ├── dhash_suspicious_groups.jpg
│   ├── inspect_dhash_duplicates.py
│   └── test_matches.py
│
├── 03_dataset_splitting/
│   ├── create_clean_split.py
│   ├── create_final_split.py
│   ├── dataset.py
│   ├── dataset_splits_final.json
│   ├── verify_clean_split.py
│   └── history/
│       ├── dataset_splits.json
│       └── dataset_splits_clean.json
│
├── 04_models/
│   ├── model.py
│   ├── model_hybrid.py
│   ├── model_hybrid_finetuned.py
│   └── model_vit.py
│
├── 05_training/
│   ├── train.py
│   ├── train_hybrid.py
│   ├── train_hybrid_finetuned.py
│   └── train_vit.py
│
├── 06_results/
│   ├── preliminary/
│   ├── resnet50_baseline/
│   ├── hybrid_cnn_vit/
│   └── hybrid_cnn_vit_finetuned/
│
├── 07_xai/
│   ├── gradcam_xai.py
│   └── xai_gradcam/
│
└── 08_figures/
    ├── final/
    └── preliminary/
```

---

# 16. Results Directory

The repository preserves both preliminary and final experimental results.

### Preliminary

```text
06_results/preliminary/
├── test_results.json
├── test_results_hybrid.json
└── test_results_vit.json
```

### ResNet-50

```text
06_results/resnet50_baseline/
├── classification_report.txt
├── config.json
├── confusion_matrix.png
├── test_results.json
├── training_curves.png
├── training_history.csv
└── training_history.json
```

### Frozen Hybrid

```text
06_results/hybrid_cnn_vit/
├── classification_report.txt
├── config.json
├── confusion_matrix.png
├── test_results.json
├── training_curves.png
└── training_history.json
```

### Fine-Tuned Hybrid

```text
06_results/hybrid_cnn_vit_finetuned/
├── classification_report.txt
├── config.json
├── confusion_matrix.png
├── test_results.json
└── training_history.json
```

---

# 17. Reproducibility

The repository contains:

* Dataset inspection code
* Duplicate/similarity analysis
* Final dataset split
* Dataset loader
* Model definitions
* Training scripts
* Evaluation results
* Confusion matrices
* Training histories
* Grad-CAM implementation
* XAI outputs

The dataset itself and trained model checkpoints are excluded from GitHub because of their size.

To reproduce the experiments:

1. Obtain the original dataset.
2. Place it in the expected dataset directory.
3. Install the required Python dependencies.
4. Run the dataset analysis and splitting scripts.
5. Use the provided model and training scripts.
6. Evaluate the trained models using the provided evaluation configuration.

---

# 18. Important Reproducibility Notes

The final evaluation uses the locked duplicate-aware split:

```text
Train: 6,172
Validation: 1,323
Test: 1,322
```

The test set must not be used during training, hyperparameter selection, or early stopping.

Model checkpoints are intentionally excluded from version control through `.gitignore`.

The dataset is also excluded from version control.

---

# 19. Research Contribution

The main contribution of this implementation is a controlled investigation of CNN and Vision Transformer representations for chilli leaf disease classification while explicitly addressing dataset similarity and potential split contamination.

The study includes:

* Six-class chilli disease classification
* Dataset-level duplicate/similarity auditing
* Duplicate-aware train/validation/test splitting
* A strong CNN baseline
* CNN + Vision Transformer feature fusion
* Comparison of frozen and fully fine-tuned hybrid models
* Class-wise evaluation
* Confusion matrix analysis
* Grad-CAM-based visual interpretation

The results show that the final fine-tuned CNN + ViT configuration achieved **98.26% test accuracy** on the locked duplicate-aware test set.

This result is specific to the dataset and experimental protocol used in this study and should not be interpreted as universal superiority over other architectures or datasets.

---

# 20. Limitations

Several limitations should be considered:

* The dataset is image-based and may not fully represent field conditions.
* Images may differ from real-world photographs in lighting, background, camera quality, and disease severity.
* The study uses a single chilli leaf dataset.
* The external generalization of the trained model has not yet been established.
* Grad-CAM explains the CNN branch rather than the complete CNN + ViT decision process.
* The final improvement over the ResNet-50 baseline is relatively small at 0.45 percentage points.

Future work can evaluate the model on independently collected field images and additional chilli disease datasets.

---

# 21. Technologies Used

* Python
* PyTorch
* torchvision
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* PIL
* Grad-CAM
* Git / GitHub
* NVIDIA CUDA

---

# 22. Authors

**Saatwik Sharma**
B.Tech Artificial Intelligence & Machine Learning
Symbiosis Institute of Technology, Pune

---

## License

This repository is intended for academic and research purposes.
