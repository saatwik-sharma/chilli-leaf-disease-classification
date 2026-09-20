# Chilli Leaf Disease Classification Using CNN-ViT Models



A deep learning study for six-class chilli leaf disease classification using ImageNet-pretrained CNN and Vision Transformer architectures, with duplicate-aware dataset auditing, controlled model comparison, and Grad-CAM-based visual interpretation.



---



## 1. Project Overview



Accurate identification of chilli leaf diseases from leaf images can support early disease recognition and reduce dependence on manual visual inspection.



This project investigates deep learning approaches for classifying chilli leaf images into six categories. The study follows a controlled experimental pipeline:



**Dataset Analysis â†’ Duplicate/Leakage Audit â†’ Duplicate-Aware Split â†’ Model Training â†’ Evaluation â†’ XAI**



Three main model configurations were evaluated:



1\. **ResNet-50** â€” CNN baseline

2\. **Frozen ResNet-50 + ViT-B/16** â€” feature-level CNN-ViT fusion with frozen backbones

3\. **Fine-tuned ResNet-50 + ViT-B/16** â€” end-to-end fine-tuned CNN-ViT hybrid



The final experiments were performed using a duplicate-aware 70/15/15 train-validation-test split.



---



## 2. Dataset



The dataset contains **8,817 images** belonging to six chilli leaf categories.



| Class                |    Images |

| -------------------- | --------: |

| Bacterial Spot       |     1,629 |

| Cercospora Leaf Spot |     1,898 |

| Curl Virus           |     1,590 |

| Healthy Leaf         |     1,647 |

| Nutrition Deficiency |     1,207 |

| Powdery Mildew       |       846 |

| **Total**            | **8,817** |



The dataset itself is **not included in this repository** because of its size and distribution considerations.



Place the dataset directory locally before running the pipeline.



Expected dataset location:



```text

Chilli Leaf Disease Image Dataset for Classificati/

```



---



## 3. Dataset Quality Audit



Before final model training, the dataset was examined for visually similar and near-duplicate images.



Perceptual hashing using **dHash** was used to identify groups of highly similar images.



The analysis identified:



* **8,817** total images

* **6,397** perceptual-hash groups

* **1,195** groups containing multiple images

* Largest detected group: **146 images**



These findings motivated a duplicate-aware splitting strategy to reduce the possibility of highly similar images being distributed across training and evaluation subsets.



The duplicate analysis does not imply that every detected similar image represents data leakage; the purpose was to identify potentially related samples before constructing the final evaluation split.



---



## 4. Final Dataset Split



The final locked split contains:



| Split      |    Images | Percentage |

| ---------- | --------: | ---------: |

| Training   |     6,172 |        70% |

| Validation |     1,323 |        15% |

| Test       |     1,322 |        15% |

| **Total**  | **8,817** |   **100%** |



The final split is stored in:



```text

03_dataset_splitting/dataset_splits_final.json

```



The test set was kept separate and was not used for model training or model-selection decisions.



Earlier intermediate split files are preserved under:



```text

03_dataset_splitting/history/

```



This maintains the chronology of the dataset preparation process.



---



## 5. Image Preprocessing



All models use the same basic preprocessing pipeline.



### Training



* RandomResizedCrop: 224 Ã— 224

* Random horizontal flip

* Random rotation: Â±15Â°

* Color jitter

* ImageNet normalization



### Validation and Testing



* Resize: 224 Ã— 224

* ImageNet normalization



The same validation and test preprocessing was maintained across experiments for controlled comparison.



---



## 6. Model Architectures



### 6.1 ResNet-50 Baseline



A standard ImageNet-pretrained **ResNet-50** was used as the CNN baseline.



The original classification head was replaced with a six-class output layer.



This establishes a strong convolutional baseline before introducing CNNâ€“Transformer fusion.



---



### 6.2 Frozen CNN-ViT Hybrid



The second experiment combines:



* ResNet-50

* ViT-B/16



Both backbones use ImageNet-pretrained weights and remain frozen during training.



Feature representations from both branches are concatenated and passed to a trainable classification layer.



The experiment evaluates whether simple feature-level fusion provides an advantage without fine-tuning the pretrained backbones.



---



### 6.3 Fine-Tuned CNN-ViT Hybrid



The final hybrid experiment uses the same ResNet-50 + ViT-B/16 architecture, but both branches are fine-tuned.



Training configuration:



```text

Batch size:          2

Gradient accumulation: 8

Effective batch size: 16

Backbone learning rate: 1e-5

Classifier learning rate: 1e-4

Weight decay:        1e-2

Dropout:             0.3

Maximum epochs:      10

Early stopping patience: 3

Optimizer:           AdamW

Scheduler:           CosineAnnealingLR

AMP:                 Disabled

```



Separate learning rates were used so that the pretrained backbones could be updated conservatively while allowing the newly initialized classifier to learn faster.



---



## 7. Experimental Chronology



The project initially included preliminary experiments on an earlier dataset split.



The workflow was subsequently refined after the duplicate-aware dataset audit.



Therefore, the repository preserves both stages:



```text

Initial experiments

&#x20;       â†“

Dataset duplicate/similarity audit

&#x20;       â†“

Duplicate-aware final split

&#x20;       â†“

Final model retraining

&#x20;       â†“

Final evaluation

&#x20;       â†“

Grad-CAM analysis

```



The preliminary results are retained for transparency and reproducibility but are not treated as the final evaluation results.



---



## 8. Final Results



The final models were evaluated on the locked **1,322-image test set**.



| Model              | Test Accuracy |   Macro F1 |

| ------------------ | ------------: | ---------: |

| ResNet-50          |    **97.81%** | **97.88%** |

| Frozen CNN-ViT     |    **93.95%** | **94.29%** |

| Fine-Tuned CNN-ViT |    **98.26%** | **98.36%** |



### Fine-Tuned CNN-ViT



Final test performance:



```text

Test Accuracy:       98.26%

Test Loss:            0.0728

Macro Precision:     98.41%

Macro Recall:        98.33%

Macro F1:            98.36%

Weighted Precision:  98.27%

Weighted Recall:     98.26%

Weighted F1:         98.26%

```



The fine-tuned hybrid achieved **98.26% accuracy**, corresponding to **1,293 correct predictions out of 1,322 test images**.



Compared with the ResNet-50 baseline, this represents an absolute improvement of **0.45 percentage points** on the final test split.



The frozen CNN-ViT configuration performed lower than both the ResNet-50 baseline and the fine-tuned hybrid, indicating that feature fusion without adapting the pretrained backbones was not sufficient to improve performance in this experimental setting.



These results describe this dataset and experimental configuration and should not be interpreted as evidence of universal superiority of one architecture over another.



---



## 9. Fine-Tuned Hybrid Per-Class Results



| Class                | Precision |  Recall |      F1 |

| -------------------- | --------: | ------: | ------: |

| Bacterial Spot       |    96.39% |  97.96% |  97.17% |

| Cercospora Leaf Spot |    98.27% |  99.65% |  98.95% |

| Curl Virus           |    98.27% |  95.78% |  97.01% |

| Healthy Leaf         |    99.19% |  98.79% |  98.99% |

| Nutrition Deficiency |    98.33% |  97.79% |  98.06% |

| Powdery Mildew       |   100.00% | 100.00% | 100.00% |



The model produced 29 misclassifications out of 1,322 test samples.



---



## 10. Explainable AI



Grad-CAM was applied to the final convolutional layer of the **ResNet-50 branch** of the fine-tuned CNN-ViT hybrid.



Target layer:



```text

model.cnn.layer4[-1]

```



The XAI analysis provides visual explanations of regions contributing to the CNN branch's prediction.



The repository contains multiple correctly classified examples across all six classes:



```text

07_xai/xai_gradcam/

```



A selected representative visualization can be used for the research paper.



Importantly, the Grad-CAM analysis is specifically applied to the CNN branch and should not be interpreted as a complete explanation of the entire hybrid model's decision.



---



## 11. Repository Structure



```text

chilli-leaf-disease-classification/

â”‚

â”œâ”€â”€ .gitignore

â”‚

â”œâ”€â”€ 01_dataset_analysis/

â”‚   â””â”€â”€ inspect_dataset.py

â”‚

â”œâ”€â”€ 02_duplicate_leakage_analysis/

â”‚   â”œâ”€â”€ check_leakage.py

â”‚   â”œâ”€â”€ dhash_suspicious_groups.jpg

â”‚   â”œâ”€â”€ inspect_dhash_duplicates.py

â”‚   â””â”€â”€ test_matches.py

â”‚

â”œâ”€â”€ 03_dataset_splitting/

â”‚   â”œâ”€â”€ create_clean_split.py

â”‚   â”œâ”€â”€ create_final_split.py

â”‚   â”œâ”€â”€ dataset.py

â”‚   â”œâ”€â”€ dataset_splits_final.json

â”‚   â”œâ”€â”€ verify_clean_split.py

â”‚   â””â”€â”€ history/

â”‚       â”œâ”€â”€ dataset_splits.json

â”‚       â””â”€â”€ dataset_splits_clean.json

â”‚

â”œâ”€â”€ 04_models/

â”‚   â”œâ”€â”€ model.py

â”‚   â”œâ”€â”€ model_hybrid.py

â”‚   â”œâ”€â”€ model_hybrid_finetuned.py

â”‚   â””â”€â”€ model_vit.py

â”‚

â”œâ”€â”€ 05_training/

â”‚   â”œâ”€â”€ train.py

â”‚   â”œâ”€â”€ train_hybrid.py

â”‚   â”œâ”€â”€ train_hybrid_finetuned.py

â”‚   â””â”€â”€ train_vit.py

â”‚

â”œâ”€â”€ 06_results/

â”‚   â”œâ”€â”€ preliminary/

â”‚   â”œâ”€â”€ resnet50_baseline/

â”‚   â”œâ”€â”€ hybrid_cnn_vit/

â”‚   â””â”€â”€ hybrid_cnn_vit_finetuned/

â”‚

â”œâ”€â”€ 07_xai/

â”‚   â”œâ”€â”€ gradcam_xai.py

â”‚   â””â”€â”€ xai_gradcam/

â”‚

â””â”€â”€ 08_figures/

&#x20;   â”œâ”€â”€ preliminary/

&#x20;   â””â”€â”€ final/

```



---



## 12. Reproducibility



### Environment



The experiments were developed using:



* Python

* PyTorch

* torchvision

* scikit-learn

* NumPy

* Pandas

* Matplotlib

* OpenCV/PIL-based image processing

* Grad-CAM



GPU training was performed using an NVIDIA RTX 3050 Laptop GPU with CUDA support.



### Dataset Setup



Place the dataset in the project root:



```text

Chilli Leaf Disease Image Dataset for Classificati/

```



### Pipeline



The project follows the numbered directory order:



```text

01_dataset_analysis

&#x20;       â†“

02_duplicate_leakage_analysis

&#x20;       â†“

03_dataset_splitting

&#x20;       â†“

04_models

&#x20;       â†“

05_training

&#x20;       â†“

06_results

&#x20;       â†“

07_xai

&#x20;       â†“

08_figures

```



The final split file should be generated/verified before final model evaluation.



---



## 13. Results and Checkpoints



Model checkpoints are intentionally excluded from GitHub because of their large file sizes.



The repository therefore contains:



* model definitions

* training scripts

* evaluation results

* dataset split information

* figures

* XAI outputs



but does not contain the trained `.pth` checkpoint files or the original dataset.



---



## 14. Research Contribution



The main methodological aspects of this study are:



1\. Evaluation of a CNN baseline for six-class chilli leaf disease classification.

2\. Investigation of CNN-ViT feature fusion.

3\. Comparison between frozen and fully fine-tuned hybrid architectures.

4\. Duplicate/similarity-aware dataset auditing before final evaluation.

5\. Controlled evaluation using a locked 70/15/15 split.

6\. Grad-CAM-based visual interpretation of the CNN branch.



The study emphasizes a reproducible experimental workflow in which dataset quality assessment precedes final model comparison.



---



## 15. Authors



**Saatwik Sharma**

B.Tech Artificial Intelligence \& Machine Learning

Symbiosis Institute of Technology, Pune



**Research work:** Chilli Leaf Disease Classification using CNN and Vision Transformer architectures.





