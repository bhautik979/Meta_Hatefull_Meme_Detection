# 🧠 Meta Hateful Meme Detection

> **Multimodal AI system** that detects hate speech in memes by jointly reasoning over **both image and text** — because neither alone is enough.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange?logo=pytorch)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-green)](https://openai.com/research/clip)
[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/Dataset-Meta%20AI-blue)](https://ai.facebook.com/tools/hatefulmemes/)

---

## 📌 Table of Contents

1. [What & Why](#-what--why)
2. [Dataset](#-dataset)
3. [Architecture](#-architecture)
4. [Project Pipeline](#-project-pipeline-notebook-by-notebook)
5. [Results](#-results)
6. [Responsible AI Features](#-responsible-ai-features)
7. [Setup & Installation](#-setup--installation)
8. [Project Structure](#-project-structure)
9. [Citation](#-citation)

---

## 🎯 What & Why

### The Problem

A meme saying *"they should go back where they came from"* over a picture of a family is hateful.  
The **same sentence** overlaid on a moving-truck ad is perfectly benign.

This is why text-only or image-only classifiers fail — **context lives in the combination**.

```
Text alone  → "They should go home"  → Ambiguous
Image alone → [Family at border]      → Ambiguous
Together    → [Family at border] +
              "They should go home"   → HATEFUL ✓
```

### What This Project Does

A full end-to-end research pipeline that:
- Builds and evaluates **5 progressively stronger models** (from TF-IDF baselines to cross-attention fusion)
- Explains predictions with **GradCAM** (image) and **token attribution** (text)
- Generates **counterfactual examples** to probe model reasoning
- Provides a **Human-in-the-Loop (HITL)** review interface
- Packages everything into a **Gradio web demo** + **FastAPI inference endpoint**

---

## 📊 Dataset

**Meta AI Hateful Memes Challenge** — originally released for NeurIPS 2020.

| Split | Samples | Labels |
|-------|---------|--------|
| Train | ~8,500  | hateful / not-hateful |
| Dev   | ~500    | hateful / not-hateful |
| Test  | ~1,000  | unlabelled (held out) |

### What Makes It Hard

The dataset includes **confounders** — adversarially constructed pairs designed to break unimodal systems:

| Scenario | Image | Text | Label |
|----------|-------|------|-------|
| Benign text, hateful image | hateful image | benign caption | ❌ Not hateful |
| Hateful text, benign image | normal image | hateful caption | ✅ Hateful |
| Same image, different text | same image | text A → not hateful, text B → hateful | Different |

### Data Format

Each `.jsonl` file has one JSON object per line:

```json
{
  "id": 42953,
  "img": "img/42953.png",
  "label": 0,
  "text": "its their character not their color that matters"
}
```

`label: 0` = not hateful, `label: 1` = hateful

---

## 🏗️ Architecture

### Core Idea: Cross-Attention Fusion

Instead of simply concatenating image + text features, we use **bidirectional cross-attention** — the model explicitly learns *which part of the image* to pay attention to *given the text*, and vice versa.

```
                    ┌──────────────────────┐
  Meme Image ──────►│  CLIP Vision Encoder  │──► Image Embedding (512-d)
                    │     (ViT-B/32)        │         │
                    └──────────────────────┘         │
                                                      ▼
                                            ┌─────────────────────┐
                                            │  Cross-Attention     │
                                            │  Fusion Layer        │◄── Text Embedding (512-d)
                                            │  (bidirectional)     │
                                            └──────────┬──────────┘
                                                       │
                    ┌──────────────────────┐           │
  Meme Text ───────►│  CLIP Text Encoder   │───────────┘
                    │     (ViT-B/32)       │
                    └──────────────────────┘
                                                       │
                                            ┌──────────▼──────────┐
                                            │    MLP Classifier    │
                                            │  1024 → 256 → 128 → 2│
                                            └──────────┬──────────┘
                                                       │
                                             Not Hateful / Hateful
```

### Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Text Encoder | CLIP ViT-B/32 | Aligned embedding space with image encoder |
| Image Encoder | CLIP ViT-B/32 | Pretrained on 400M image-text pairs |
| Fusion | Cross-Attention | Forces inter-modal reasoning, not just feature stacking |
| Classifier | MLP (4 layers) | Lightweight, fast, interpretable |
| Encoder Weights | **Frozen** | Limited GPU budget; prevents catastrophic forgetting |
| Class Imbalance | Weighted Random Sampler + Focal Loss | Balanced batches + penalises easy negatives |

---

## 📓 Project Pipeline: Notebook by Notebook

The project is organized as a **sequential pipeline** of 16+ notebooks.

### 🔧 Phase 0 — Foundation (Notebooks 00–05)

| Notebook | Purpose |
|----------|---------|
| `00_setup.ipynb` | Environment check, GPU verification, dependency install |
| `01_problem-scope.ipynb` | Task definition, metrics, architecture decisions |
| `02_data-inspection.ipynb` | Load & visualise raw dataset samples, label distribution |
| `03_eda.ipynb` | Exploratory Data Analysis — text length, image stats, class balance |
| `04_ocr_pipeline.ipynb` | Extract & clean embedded text from memes using Tesseract OCR |
| `05_preprocessing.ipynb` | Build `MemeDataset` class, CLIP transforms, DataLoaders |

### 📈 Phase 1 — Modelling (Notebooks 06–09c)

| Notebook | Purpose |
|----------|---------|
| `06_baselines.ipynb` | 4 baseline models: TF-IDF+LR → RoBERTa → CLIP Image → CLIP Concat |
| `07-main-model.ipynb` | Cross-attention fusion model definition |
| `08-training.ipynb` | Full training loop: Focal Loss, mixed precision, early stopping |
| `09-auroc-optimization.ipynb` | Threshold tuning, AUROC maximisation |
| `09b-auroc-unfreeze-distill.ipynb` | Layer unfreezing + knowledge distillation experiments |
| `09c-vitl14-upgrade.ipynb` | Upgrade to ViT-L/14 backbone for performance boost |

### 🔍 Phase 2 — Analysis & Explainability (Notebooks 10–13)

| Notebook | Purpose |
|----------|---------|
| `10-evaluation.ipynb` | Full evaluation suite: AUROC, F1, confusion matrix, calibration |
| `11-explainability.ipynb` | GradCAM heatmaps + token-level text attribution |
| `12-counterfactuals.ipynb` | Minimal text/image edits that flip predictions |
| `13_error_analysis.ipynb` | Systematic failure analysis — grouped by error type |

### 🚀 Phase 3 — Deployment (Notebooks 14–16)

| Notebook | Purpose |
|----------|---------|
| `14_hitl.ipynb` | Human-in-the-Loop review queue with 3-tier routing |
| `15_packaging_api.ipynb` | FastAPI inference endpoint with request/response schema |
| `16_demo.ipynb` | Gradio 4-tab demo: Predict / Explain / Counterfactual / Flag |

---

## 📊 Results

### Baseline Ladder

Each model adds one capability. The table shows how multimodal fusion progressively improves performance:

| # | Model | What It Uses | Accuracy | AUROC | Macro F1 |
|---|-------|-------------|----------|-------|----------|
| 1 | TF-IDF + Logistic Regression | Text only | 56.0% | 0.594 | 0.545 |
| 2 | RoBERTa | Text only (deep) | 54.8% | 0.645 | 0.495 |
| 3 | CLIP Image + Linear | Image only | 56.4% | 0.647 | 0.549 |
| 4 | CLIP Concat Fusion | Image + Text (simple) | **63.4%** | **0.700** | **0.628** |
| 5 | **Cross-Attention Fusion** | Image + Text (ours) | — | **≥ 0.75 (target)** | **≥ 0.70 (target)** |

> **Key insight**: Simple fusion (model 4) already outperforms any unimodal model — confirming the multimodal hypothesis. Cross-attention (model 5) pushes further by learning *how* modalities relate.

### Primary Metric: AUROC

**Why AUROC?** It's threshold-independent — critical for content moderation where you tune the decision threshold per deployment context:
- **Strict mode** (safety-first): low false negatives → catch as much hate as possible
- **Lenient mode** (user experience): low false positives → avoid wrongly removing content

---

## 🛡️ Responsible AI Features

This project goes beyond accuracy and implements a full responsible AI stack:

| Feature | Implementation | Notebook |
|---------|---------------|----------|
| 🔍 **Explainability** | GradCAM image heatmaps + token attribution scores | 11 |
| 🔄 **Counterfactuals** | Minimal text/image edits that flip the label | 12 |
| ⚖️ **Fairness Audit** | Per-demographic-group performance breakdown | 10 |
| 📉 **Calibration** | Reliability diagram — confidence should match accuracy | 10 |
| 👤 **Human-in-the-Loop** | 3-tier routing: auto-approve / review / auto-remove | 14 |
| 💪 **Robustness** | OCR noise injection + paraphrase stress testing | 12 |
| 🚩 **User Flagging** | Demo UI allows users to flag wrong predictions | 16 |

### Human-in-the-Loop (HITL) Routing

```
    Model Confidence
           │
    ───────┼───────────────
    High   │  → Auto-approve (not hateful)
           │  → Auto-remove  (clearly hateful)
    Low    │  → Human review queue
    ───────┼───────────────
```

All HITL decisions are logged to a JSONL audit trail for accountability.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU recommended (project runs on Kaggle T4)
- [Kaggle account](https://www.kaggle.com/) for dataset access

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Meta-Hateful-Meme-Detection.git
cd Meta-Hateful-Meme-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>What's installed?</summary>

| Category | Packages |
|----------|---------|
| Deep Learning | `torch>=2.1`, `torchvision`, `torchaudio` |
| Vision-Language | `transformers>=4.36`, `accelerate`, `sentencepiece` |
| Data / ML | `numpy`, `pandas`, `scikit-learn`, `tqdm` |
| Image Processing | `Pillow` |
| Visualisation | `matplotlib`, `seaborn` |
| Config | `pyyaml` |
| Notebooks | `jupyter`, `ipykernel` |
| Phase 2 (optional) | `pytesseract`, `opencv-python` |

</details>

### 3. Get the Dataset

The dataset is from Meta AI's Hateful Memes Challenge. Place files so the structure looks like:

```
data/
├── img/          ← PNG meme images
├── train.jsonl   ← ~8,500 labelled training samples
├── dev.jsonl     ← ~500 labelled dev samples
└── test.jsonl    ← ~1,000 test samples
```

> **On Kaggle**: The notebooks auto-detect the dataset path — no manual setup needed.  
> **Locally**: Set the `META_HATEFUL_MEME_DATA_DIR` environment variable to your `data/` folder.

```bash
# Windows
set META_HATEFUL_MEME_DATA_DIR=C:\path\to\data

# macOS / Linux
export META_HATEFUL_MEME_DATA_DIR=/path/to/data
```

### 4. Run Notebooks in Order

```bash
jupyter notebook
```

Start from `00_setup.ipynb` and proceed sequentially. Each notebook is also **independently runnable** — it auto-detects the dataset path on startup.

---

## 📁 Project Structure

```
Meta-Hateful-Meme-Detection/
│
├── 📓 notebooks/
│   ├── 00_setup.ipynb                      # Environment & GPU check
│   ├── 01_problem-scope.ipynb              # Task definition & design
│   ├── 02_data-inspection.ipynb            # Raw data exploration
│   ├── 03_eda.ipynb                        # Exploratory Data Analysis
│   ├── 04_ocr_pipeline.ipynb              # OCR text extraction
│   ├── 05_preprocessing.ipynb             # Dataset class & transforms
│   │
│   ├── 06_baselines.ipynb                 # 4 baseline models
│   ├── 07-main-model.ipynb                # Cross-attention model
│   ├── 08-training.ipynb                  # Training loop
│   ├── 09-auroc-optimization.ipynb        # Threshold & AUROC tuning
│   ├── 09b-auroc-unfreeze-distill.ipynb   # Unfreezing + distillation
│   ├── 09c-vitl14-upgrade.ipynb           # ViT-L/14 upgrade
│   │
│   ├── 10-evaluation.ipynb                # Full evaluation suite
│   ├── 11-explainability.ipynb            # GradCAM + attribution
│   ├── 12-counterfactuals.ipynb           # Counterfactual generation
│   ├── 13_error_analysis.ipynb            # Failure case analysis
│   │
│   ├── 14_hitl.ipynb                      # Human-in-the-Loop
│   ├── 15_packaging_api.ipynb             # FastAPI endpoint
│   └── 16_demo.ipynb                      # Gradio demo app
│
├── 📂 data/
│   ├── img/                               # Meme PNG images
│   ├── train.jsonl                        # Training split
│   ├── dev.jsonl                          # Dev split
│   ├── test.jsonl                         # Test split
│   └── README.md                          # Meta AI dataset README
│
├── 📂 outputs/
│   ├── baselines/
│   │   ├── baseline_results.csv           # Baseline model metrics
│   │   └── baseline_comparison.png        # Comparison chart
│   └── evaluation/
│       └── notebooks_00_04_deep_guide.*   # Auto-generated docs
│
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                          # Git ignore rules
└── 📄 README.md                           # This file
```

---

## 🔬 Research Papers

This project draws from the following papers (included as PDFs):

| File | Paper |
|------|-------|
| `3462244.3479949.pdf` | The Hateful Memes Challenge (Kiela et al., 2020) — original dataset paper |
| `1-s2.0-S0952197623011752-main.pdf` | Multimodal hate speech detection survey |
| `1-s2.0-S2666827025000301-main.pdf` | Recent advances in multimodal content moderation |

---

## 📐 Metrics Reference

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **AUROC** | Ranking quality across all thresholds | Area under ROC curve |
| **Accuracy** | Overall % correct | (TP + TN) / Total |
| **Macro F1** | Average F1 for both classes | mean(F1_hateful, F1_benign) |
| **F1 (hateful)** | Balance of precision & recall on hate class | 2 × P × R / (P + R) |

**Target thresholds** (from `01_problem-scope.ipynb`):

| Metric | Target |
|--------|--------|
| AUROC | ≥ 0.75 |
| Macro F1 | ≥ 0.70 |
| F1 (hateful) | ≥ 0.65 |
| Accuracy | ≥ 0.72 |

---

## 📜 Citation

If you use this work, please cite the original dataset:

```bibtex
@inproceedings{Kiela2020TheHM,
  title     = {The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author    = {Douwe Kiela and Hamed Firooz and Aravind Mohan and Vedanuj Goswami
               and Amanpreet Singh and Pratik Ringshia and Davide Testuggine},
  booktitle = {NeurIPS},
  year      = {2020}
}
```

---

## ⚠️ Disclaimer

This project is a **research prototype** for studying AI content moderation. The dataset contains examples of hate speech and offensive content collected for academic purposes. All annotations were provided by third-party annotators and do not reflect the views of Meta/Facebook or the authors of this project.

---

<div align="center">

**Built for the Meta AI Hateful Memes Challenge**  
*Multimodal AI | Computer Vision | NLP | Responsible AI*

</div>
