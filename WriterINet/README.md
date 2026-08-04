# WriterINet: Offline Text-Independent Writer Identification

This project presents an implementation of **WriterINet**, a deep learning framework for **offline text-independent writer identification** using handwritten document images. The implementation follows the proposed methodology from the research paper while leveraging transfer learning with pretrained CNN backbones and an Artificial Neural Network (ANN) classifier.

---

## Project Overview

The objective of this project is to identify the author of a handwritten document regardless of the written text.

The complete implementation was developed in **PyTorch** using a Kaggle notebook and includes:

- Dataset preparation
- XML annotation parsing
- Writer label encoding
- Custom PyTorch Dataset
- DenseNet201 feature extraction
- ResNet50 feature extraction
- Feature fusion
- ANN-based writer classification
- Model evaluation
- Classification report
- Confusion matrix
- Training curves
- Model checkpoint generation

---

## Dataset

**Dataset:** IAM Handwriting Database

The dataset contains handwritten text samples from multiple writers and is used for text-independent writer identification.

Dataset preprocessing includes:

- XML annotation parsing
- Writer ID extraction
- Label encoding
- Train/Test split
- Image preprocessing
- PyTorch DataLoader creation

---

## Model Architecture

The proposed pipeline consists of the following stages:

```
Handwritten Image
        │
        ▼
 Image Preprocessing
        │
        ▼
──────────────────────────────
│                            │
▼                            ▼
DenseNet201              ResNet50
Feature Extractor        Feature Extractor
│                            │
────────── Feature Fusion ──────────
                │
                ▼
      Artificial Neural Network
                │
                ▼
      Writer Identification
```

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PIL

---

## Repository Structure

```
WriterINet/
│
├── notebook/
│   └── WRITERINET_CODE.ipynb
│
├── docs/
│   ├── WRITERINET.pdf
│   └── WriterINet_Research_Paper.pdf
│
├── results/
│   ├── classification_report.png
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   └── training_loss.png
│
├── checkpoints/
│   └── writerinet_ann_model.pth (if available)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Experimental Results

The model was evaluated on the held-out test set.

| Metric | Value |
|---------|--------|
| Test Accuracy | **62%** |

Additional evaluation artifacts are available in the **results/** directory, including:

- Classification Report
- Confusion Matrix
- Training Accuracy Curve
- Training Loss Curve

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/<your-username>/SUMMER_PROJECT.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebook

```
WriterINet/notebook/WRITERINET_CODE.ipynb
```

4. Execute all notebook cells sequentially.

---

## Files Included

- Complete Kaggle implementation notebook
- Project report
- Research paper
- Evaluation results
- Model checkpoint (if available)

---

## Future Improvements

- Convert notebook into modular Python scripts
- Improve feature fusion strategy
- Hyperparameter optimization
- Experiment with Vision Transformers
- Compare additional CNN backbones

---

## Reference

**WriterINet: A Multi-Path Deep CNN for Offline Text-Independent Writer Identification**

This repository is intended for educational and research purposes and demonstrates an implementation inspired by the published methodology.
