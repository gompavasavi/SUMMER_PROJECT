# evaluate.py

import numpy as np
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from config import DEVICE


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_loader
):

    model.eval()

    all_predictions = []

    all_true_labels = []

    correct = 0

    total = 0

    with torch.no_grad():

        for features, labels in tqdm(
            test_loader,
            desc="Evaluating"
        ):

            features = features.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                features
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            all_predictions.extend(
                predicted
                .cpu()
                .numpy()
            )

            all_true_labels.extend(
                labels
                .cpu()
                .numpy()
            )

    accuracy = (
        100 * correct / total
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.2f}%"
    )

    return (
        accuracy,
        np.array(all_true_labels),
        np.array(all_predictions)
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def print_classification_report(
    y_true,
    y_pred
):

    report = classification_report(
        y_true,
        y_pred
    )

    print(
        "\nClassification Report:\n"
    )

    print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    top_n=30
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # Find classes with the highest
    # number of test samples
    class_counts = np.bincount(
        y_true
    )

    top_classes = np.argsort(
        class_counts
    )[-top_n:]

    cm_top = cm[
        np.ix_(
            top_classes,
            top_classes
        )
    ]

    # Normalize row-wise
    row_sums = cm_top.sum(
        axis=1,
        keepdims=True
    )

    cm_normalized = np.divide(
        cm_top,
        row_sums,
        out=np.zeros_like(
            cm_top,
            dtype=float
        ),
        where=row_sums != 0
    )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        cm_normalized,
        interpolation="nearest"
    )

    plt.title(
        "Normalized Confusion Matrix - Top 30 Writers"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.show()


# ============================================================
# TRAINING LOSS CURVE
# ============================================================

def plot_training_loss(
    train_losses
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        range(
            1,
            len(train_losses) + 1
        ),
        train_losses
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training Loss"
    )

    plt.grid()

    plt.show()


# ============================================================
# TRAINING ACCURACY CURVE
# ============================================================

def plot_training_accuracy(
    train_accuracies
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        range(
            1,
            len(train_accuracies) + 1
        ),
        train_accuracies
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Training Accuracy"
    )

    plt.grid()

    plt.show()
