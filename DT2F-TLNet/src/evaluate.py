# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.manifold import TSNE

from tensorflow import keras


# ============================================================
# BASIC TEST EVALUATION
# ============================================================

def evaluate_model(model, test_ds):

    test_loss, test_acc, test_auc = model.evaluate(
        test_ds,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    print(
        f"Test Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_acc * 100:.2f}%"
    )

    print(
        f"Test AUC      : {test_auc:.4f}"
    )

    return (
        test_loss,
        test_acc,
        test_auc
    )


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    model,
    test_ds
):

    y_prob = model.predict(
        test_ds,
        verbose=1
    )

    y_prob = y_prob.ravel()

    y_pred = (
        y_prob > 0.5
    ).astype(int)

    y_true = np.concatenate(
        [
            y.numpy()
            for x, y in test_ds
        ]
    ).astype(int)

    return (
        y_true,
        y_pred,
        y_prob
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("TEST METRICS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Forged",
                "Genuine"
            ]
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Forged",
            "Genuine"
        ],
        yticklabels=[
            "Forged",
            "Genuine"
        ]
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        "DT2F-TLNet Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()

    print(
        "Confusion Matrix:"
    )

    print(cm)

    return cm


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true,
    y_prob
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"DT2F-TLNet "
            f"(AUC = {roc_auc:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlim(
        [0, 1]
    )

    plt.ylim(
        [0, 1.05]
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - DT2F-TLNet Signature Verification"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    print(
        f"AUC = {roc_auc:.4f}"
    )

    return (
        fpr,
        tpr,
        roc_auc
    )


# ============================================================
# t-SNE VISUALIZATION
# ============================================================

def plot_tsne(
    model,
    test_ds
):

    # Use the layer before the final output
    feature_model = keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )

    features = feature_model.predict(
        test_ds,
        verbose=1
    )

    y_true = np.concatenate(
        [
            y.numpy()
            for x, y in test_ds
        ]
    ).astype(int)

    print(
        "Feature Shape:",
        features.shape
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    tsne_features = tsne.fit_transform(
        features
    )

    plt.figure(
        figsize=(8, 6)
    )

    for cls, color, name in [
        (0, "red", "Forged"),
        (1, "blue", "Genuine")
    ]:

        idx = (
            y_true == cls
        )

        plt.scatter(
            tsne_features[
                idx,
                0
            ],
            tsne_features[
                idx,
                1
            ],
            c=color,
            label=name,
            alpha=0.7,
            s=25
        )

    plt.legend()

    plt.title(
        "t-SNE Visualization of DT2F-TLNet Features"
    )

    plt.xlabel(
        "t-SNE Dimension 1"
    )

    plt.ylabel(
        "t-SNE Dimension 2"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    return tsne_features
