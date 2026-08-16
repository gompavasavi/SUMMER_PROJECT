# main.py

from data import create_datasets

from model import build_dt2f_tlnet

from train import train_model

from evaluate import (
    evaluate_model,
    get_predictions,
    classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_tsne
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("DT2F-TLNet")
    print("Text-Independent Signature Verification")
    print("=" * 60)


    # ========================================================
    # 1. CREATE DATASETS
    # ========================================================

    print("\n[1] Preparing CEDAR dataset...")

    (
        train_ds,
        val_ds,
        test_ds,
        image_paths,
        labels,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = create_datasets()


    # ========================================================
    # 2. TRAIN MODEL
    # ========================================================

    print("\n[2] Training DT2F-TLNet...")

    (
        model,
        history_p1,
        history_p2
    ) = train_model(
        train_ds,
        val_ds,
        num_classes=2
    )


    # ========================================================
    # 3. TEST EVALUATION
    # ========================================================

    print("\n[3] Evaluating model...")

    evaluate_model(
        model,
        test_ds
    )


    # ========================================================
    # 4. PREDICTIONS
    # ========================================================

    print("\n[4] Generating predictions...")

    (
        y_true,
        y_pred,
        y_prob
    ) = get_predictions(
        model,
        test_ds
    )


    # ========================================================
    # 5. CLASSIFICATION METRICS
    # ========================================================

    print("\n[5] Classification metrics...")

    classification_metrics(
        y_true,
        y_pred
    )


    # ========================================================
    # 6. CONFUSION MATRIX
    # ========================================================

    print("\n[6] Confusion matrix...")

    plot_confusion_matrix(
        y_true,
        y_pred
    )


    # ========================================================
    # 7. ROC CURVE
    # ========================================================

    print("\n[7] ROC curve...")

    plot_roc_curve(
        y_true,
        y_prob
    )


    # ========================================================
    # 8. t-SNE
    # ========================================================

    print("\n[8] t-SNE visualization...")

    plot_tsne(
        model,
        test_ds
    )


    print("\n" + "=" * 60)
    print("DT2F-TLNet PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()
