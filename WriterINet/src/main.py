# main.py

import numpy as np
import torch

from config import (
    DEVICE,
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_TEST_FILE,
    Y_TEST_FILE,
    MODEL_FILE
)

from dataset import (
    extract_dataset,
    verify_dataset,
    create_dataframe,
    encode_labels,
    split_dataset,
    create_image_loaders
)

from models import (
    create_densenet,
    create_resnet
)

from feature_extraction import (
    extract_train_features,
    extract_test_features,
    create_feature_loaders
)

from train import (
    create_ann_model,
    train_model
)

from evaluate import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_loss,
    plot_training_accuracy
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WriterINet")
    print("=" * 60)

    print(
        "\nUsing Device:",
        DEVICE
    )


    # ========================================================
    # 1. EXTRACT DATASET
    # ========================================================

    print(
        "\n[1] Dataset Extraction"
    )

    extract_dataset()


    # ========================================================
    # 2. VERIFY DATASET
    # ========================================================

    print(
        "\n[2] Dataset Verification"
    )

    _, xml_paths = verify_dataset()


    # ========================================================
    # 3. CREATE DATAFRAME
    # ========================================================

    print(
        "\n[3] Creating Dataset DataFrame"
    )

    df = create_dataframe(
        xml_paths
    )


    # ========================================================
    # 4. LABEL ENCODING
    # ========================================================

    print(
        "\n[4] Label Encoding"
    )

    df, encoder = encode_labels(
        df
    )

    num_classes = df[
        "label"
    ].nunique()

    print(
        "Number of classes:",
        num_classes
    )


    # ========================================================
    # 5. TRAIN TEST SPLIT
    # ========================================================

    print(
        "\n[5] Train/Test Split"
    )

    train_df, test_df = split_dataset(
        df
    )


    # ========================================================
    # 6. IMAGE DATA LOADERS
    # ========================================================

    print(
        "\n[6] Creating Image Loaders"
    )

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = create_image_loaders(
        train_df,
        test_df
    )


    # ========================================================
    # 7. CREATE CNN MODELS
    # ========================================================

    print(
        "\n[7] Loading CNN Feature Extractors"
    )

    densenet = create_densenet()

    resnet = create_resnet()

    print(
        "DenseNet201 loaded."
    )

    print(
        "ResNet50 loaded."
    )


    # ========================================================
    # 8. FEATURE EXTRACTION
    # ========================================================

    print(
        "\n[8] Extracting Training Features"
    )

    X_train, y_train = extract_train_features(
        train_loader,
        densenet,
        resnet
    )


    print(
        "\n[9] Extracting Testing Features"
    )

    X_test, y_test = extract_test_features(
        test_loader,
        densenet,
        resnet
    )


    # ========================================================
    # 9. FEATURE LOADERS
    # ========================================================

    print(
        "\n[10] Creating Feature DataLoaders"
    )

    (
        train_feature_loader,
        test_feature_loader
    ) = create_feature_loaders(
        X_train,
        y_train,
        X_test,
        y_test
    )


    # ========================================================
    # 10. ANN MODEL
    # ========================================================

    print(
        "\n[11] Creating ANN"
    )

    (
        ann_model,
        criterion,
        optimizer
    ) = create_ann_model(
        num_classes
    )


    # ========================================================
    # 11. TRAIN ANN
    # ========================================================

    print(
        "\n[12] Training ANN"
    )

    (
        ann_model,
        train_losses,
        train_accuracies
    ) = train_model(
        ann_model,
        criterion,
        optimizer,
        train_feature_loader
    )


    # ========================================================
    # 12. SAVE MODEL
    # ========================================================

    print(
        "\n[13] Saving Model"
    )

    torch.save(
        ann_model.state_dict(),
        MODEL_FILE
    )

    print(
        "Model saved to:",
        MODEL_FILE
    )


    # ========================================================
    # 13. EVALUATION
    # ========================================================

    print(
        "\n[14] Model Evaluation"
    )

    (
        test_accuracy,
        y_true,
        y_pred
    ) = evaluate_model(
        ann_model,
        test_feature_loader
    )


    # ========================================================
    # 14. CLASSIFICATION REPORT
    # ========================================================

    print_classification_report(
        y_true,
        y_pred
    )


    # ========================================================
    # 15. TRAINING CURVES
    # ========================================================

    plot_training_loss(
        train_losses
    )

    plot_training_accuracy(
        train_accuracies
    )


    # ========================================================
    # 16. CONFUSION MATRIX
    # ========================================================

    plot_confusion_matrix(
        y_true,
        y_pred,
        top_n=30
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
