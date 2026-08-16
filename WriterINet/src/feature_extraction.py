# feature_extraction.py

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from tqdm import tqdm

from config import (
    DEVICE,
    FEATURE_BATCH_SIZE,
    NUM_WORKERS,
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    loader,
    densenet,
    resnet,
    description
):

    features = []

    labels_list = []

    densenet.eval()
    resnet.eval()

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc=description
        ):

            images = images.to(
                DEVICE
            )

            # DenseNet features
            dense_features = densenet(
                images
            )

            # ResNet features
            res_features = resnet(
                images
            )

            # Feature fusion
            fused_features = torch.cat(
                [
                    dense_features,
                    res_features
                ],
                dim=1
            )

            features.append(
                fused_features
                .cpu()
                .numpy()
            )

            labels_list.append(
                labels.numpy()
            )

    X = np.concatenate(
        features,
        axis=0
    )

    y = np.concatenate(
        labels_list,
        axis=0
    )

    return X, y


# ============================================================
# EXTRACT TRAIN FEATURES
# ============================================================

def extract_train_features(
    train_loader,
    densenet,
    resnet
):

    X_train, y_train = extract_features(
        train_loader,
        densenet,
        resnet,
        "Extracting train features"
    )

    print(
        "Train Features:",
        X_train.shape
    )

    print(
        "Train Labels:",
        y_train.shape
    )

    np.save(
        X_TRAIN_FILE,
        X_train
    )

    np.save(
        Y_TRAIN_FILE,
        y_train
    )

    print(
        "Training features saved."
    )

    return X_train, y_train


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

def extract_test_features(
    test_loader,
    densenet,
    resnet
):

    X_test, y_test = extract_features(
        test_loader,
        densenet,
        resnet,
        "Extracting test features"
    )

    print(
        "Test Features:",
        X_test.shape
    )

    print(
        "Test Labels:",
        y_test.shape
    )

    np.save(
        X_TEST_FILE,
        X_test
    )

    np.save(
        Y_TEST_FILE,
        y_test
    )

    print(
        "Testing features saved."
    )

    return X_test, y_test


# ============================================================
# CREATE FEATURE DATALOADERS
# ============================================================

def create_feature_loaders(
    X_train,
    y_train,
    X_test,
    y_test
):

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        "FEATURE LOADERS READY"
    )

    return (
        train_loader,
        test_loader
    )
