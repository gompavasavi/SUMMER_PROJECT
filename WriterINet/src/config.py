# config.py

import os
import torch


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET PATHS
# ============================================================

DATASET_PATH = (
    "/kaggle/input/"
    "datasets/teykaicong/"
    "iamondb-handwriting-dataset"
)

WORDS_TGZ = os.path.join(
    DATASET_PATH,
    "words.tgz"
)

XML_TGZ = os.path.join(
    DATASET_PATH,
    "xml.tgz"
)

WORKING_DIR = "/kaggle/working"


# ============================================================
# EXTRACTED DATA PATH
# ============================================================

EXTRACTED_DIR = WORKING_DIR


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)


# ============================================================
# TRAIN / TEST CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DATALOADER CONFIGURATION
# ============================================================

IMAGE_BATCH_SIZE = 128
FEATURE_BATCH_SIZE = 256

NUM_WORKERS = 0


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

DENSENET_FEATURE_SIZE = 1920
RESNET_FEATURE_SIZE = 2048

FUSED_FEATURE_SIZE = (
    DENSENET_FEATURE_SIZE +
    RESNET_FEATURE_SIZE
)


# ============================================================
# ANN CONFIGURATION
# ============================================================

HIDDEN_SIZE_1 = 1024
HIDDEN_SIZE_2 = 512

DROPOUT = 0.3

LEARNING_RATE = 0.001

NUM_EPOCHS = 30


# ============================================================
# OUTPUT FILES
# ============================================================

X_TRAIN_FILE = os.path.join(
    WORKING_DIR,
    "X_train.npy"
)

Y_TRAIN_FILE = os.path.join(
    WORKING_DIR,
    "y_train.npy"
)

X_TEST_FILE = os.path.join(
    WORKING_DIR,
    "X_test.npy"
)

Y_TEST_FILE = os.path.join(
    WORKING_DIR,
    "y_test.npy"
)

MODEL_FILE = os.path.join(
    WORKING_DIR,
    "writerinet_ann_model.pth"
)
