# config.py

import os
import random
import numpy as np
import tensorflow as tf


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATASET PATHS
# ============================================================

ORG_DIR = (
    "/kaggle/input/datasets/"
    "shreelakshmigp/cedardataset/"
    "signatures/full_org"
)

FORG_DIR = (
    "/kaggle/input/datasets/"
    "shreelakshmigp/cedardataset/"
    "signatures/full_forg"
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 32


# ============================================================
# DATASET SPLIT
# ============================================================

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.50


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 2


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 10

PHASE1_LEARNING_RATE = 1e-3
PHASE2_LEARNING_RATE = 1e-5


# ============================================================
# CHECKPOINT PATHS
# ============================================================

WORKING_DIR = "/kaggle/working"

PHASE1_CHECKPOINT = os.path.join(
    WORKING_DIR,
    "best_verification_p1.keras"
)

PHASE2_CHECKPOINT = os.path.join(
    WORKING_DIR,
    "best_verification_p2.keras"
)

PHASE1_MODEL = os.path.join(
    WORKING_DIR,
    "verification_phase1.keras"
)
