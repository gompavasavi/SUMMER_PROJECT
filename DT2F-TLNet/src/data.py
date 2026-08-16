# data.py

import os
import random

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from config import (
    ORG_DIR,
    FORG_DIR,
    SEED,
    IMG_SIZE,
    BATCH_SIZE
)


# ============================================================
# LOAD CEDAR DATASET
# ============================================================

def load_dataset():

    image_paths = []
    labels = []

    # Genuine = 1
    for fname in os.listdir(ORG_DIR):

        if fname.endswith(".png"):

            image_paths.append(
                os.path.join(
                    ORG_DIR,
                    fname
                )
            )

            labels.append(1)

    # Forged = 0
    for fname in os.listdir(FORG_DIR):

        if fname.endswith(".png"):

            image_paths.append(
                os.path.join(
                    FORG_DIR,
                    fname
                )
            )

            labels.append(0)

    print(
        "Total Images :",
        len(image_paths)
    )

    print(
        "Genuine      :",
        sum(labels)
    )

    print(
        "Forged       :",
        len(labels) - sum(labels)
    )

    return image_paths, labels


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(
    image_paths,
    labels
):

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            image_paths,
            labels,
            test_size=0.20,
            random_state=SEED,
            stratify=labels
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=SEED,
            stratify=y_temp
        )
    )

    print("=" * 50)
    print("DATASET SPLIT")
    print("=" * 50)

    print(
        "Train :",
        len(X_train)
    )

    print(
        "Val   :",
        len(X_val)
    )

    print(
        "Test  :",
        len(X_test)
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# READ IMAGE
# ============================================================

def read_image(
    path,
    label
):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE]
    )

    img = tf.cast(
        img,
        tf.float32
    ) / 255.0

    # InceptionV3 preprocessing
    img = (
        img - 0.5
    ) * 2.0

    return (
        img,
        tf.cast(
            label,
            tf.float32
        )
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment(
    img,
    label
):

    img = tf.image.random_brightness(
        img,
        max_delta=0.15
    )

    img = tf.image.random_contrast(
        img,
        lower=0.85,
        upper=1.15
    )

    crop_size = tf.random.uniform(
        [],
        minval=190,
        maxval=224,
        dtype=tf.int32
    )

    img = tf.image.random_crop(
        img,
        [
            crop_size,
            crop_size,
            3
        ]
    )

    img = tf.image.resize(
        img,
        [224, 224]
    )

    dx = tf.random.uniform(
        [],
        minval=-15,
        maxval=15,
        dtype=tf.int32
    )

    dy = tf.random.uniform(
        [],
        minval=-15,
        maxval=15,
        dtype=tf.int32
    )

    img = tf.roll(
        img,
        dx,
        axis=1
    )

    img = tf.roll(
        img,
        dy,
        axis=0
    )

    img = tf.clip_by_value(
        img,
        -1.0,
        1.0
    )

    return (
        img,
        label
    )


# ============================================================
# CREATE TF.DATA DATASET
# ============================================================

def make_dataset(
    paths,
    labels,
    training=False
):

    auto = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(paths),
            tf.constant(
                labels,
                dtype=tf.float32
            )
        )
    )

    if training:

        ds = ds.shuffle(
            len(paths),
            seed=SEED
        )

    ds = ds.map(
        read_image,
        num_parallel_calls=auto
    )

    if training:

        ds = ds.map(
            augment,
            num_parallel_calls=auto
        )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        auto
    )

    return ds


# ============================================================
# CREATE ALL DATASETS
# ============================================================

def create_datasets():

    image_paths, labels = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(
        image_paths,
        labels
    )

    train_ds = make_dataset(
        X_train,
        y_train,
        training=True
    )

    val_ds = make_dataset(
        X_val,
        y_val,
        training=False
    )

    test_ds = make_dataset(
        X_test,
        y_test,
        training=False
    )

    return (
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
    )
