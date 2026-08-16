# train.py

import tensorflow as tf
from tensorflow import keras

from model import build_dt2f_tlnet


# ============================================================
# PHASE 1: TRANSFER LEARNING
# ============================================================

def train_phase1(
    train_ds,
    val_ds,
    num_classes=2,
    epochs=15
):

    print("=" * 60)
    print("PHASE 1: TRANSFER LEARNING")
    print("=" * 60)

    model = build_dt2f_tlnet(
        num_classes=num_classes
    )

    # InceptionV3 remains frozen
    inception = model.get_layer(
        "inception_v3"
    )

    inception.trainable = False

    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=5e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            "best_verification_p1.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history


# ============================================================
# PHASE 2: FINE-TUNING
# ============================================================

def train_phase2(
    model,
    train_ds,
    val_ds,
    epochs=10
):

    print("=" * 60)
    print("PHASE 2: FINE-TUNING")
    print("=" * 60)

    inception = model.get_layer(
        "inception_v3"
    )

    # First freeze all layers
    inception.trainable = True

    # Freeze all but the last 20 layers
    for layer in inception.layers[:-20]:

        layer.trainable = False

    for layer in inception.layers[-20:]:

        layer.trainable = True

    # Recompile with smaller learning rate
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            "best_verification_p2.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def train_model(
    train_ds,
    val_ds,
    num_classes=2
):

    # Phase 1
    model, history_p1 = train_phase1(
        train_ds,
        val_ds,
        num_classes=num_classes,
        epochs=15
    )

    # Phase 2
    model, history_p2 = train_phase2(
        model,
        train_ds,
        val_ds,
        epochs=10
    )

    return (
        model,
        history_p1,
        history_p2
    )
