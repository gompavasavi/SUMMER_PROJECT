# model.py

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3

from fuzzy_layer import Type2Fuzzy


def build_dt2f_tlnet(num_classes=2):

    inp = keras.Input(
        shape=(224, 224, 3)
    )

    # =====================================================
    # MODEL II - InceptionV3 Transfer Learning Branch
    # =====================================================

    inception = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Initially freeze InceptionV3
    inception.trainable = False

    x2 = inception(inp)

    x2 = layers.GlobalAveragePooling2D()(x2)

    x2 = layers.BatchNormalization()(x2)

    F2 = layers.Dense(
        512,
        activation=None
    )(x2)

    # =====================================================
    # MODEL I - Type-2 Fuzzy CNN Branch
    # =====================================================

    x1 = layers.Conv2D(
        16,
        3,
        padding="same",
        use_bias=False
    )(inp)

    x1 = layers.BatchNormalization()(x1)
    x1 = Type2Fuzzy()(x1)
    x1 = layers.MaxPooling2D()(x1)

    # -----------------------------------------------------

    x1 = layers.Conv2D(
        32,
        3,
        padding="same",
        use_bias=False
    )(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = Type2Fuzzy()(x1)
    x1 = layers.MaxPooling2D()(x1)

    # -----------------------------------------------------

    x1 = layers.Conv2D(
        64,
        3,
        padding="same",
        use_bias=False
    )(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = Type2Fuzzy()(x1)
    x1 = layers.MaxPooling2D()(x1)

    # -----------------------------------------------------

    x1 = layers.Conv2D(
        128,
        3,
        padding="same",
        use_bias=False
    )(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = Type2Fuzzy()(x1)
    x1 = layers.MaxPooling2D()(x1)

    # -----------------------------------------------------

    x1 = layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False
    )(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = Type2Fuzzy()(x1)

    F1 = layers.GlobalAveragePooling2D()(x1)

    # =====================================================
    # FEATURE FUSION
    # =====================================================

    fusion = layers.Concatenate()(
        [F1, F2]
    )

    x = layers.Dense(
        200,
        use_bias=False
    )(fusion)

    x = layers.BatchNormalization()(x)

    x = Type2Fuzzy()(x)

    x = layers.Dropout(
        0.5
    )(x)

    # =====================================================
    # OUTPUT
    # =====================================================

    out = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = keras.Model(
        inputs=inp,
        outputs=out,
        name="DT2F_TLNet"
    )

    return model
