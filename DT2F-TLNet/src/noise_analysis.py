# noise_analysis.py

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# ADD GAUSSIAN NOISE AT A GIVEN SNR
# ============================================================

def add_noise_snr(
    img,
    snr_db
):

    img = img.astype(
        np.float32
    )

    signal_power = np.mean(
        img ** 2
    )

    snr_linear = (
        10 ** (snr_db / 10)
    )

    noise_power = (
        signal_power /
        snr_linear
    )

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        img.shape
    )

    noisy = img + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(
        np.uint8
    )


# ============================================================
# VISUALIZE DIFFERENT SNR LEVELS
# ============================================================

def visualize_snr_levels(
    img,
    snr_levels=None
):

    if snr_levels is None:

        snr_levels = [
            40,
            30,
            20,
            15,
            10,
            5,
            0,
            -4
        ]

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(12, 8)
    )

    for ax, snr in zip(
        axes.ravel(),
        snr_levels
    ):

        noisy_img = add_noise_snr(
            img,
            snr
        )

        ax.imshow(
            noisy_img,
            cmap="gray"
        )

        ax.set_title(
            f"SNR = {snr} dB"
        )

        ax.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# GENERATE A SINGLE NOISY IMAGE
# ============================================================

def generate_noisy_image(
    img,
    snr_db
):

    return add_noise_snr(
        img,
        snr_db
    )
