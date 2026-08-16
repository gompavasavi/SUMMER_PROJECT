# models.py

import torch
import torch.nn as nn

from torchvision import models

from torchvision.models import (
    DenseNet201_Weights,
    ResNet50_Weights
)

from config import (
    DEVICE,
    FUSED_FEATURE_SIZE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    DROPOUT
)


# ============================================================
# DENSENET201 FEATURE EXTRACTOR
# ============================================================

def create_densenet():

    densenet = models.densenet201(
        weights=DenseNet201_Weights.DEFAULT
    )

    # Remove final classifier
    densenet.classifier = nn.Identity()

    densenet = densenet.to(
        DEVICE
    )

    densenet.eval()

    return densenet


# ============================================================
# RESNET50 FEATURE EXTRACTOR
# ============================================================

def create_resnet():

    resnet = models.resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    # Remove final classifier
    resnet.fc = nn.Identity()

    resnet = resnet.to(
        DEVICE
    )

    resnet.eval()

    return resnet


# ============================================================
# WRITERINET ANN
# ============================================================

class WriterINetANN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                FUSED_FEATURE_SIZE,
                HIDDEN_SIZE_1
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_SIZE_1,
                HIDDEN_SIZE_2
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_SIZE_2,
                num_classes
            )

        )

    def forward(self, x):

        return self.network(x)
