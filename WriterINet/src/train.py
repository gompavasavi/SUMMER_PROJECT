# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS
)

from models import WriterINetANN


# ============================================================
# CREATE ANN MODEL
# ============================================================

def create_ann_model(
    num_classes
):

    model = WriterINetANN(
        num_classes=num_classes
    )

    model = model.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return (
        model,
        criterion,
        optimizer
    )


# ============================================================
# TRAIN ANN
# ============================================================

def train_model(
    model,
    criterion,
    optimizer,
    train_loader
):

    train_losses = []

    train_accuracies = []

    for epoch in range(
        NUM_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for features, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        ):

            features = features.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            # Forward pass
            outputs = model(
                features
            )

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Clear gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += (
                loss.item()
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        epoch_loss = (
            running_loss /
            len(train_loader)
        )

        epoch_accuracy = (
            100 * correct / total
        )

        train_losses.append(
            epoch_loss
        )

        train_accuracies.append(
            epoch_accuracy
        )

        print()

        print(
            f"Epoch "
            f"[{epoch + 1}/{NUM_EPOCHS}]"
        )

        print(
            f"Loss: "
            f"{epoch_loss:.4f}"
        )

        print(
            f"Accuracy: "
            f"{epoch_accuracy:.2f}%"
        )

    return (
        model,
        train_losses,
        train_accuracies
    )
