# dataset.py

import os
import glob
import tarfile
import xml.etree.ElementTree as ET

import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

from tqdm import tqdm

from config import (
    WORDS_TGZ,
    XML_TGZ,
    EXTRACTED_DIR,
    IMAGE_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
    IMAGE_BATCH_SIZE,
    NUM_WORKERS
)


# ============================================================
# DATASET EXTRACTION
# ============================================================

def extract_dataset():

    print("Extracting words.tgz...")

    with tarfile.open(WORDS_TGZ) as tar:
        tar.extractall(
            path=EXTRACTED_DIR
        )

    print("Extracting xml.tgz...")

    with tarfile.open(XML_TGZ) as tar:
        tar.extractall(
            path=EXTRACTED_DIR
        )

    print("DATASET EXTRACTION COMPLETED")


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset():

    png_files = glob.glob(
        os.path.join(
            EXTRACTED_DIR,
            "**",
            "*.png"
        ),
        recursive=True
    )

    xml_files = glob.glob(
        os.path.join(
            EXTRACTED_DIR,
            "**",
            "*.xml"
        ),
        recursive=True
    )

    print(
        "Total PNG Files:",
        len(png_files)
    )

    print(
        "Total XML Files:",
        len(xml_files)
    )

    return png_files, xml_files


# ============================================================
# XML PARSING
# ============================================================

def create_dataframe(xml_paths):

    data = []

    for xml_path in tqdm(
        xml_paths,
        desc="Parsing XML files"
    ):

        try:

            tree = ET.parse(xml_path)

            root = tree.getroot()

            writer_id = root.attrib.get(
                "writer-id",
                "unknown"
            )

            for word in root.iter("word"):

                word_id = word.attrib.get("id")

                if not word_id:
                    continue

                folder1 = word_id[:3]

                folder2 = "-".join(
                    word_id.split("-")[:2]
                )

                image_path = os.path.join(
                    EXTRACTED_DIR,
                    folder1,
                    folder2,
                    f"{word_id}.png"
                )

                if os.path.exists(image_path):

                    data.append(
                        [
                            image_path,
                            writer_id
                        ]
                    )

        except Exception as e:

            print(
                f"Error processing {xml_path}: {e}"
            )

    df = pd.DataFrame(
        data,
        columns=[
            "image_path",
            "writer_id"
        ]
    )

    print(
        "\nTotal Samples:",
        len(df)
    )

    print(
        "Unique Writers:",
        df["writer_id"].nunique()
    )

    return df


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(df):

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(
        df["writer_id"]
    )

    print(
        "Total Classes:",
        df["label"].nunique()
    )

    return df, encoder


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_dataset(df):

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    print(
        "Train Samples:",
        len(train_df)
    )

    print(
        "Test Samples:",
        len(test_df)
    )

    return train_df, test_df


# ============================================================
# CUSTOM IAM DATASET
# ============================================================

class IAMDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transforms.Compose([

            transforms.Resize(
                IMAGE_SIZE
            ),

            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.ToTensor()

        ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        while True:

            try:

                image_path = self.df.iloc[
                    idx
                ]["image_path"]

                label = self.df.iloc[
                    idx
                ]["label"]

                image = Image.open(
                    image_path
                ).convert("RGB")

                image = self.transform(
                    image
                )

                return image, label

            except Exception:

                idx = (
                    idx + 1
                ) % len(self.df)


# ============================================================
# CREATE IMAGE DATALOADERS
# ============================================================

def create_image_loaders(
    train_df,
    test_df
):

    train_dataset = IAMDataset(
        train_df
    )

    test_dataset = IAMDataset(
        test_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        "Train Loader Ready"
    )

    print(
        "Test Loader Ready"
    )

    return (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    )
