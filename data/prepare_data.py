"""
Downloads the official GTSRB release (39,209 training images, 12,630 test images) and organizes the flat test set into class subfolders using the
official ground-truth CSV.
"""
import os
import sys
import shutil
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (GTSRB_BASE_URL, RAW_DIR, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
                     TEST_GT_CSV, ORGANIZED_TEST_DIR, NUM_CLASSES)

import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive


def download_if_needed():
    os.makedirs(RAW_DIR, exist_ok=True)

    if not os.path.isdir(TRAIN_IMAGES_DIR):
        print("Downloading official training images (39,209 images, ~280MB)...")
        download_and_extract_archive(
            f"{GTSRB_BASE_URL}GTSRB_Final_Training_Images.zip",
            download_root=RAW_DIR,
        )
    else:
        print("Training images already downloaded, skipping.")

    if not os.path.isdir(TEST_IMAGES_DIR):
        print("Downloading official test images (12,630 images, ~85MB)...")
        download_and_extract_archive(
            f"{GTSRB_BASE_URL}GTSRB_Final_Test_Images.zip",
            download_root=RAW_DIR,
        )
    else:
        print("Test images already downloaded, skipping.")

    if not os.path.exists(TEST_GT_CSV):
        print("Downloading test ground-truth labels...")
        download_and_extract_archive(
            f"{GTSRB_BASE_URL}GTSRB_Final_Test_GT.zip",
            download_root=TEST_IMAGES_DIR,
        )
    else:
        print("Test ground-truth CSV already present, skipping.")


def organize_test_set():
    if os.path.isdir(ORGANIZED_TEST_DIR) and len(os.listdir(ORGANIZED_TEST_DIR)) == NUM_CLASSES:
        print("Test set already organized into class folders, skipping.")
        return

    print(f"\nReading test ground truth from {TEST_GT_CSV}...")
    df = pd.read_csv(TEST_GT_CSV, sep=";")
    if "ClassId" not in df.columns or "Filename" not in df.columns:
        print(f"ERROR: expected 'Filename' and 'ClassId' columns, found: {list(df.columns)}")
        return

    for class_id in range(NUM_CLASSES):
        os.makedirs(os.path.join(ORGANIZED_TEST_DIR, f"{class_id:05d}"), exist_ok=True)

    print("Organizing test images into class folders (one-time step)...")
    copied = 0
    for _, row in df.iterrows():
        src = os.path.join(TEST_IMAGES_DIR, row["Filename"])
        dst = os.path.join(ORGANIZED_TEST_DIR, f"{row['ClassId']:05d}", row["Filename"])
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            copied += 1
    print(f"  copied {copied} test images into class folders.")


def main():
    download_if_needed()
    organize_test_set()

    print("\nCounting images per class in the training set...")
    counts = Counter()
    for class_folder in sorted(os.listdir(TRAIN_IMAGES_DIR)):
        class_path = os.path.join(TRAIN_IMAGES_DIR, class_folder)
        if os.path.isdir(class_path):
            n = len([f for f in os.listdir(class_path) if f.endswith(".ppm")])
            counts[class_folder] = n

    total_train = sum(counts.values())
    print(f"\nTotal training images: {total_train}")
    for class_folder in sorted(counts):
        print(f"  class {class_folder}: {counts[class_folder]:5d} images")

    total_test = sum(
        len([f for f in os.listdir(os.path.join(ORGANIZED_TEST_DIR, d)) if f.endswith(".ppm")])
        for d in os.listdir(ORGANIZED_TEST_DIR)
    )
    print(f"\nTotal test images (organized): {total_test}")

    most_common = max(counts.values())
    least_common = min(counts.values())
    print(f"\nMost common class: {most_common} images. "
          f"Least common class: {least_common} images. "
          f"Imbalance ratio: {most_common / least_common:.1f}x")


if __name__ == "__main__":
    main()
