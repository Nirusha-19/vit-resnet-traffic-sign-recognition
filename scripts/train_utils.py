"""
Shared utilities for training and evaluating both models on GTSRB: dataset loading with a stratified train/val split, focal loss (to
handle the genuine imbalance across the 43 sign classes), and a common evaluation function reporting accuracy, macro-averaged precision/
recall/F1, and the recall on the single worst-performing class, called out specifically, since that's the class most likely to be a
rare sign type the model struggles to learn.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (TRAIN_IMAGES_DIR, ORGANIZED_TEST_DIR, NUM_CLASSES,
                     IMAGE_SIZE, DEVICE, FOCAL_LOSS_GAMMA)

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

VAL_FRACTION = 0.15
RANDOM_SEED = 42


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class FocalLoss(nn.Module):
    """Generalizes cross-entropy by down-weighting predictions the model
    already gets right easily (high predicted probability for the
    correct class), so training effort concentrates on the harder,
    rarer classes -- a stronger fix for severe imbalance than plain
    class-weighted loss alone. Class weights (alpha) are still applied
    on top, folding both techniques together.
    """
    def __init__(self, class_weights=None, gamma=FOCAL_LOSS_GAMMA):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce_loss)  # predicted probability of the true class
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()


def get_dataloaders(batch_size=16):
    # PPM is a supported extension for ImageFolder, but pass an explicit
    # is_valid_file check anyway -- each class folder also contains a
    # GT-<classid>.csv annotation file, and this guarantees only the
    # actual .ppm images get picked up, not the CSV alongside them.
    is_ppm = lambda path: path.lower().endswith(".ppm")

    train_full = ImageFolder(TRAIN_IMAGES_DIR, transform=get_transforms(train=True), is_valid_file=is_ppm)
    val_full = ImageFolder(TRAIN_IMAGES_DIR, transform=get_transforms(train=False), is_valid_file=is_ppm)

    # ImageFolder exposes .targets directly -- no need to read every label manually just to build the stratified split
    labels = train_full.targets

    train_idx, val_idx = train_test_split(
        range(len(train_full)), test_size=VAL_FRACTION,
        stratify=labels, random_state=RANDOM_SEED
    )

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(val_full, val_idx)

    test_ds = ImageFolder(ORGANIZED_TEST_DIR, transform=get_transforms(train=False), is_valid_file=is_ppm)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_labels = [labels[i] for i in train_idx]
    return train_loader, val_loader, test_loader, train_labels


def compute_class_weights(train_labels, num_classes=NUM_CLASSES):
    counts = torch.zeros(num_classes)
    for label in train_labels:
        counts[label] += 1
    counts = torch.clamp(counts, min=1)  # avoid divide-by-zero for any unseen class
    total = counts.sum()
    weights = total / (num_classes * counts)
    print(f"Class weight range: min={weights.min():.3f}, max={weights.max():.3f} "
          f"(higher weight = rarer class)")
    return weights


def evaluate_model(model, data_loader, topk=(3, 5)):
    """Top-1 metrics here are what actually matters for deployment --
    this is a safety-relevant classification task, and there's no such
    thing as "close enough" when a wrong answer could mean acting on
    the wrong speed limit or missing a stop sign. Top-1 accuracy,
    precision, recall, and F1 are the numbers to trust.

    Top-K accuracy (below) is included purely as a DIAGNOSTIC metric,
    to understand whether the model's mistakes are near-misses (e.g.
    confusing two similar speed limit signs) or wildly wrong (confusing
    a speed limit with a stop sign). It is NOT a substitute for Top-1
    accuracy and should never be reported as if it reflects real-world
    reliability.
    """
    model.eval()
    all_preds, all_labels = [], []
    topk_correct = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

            total += labels.size(0)
            max_k = max(topk)
            _, topk_preds = logits.topk(max_k, dim=1)
            for k in topk:
                correct_k = (topk_preds[:, :k] == labels.to(DEVICE).unsqueeze(1)).any(dim=1)
                topk_correct[k] += correct_k.sum().item()

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES))
    )
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    worst_class = int(recall.argmin())
    worst_class_recall = float(recall[worst_class])

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    topk_accuracy = {f"top{k}_accuracy_DIAGNOSTIC_ONLY": topk_correct[k] / total for k in topk}

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "worst_class": worst_class,
        "worst_class_recall": worst_class_recall,
        "confusion_matrix": cm.tolist(),
        **topk_accuracy,
    }


def save_results(results, path):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")
