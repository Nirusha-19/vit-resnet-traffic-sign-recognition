"""
Fine-tunes a pretrained ResNet-50 for the same GTSRB traffic sign classification task (43 classes), using the identical data split and
focal loss as train_vit.py, so the comparison against the ViT is fair.

Run this after data/prepare_data.py (and independently of train_vit.py).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import RESNET_CHECKPOINT, RESNET_RESULTS, DEVICE, NUM_CLASSES
from train_utils import get_dataloaders, compute_class_weights, evaluate_model, save_results, FocalLoss

import copy
import torch
from tqdm import tqdm
from torch import nn
from torchvision import models

EPOCHS = 5
LEARNING_RATE = 1e-4


def main():
    train_loader, val_loader, test_loader, train_labels = get_dataloaders()
    class_weights = compute_class_weights(train_labels).to(DEVICE)

    print("Loading pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = FocalLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in progress:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_results = evaluate_model(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} -- train loss: {avg_loss:.4f}, "
              f"val accuracy: {val_results['accuracy']:.4f}, "
              f"val macro F1: {val_results['macro_f1']:.4f}, "
              f"worst-class recall: {val_results['worst_class_recall']:.4f}")

        if val_results["macro_f1"] > best_val_f1:
            best_val_f1 = val_results["macro_f1"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  -> new best checkpoint (epoch {epoch}, val macro F1 {best_val_f1:.4f})")

    print(f"\nRestoring best checkpoint from epoch {best_epoch} "
          f"(val macro F1 {best_val_f1:.4f}) for final evaluation...")
    model.load_state_dict(best_state_dict)

    print("\nEvaluating on held-out test set...")
    test_results = evaluate_model(model, test_loader)
    print(f"Test accuracy (Top-1, the number that matters for deployment): {test_results['accuracy']:.4f}")
    print(f"Test macro F1: {test_results['macro_f1']:.4f}")
    print(f"Worst-performing class: {test_results['worst_class']} "
          f"(recall: {test_results['worst_class_recall']:.4f})")
    print(f"[Diagnostic only, not a safety metric] Top-3 accuracy: "
          f"{test_results['top3_accuracy_DIAGNOSTIC_ONLY']:.4f}")
    print(f"[Diagnostic only, not a safety metric] Top-5 accuracy: "
          f"{test_results['top5_accuracy_DIAGNOSTIC_ONLY']:.4f}")

    model.to("cpu")
    os.makedirs(os.path.dirname(RESNET_CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), RESNET_CHECKPOINT)
    print(f"\nSaved model to {RESNET_CHECKPOINT}")

    save_results({**test_results, "best_epoch": best_epoch, "best_val_macro_f1": best_val_f1}, RESNET_RESULTS)


if __name__ == "__main__":
    main()