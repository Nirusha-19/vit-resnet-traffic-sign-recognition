"""
Robustness testing under synthetic corruptions.
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (VIT_MODEL_NAME, VIT_CHECKPOINT, RESNET_CHECKPOINT,
                     ORGANIZED_TEST_DIR, NUM_CLASSES, IMAGE_SIZE, DEVICE, ROBUSTNESS_RESULTS)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score

NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
BASE_RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

CONDITIONS = {
    "clean": transforms.Compose([BASE_RESIZE, transforms.ToTensor(), NORMALIZE]),
    "blur": transforms.Compose([
        BASE_RESIZE, transforms.GaussianBlur(kernel_size=9, sigma=(3.0, 5.0)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "low_light": transforms.Compose([
        BASE_RESIZE, transforms.ColorJitter(brightness=(0.25, 0.35)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "noise": transforms.Compose([
        BASE_RESIZE, transforms.ToTensor(),
        transforms.Lambda(lambda t: torch.clamp(t + torch.randn_like(t) * 0.15, 0, 1)),
        NORMALIZE,
    ]),
}


def load_model(model_type):
    if model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            VIT_MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(VIT_CHECKPOINT, map_location="cpu"))
    else:
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def evaluate_under_condition(model, condition_name, transform):
    is_ppm = lambda path: path.lower().endswith(".ppm")
    test_ds = ImageFolder(ORGANIZED_TEST_DIR, transform=transform, is_valid_file=is_ppm)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return accuracy_score(all_labels, all_preds)


def main():
    results = {}
    for model_type in ["vit", "resnet"]:
        print(f"\n=== Testing {model_type} under each condition ===")
        model = load_model(model_type)
        model_results = {}
        for condition_name, transform in CONDITIONS.items():
            acc = evaluate_under_condition(model, condition_name, transform)
            model_results[condition_name] = acc
            print(f"  {condition_name:12s}: {acc:.4f}")
        results[model_type] = model_results

    print("\n" + "=" * 60)
    print("Accuracy drop from clean baseline (the number that matters):")
    print("=" * 60)
    for model_type, model_results in results.items():
        clean_acc = model_results["clean"]
        print(f"\n{model_type}:")
        for condition_name, acc in model_results.items():
            if condition_name == "clean":
                continue
            drop = clean_acc - acc
            print(f"  {condition_name:12s}: {acc:.4f} (-{drop:.4f} from clean)")

    os.makedirs(os.path.dirname(ROBUSTNESS_RESULTS), exist_ok=True)
    with open(ROBUSTNESS_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {ROBUSTNESS_RESULTS}")


if __name__ == "__main__":
    main()
