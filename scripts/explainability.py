"""
Explainability visualizations for both models.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (VIT_MODEL_NAME, VIT_CHECKPOINT, RESNET_CHECKPOINT,
                     ORGANIZED_TEST_DIR, OUTPUTS_DIR, NUM_CLASSES, IMAGE_SIZE, DEVICE)
from train_utils import get_transforms

import torch
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

N_SAMPLES = 6
HEATMAP_DIR = os.path.join(OUTPUTS_DIR, "heatmaps")


def load_sample_images(n=N_SAMPLES):
    """Grabs a few real images directly from the organized test set,
    spread across a few different sign classes for variety."""
    is_ppm = lambda path: path.lower().endswith(".ppm")
    test_ds = ImageFolder(ORGANIZED_TEST_DIR, is_valid_file=is_ppm)
    step = max(1, len(test_ds) // n)
    samples = []
    for i in range(0, len(test_ds), step):
        if len(samples) >= n:
            break
        image, label = test_ds[i]
        samples.append((f"sample_{i}_class{label}", image.convert("RGB")))
    return samples


def resnet_gradcam(images):
    print("Generating Grad-CAM heatmaps for ResNet-50...")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location="cpu"))
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    transform = get_transforms(train=False)

    for name, img in images:
        input_tensor = transform(img).unsqueeze(0)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        out_path = os.path.join(HEATMAP_DIR, f"resnet_gradcam_{name}.jpg")
        Image.fromarray(overlay).save(out_path)
        print(f"  saved {out_path}")


def vit_attention_map(images):
    print("Generating attention-map visualizations for ViT...")
    model = ViTForImageClassification.from_pretrained(
        VIT_MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    model.load_state_dict(torch.load(VIT_CHECKPOINT, map_location="cpu"))
    model.eval()

    transform = get_transforms(train=False)
    patches_per_side = IMAGE_SIZE // 16  # ViT-base uses 16x16 patches

    for name, img in images:
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor, output_attentions=True)
        last_layer_attn = outputs.attentions[-1][0]  # (heads, tokens, tokens)
        cls_to_patches = last_layer_attn[:, 0, 1:].mean(dim=0)
        attn_grid = cls_to_patches.reshape(patches_per_side, patches_per_side).numpy()

        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
        attn_img = Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )

        rgb_img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        heatmap = np.array(attn_img.convert("L"))
        overlay = np.array(rgb_img).copy().astype(np.float32)
        overlay[..., 0] = np.clip(overlay[..., 0] + heatmap * 0.5, 0, 255)
        overlay = overlay.astype(np.uint8)

        out_path = os.path.join(HEATMAP_DIR, f"vit_attention_{name}.jpg")
        Image.fromarray(overlay).save(out_path)
        print(f"  saved {out_path}")


def main():
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    images = load_sample_images()
    print(f"Using {len(images)} sample test images, spread across different sign classes.\n")
    resnet_gradcam(images)
    print()
    vit_attention_map(images)


if __name__ == "__main__":
    main()
