"""
Applies INT8 dynamic quantization to the trained model and benchmarks latency (ms per image) and file size, before vs. after -- this is the
project's second headline result, alongside the accuracy/F1 comparison.

Runs entirely on CPU on purpose (see common.py) -- torch.quantization's dynamic quantization is a CPU feature; this isn't a limitation being
worked around, it's the correct environment for this technique.

Run this after choosing which model (ViT or ResNet) performed better in compare_results.py -- pass --model vit or --model resnet.
"""
import os
import sys
import time
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (VIT_MODEL_NAME, VIT_CHECKPOINT, RESNET_CHECKPOINT,
                     OUTPUTS_DIR, NUM_CLASSES, IMAGE_SIZE, QUANTIZATION_RESULTS)

import torch
from torchvision import models
from transformers import ViTForImageClassification

N_BENCHMARK_RUNS = 50


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def benchmark_latency(model, n_runs=N_BENCHMARK_RUNS):
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warm-up, not counted
            model(dummy_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy_input)
    elapsed = time.time() - start
    return (elapsed / n_runs) * 1000


def load_model(model_type):
    if model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            VIT_MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(VIT_CHECKPOINT, map_location="cpu"))
        checkpoint_path = VIT_CHECKPOINT
    else:
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location="cpu"))
        checkpoint_path = RESNET_CHECKPOINT
    model.eval()
    return model, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "resnet"], required=True)
    args = parser.parse_args()

    print(f"Loading trained {args.model} model...")
    model, checkpoint_path = load_model(args.model)

    print("Benchmarking original (unquantized) model...")
    original_latency_ms = benchmark_latency(model)
    original_size_mb = get_file_size_mb(checkpoint_path)
    print(f"  Original: {original_latency_ms:.2f} ms/image, {original_size_mb:.2f} MB")

    print("\nApplying INT8 dynamic quantization...")
    # Apple Silicon (ARM) needs the qnnpack backend explicitly set --
    # without this, quantize_dynamic fails with "Didn't find engine for
    # operation quantized::linear_prepack NoQEngine", since PyTorch
    # doesn't always auto-select the right backend on ARM.
    torch.backends.quantized.engine = "qnnpack"
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    quantized_path = os.path.join(OUTPUTS_DIR, f"{args.model}_quantized.pt")
    torch.save(quantized_model.state_dict(), quantized_path)

    print("Benchmarking quantized model...")
    quantized_latency_ms = benchmark_latency(quantized_model)
    quantized_size_mb = get_file_size_mb(quantized_path)
    print(f"  Quantized: {quantized_latency_ms:.2f} ms/image, {quantized_size_mb:.2f} MB")

    results = {
        "model": args.model,
        "original_latency_ms": original_latency_ms,
        "quantized_latency_ms": quantized_latency_ms,
        "latency_speedup_pct": (1 - quantized_latency_ms / original_latency_ms) * 100,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "size_reduction_pct": (1 - quantized_size_mb / original_size_mb) * 100,
    }

    print("\n" + "=" * 50)
    print(f"Latency:  {original_latency_ms:.2f} ms -> {quantized_latency_ms:.2f} ms "
          f"({results['latency_speedup_pct']:+.1f}%)")
    print(f"Size:     {original_size_mb:.2f} MB -> {quantized_size_mb:.2f} MB "
          f"({results['size_reduction_pct']:+.1f}%)")
    print("=" * 50)

    with open(QUANTIZATION_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {QUANTIZATION_RESULTS}")


if __name__ == "__main__":
    main()
