"""
Prints the ViT vs. ResNet comparison table.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import VIT_RESULTS, RESNET_RESULTS


def load_results(path):
    if not os.path.exists(path):
        print(f"Missing {path} -- run the corresponding training script first.")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    vit = load_results(VIT_RESULTS)
    resnet = load_results(RESNET_RESULTS)
    if vit is None or resnet is None:
        return

    print("=" * 65)
    print(f"{'Metric':<30}{'ViT':>15}{'ResNet-50':>18}")
    print("-" * 65)
    print(f"{'Overall accuracy (Top-1)':<30}{vit['accuracy']:>14.2%} {resnet['accuracy']:>17.2%}")
    print(f"{'Macro precision':<30}{vit['macro_precision']:>14.2%} {resnet['macro_precision']:>17.2%}")
    print(f"{'Macro recall':<30}{vit['macro_recall']:>14.2%} {resnet['macro_recall']:>17.2%}")
    print(f"{'Macro F1':<30}{vit['macro_f1']:>14.2%} {resnet['macro_f1']:>17.2%}")
    print("-" * 65)
    print(f"{'Worst class (ViT)':<30}{'class ' + str(vit['worst_class']):>15}")
    print(f"{'  its recall':<30}{vit['worst_class_recall']:>14.2%}")
    print(f"{'Worst class (ResNet)':<30}{'class ' + str(resnet['worst_class']):>15}")
    print(f"{'  its recall':<30}{resnet['worst_class_recall']:>14.2%}")
    print("-" * 65)
    print("DIAGNOSTIC ONLY -- not a deployment/safety metric, see note below")
    print(f"{'Top-3 accuracy':<30}{vit['top3_accuracy_DIAGNOSTIC_ONLY']:>14.2%} "
          f"{resnet['top3_accuracy_DIAGNOSTIC_ONLY']:>17.2%}")
    print(f"{'Top-5 accuracy':<30}{vit['top5_accuracy_DIAGNOSTIC_ONLY']:>14.2%} "
          f"{resnet['top5_accuracy_DIAGNOSTIC_ONLY']:>17.2%}")
    print("=" * 65)
    print("\nNote: macro-averaged metrics weight every one of the 43 "
          "classes equally, regardless of how many examples each has --")
    print("this is what actually reveals whether the model is doing well "
          "on rare sign types, not just the common ones.")
    print("\nTop-K accuracy only shows whether the correct sign was among "
          "the model's top few guesses -- it does NOT reflect real-world")
    print("reliability. In a vehicle, only the single top-1 guess is ever "
          "acted on, so Top-1 accuracy above is the only number that")
    print("matters for deployment. Top-K is included purely to help "
          "diagnose whether the model's mistakes are near-misses (e.g.")
    print("confusing similar speed limits) or wildly wrong (confusing a "
          "speed limit with a stop sign).")


if __name__ == "__main__":
    main()
