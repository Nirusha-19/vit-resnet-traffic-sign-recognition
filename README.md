# 🚦 ViT vs. ResNet-50: Robustness and Edge Quantization for Traffic Sign Recognition

---

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![License](https://img.shields.io/badge/License-MIT-green) ![Dataset](https://img.shields.io/badge/Dataset-GTSRB-orange)

---

Fine-tunes and compares a Vision Transformer (ViT) and a ResNet-50 CNN on German traffic sign classification (43 real sign types, genuinely imbalanced), evaluates both under simulated real world degraded conditions, shows what each model focuses on using the technique appropriate to each architecture, and quantizes both models to measure the real cost and benefit of edge deployment.

## 🎯 What Is This?

In a real deployment, recognizing a road sign has to happen instantly, on the vehicle's own onboard hardware. There's no "close enough" when a wrong answer could mean acting on the wrong speed limit or missing a stop sign. This project trains and rigorously evaluates two different architectures for exactly this task, going beyond a single accuracy number to test what actually matters for real deployment: how each model performs under degraded conditions, what it's actually looking at when it makes a decision, and what quantization actually costs or saves for each architecture.

## 🔍 What It Does

- Fine-tunes a Vision Transformer (`google/vit-base-patch16-224`) and a ResNet-50 CNN on identical data, using focal loss to handle genuine class imbalance across 43 sign types
- Evaluates both models on a held out test set with macro averaged precision, recall, and F1, metrics that reveal performance on rare sign types, not just common ones
- Tests both models under simulated blur, low light, and noise, measuring how much accuracy drops from the clean baseline under each condition
- Shows what each model focuses on using the technique appropriate to each architecture: Grad CAM for the CNN, attention map visualization for the transformer
- Quantizes both models to INT8 and measures the real latency and file size cost of that optimization on each architecture

## 📊 Dataset

**GTSRB** (German Traffic Sign Recognition Benchmark), created by researchers at Ruhr Universität Bochum for a competition at IJCNN 2011.

**Important note on which release this uses:** GTSRB has two official releases. A smaller IJCNN 2011 competition version (26,640 train / 12,569 test) exists, but this project uses the larger, later official release: **39,209 training images, 12,630 test images**, across 43 sign classes.

- **License:** CC BY. Freely usable with attribution, no click through terms, no registration required.
- **Download:** fully automated by `data/prepare_data.py`, which downloads and extracts the official archives directly and organizes the flat test set into class subfolders using its ground truth CSV
- **Imbalance:** genuine and confirmed. The most common class has 2,250 training images, the rarest has 210, a 10.7x imbalance ratio.

## 🖥️ How It Works, End to End

**Data split:** The 39,209 training images are split 85/15, stratified by class, into 33,328 for training and 5,881 for validation. The official, separate 12,630 image test set is used only once, for final evaluation.

**Training:** Both models are trained for 5 epochs using **focal loss** (not plain class weighting), a technique that down weights predictions the model already gets right easily, concentrating training effort on the harder, rarer classes, with class weights folded in on top.

**Checkpoint selection:** Validation macro F1 is tracked across epochs, and the final evaluation uses the checkpoint from whichever epoch scored highest (epoch 3 for ViT, epoch 5 for ResNet-50).

**Evaluation:** Both models are tested on the identical, untouched 12,630 image test set, reporting overall accuracy, macro averaged precision, recall, and F1, and each model's single worst performing class.

**Robustness testing:** Both trained models are re evaluated on the same test set after applying synthetic blur, low light, and noise corruptions, measuring the accuracy drop from clean baseline under each condition.

**Explainability:** Grad CAM heatmaps are generated for ResNet, a convolutional technique, and attention weight visualizations are generated for ViT, the correct approach for transformers, since Grad CAM has no convolutional layer to hook into on a ViT.

**Quantization:** Both models are quantized to INT8 and benchmarked for latency and file size, before and after.

## 📈 Model Performance

| Metric | ViT | ResNet-50 |
|---|---|---|
| Overall accuracy (Top 1) | 98.27% | **98.71%** |
| Macro precision | 97.69% | **98.06%** |
| Macro recall | 98.30% | **98.62%** |
| Macro F1 | 97.91% | **98.27%** |
| Worst class | class 22, 85.83% recall | class 22, **87.50%** recall |

Both models share the same single hardest sign type (class 22), suggesting that class is intrinsically difficult rather than a weakness specific to one architecture. On clean condition metrics alone, ResNet-50 is the stronger model.

## 🌧️ Robustness Under Degraded Conditions

Both models were re tested on the full test set after applying synthetic blur, low light, and noise corruptions. This tests conditions harsher than GTSRB's own natural variation covers, closer to genuinely adverse real world driving conditions.

| Condition | ViT accuracy | ViT drop | ResNet accuracy | ResNet drop |
|---|---|---|---|---|
| Clean | 98.27% | — | 98.71% | — |
| Blur | 98.22% | 0.05 pts | 98.80% | ~flat |
| Low light | 95.34% | 2.93 pts | 98.12% | 0.59 pts |
| **Noise** | **91.30%** | **6.97 pts** | **43.47%** | **55.24 pts** |

ResNet is slightly more robust under blur and low light, and it wins on every clean condition metric. But under noise, ResNet's accuracy collapses to near unusable levels, while ViT remains largely reliable.

## 🔬 Explainability

Grad CAM heatmaps (ResNet) and attention weight visualizations (ViT) were generated on real test images, saved to `outputs/heatmaps/`. In most samples, both models correctly focus on the actual sign shape rather than background clutter, a real, checkable sanity test confirming the models learned to recognize signs rather than some unrelated visual artifact. Not every sample produces an equally crisp result. A small number of low quality source images produce less clearly interpretable attention maps, which is a limitation of the underlying image quality rather than the technique itself.

## ⚙️ Edge Quantization

Both models were quantized to INT8 using `torch.quantization.quantize_dynamic`, benchmarked before and after:

| Model | Metric | Original | Quantized | Change |
|---|---|---|---|---|
| **ViT** | File size | 327.49 MB | 84.45 MB | **74.2% smaller** |
| **ViT** | Latency | 55.22 ms/image | 95.44 ms/image | **72.8% slower** |
| **ResNet-50** | File size | 90.31 MB | 90.06 MB | 0.3% smaller (negligible) |
| **ResNet-50** | Latency | 20.33 ms/image | 19.03 ms/image | 6.4% faster |

The same technique affected each architecture differently, for a technically explainable reason. PyTorch's dynamic quantization only converts `Linear` layers. ViT's parameters live predominantly in linear layers, so quantization substantially shrank its file size, but the runtime overhead of converting between full precision and INT8 on every forward pass outweighed the compute savings, making it slower overall. ResNet-50's parameters live predominantly in convolutional layers, which this technique doesn't touch, so its file size and latency both stayed close to unchanged.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Framework | PyTorch · torchvision |
| Base Models | `google/vit-base-patch16-224` (HuggingFace `transformers`) · ResNet-50 (`torchvision`) |
| Loss | Focal loss, custom implementation, gamma of 2.0 |
| Explainability | `pytorch-grad-cam` (ResNet) · custom attention extraction (ViT) |
| Optimization | `torch.quantization`, INT8 dynamic quantization |
| Evaluation | `scikit-learn` |
| Data handling | `pandas` |
| Language | Python 3.11 |

## Why CPU Throughout

Every script explicitly runs on CPU. Two reasons: `torch.quantization`'s dynamic INT8 quantization is a CPU focused feature requiring an explicitly set backend engine (`qnnpack` on ARM), and saving a model while on an MPS device embeds device specific info in the checkpoint, which then fails to load on machines without MPS. Training on CPU here avoids that problem entirely.

## 📁 Project Structure

```
vit-edge-optimization/
├── README.md
├── requirements.txt
├── common.py                       ← shared constants, paths, GTSRB download URL
├── data/
│   └── prepare_data.py             ← downloads GTSRB, organizes test set into class folders
├── scripts/
│   ├── train_utils.py              ← data loading, focal loss, evaluation
│   ├── train_vit.py                ← fine tunes ViT with checkpoint selection
│   ├── train_resnet.py             ← fine tunes ResNet-50 with checkpoint selection
│   ├── compare_results.py          ← prints the ViT vs. ResNet comparison table
│   ├── robustness_test.py          ← tests both models under blur, low light, noise
│   ├── explainability.py           ← generates Grad CAM and attention map heatmaps
│   └── quantize_and_benchmark.py   ← INT8 quantization, latency/size benchmark
└── outputs/                        ← generated locally when you run the scripts (weights, results JSON, heatmaps); not tracked
```

## 🚀 Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the dataset (fully automated download):
```bash
python data/prepare_data.py
```

Train both models:
```bash
python scripts/train_vit.py
python scripts/train_resnet.py
```

Compare them:
```bash
python scripts/compare_results.py
```

Test robustness:
```bash
python scripts/robustness_test.py
```

Generate explainability visualizations:
```bash
python scripts/explainability.py
```

Quantize and benchmark each model:
```bash
python scripts/quantize_and_benchmark.py --model vit
python scripts/quantize_and_benchmark.py --model resnet
```

## 💡 What Makes This Different

Goes beyond a single accuracy number, testing robustness under degraded conditions and quantization behavior for both architectures rather than reporting one clean condition result.

ResNet won on every clean condition metric, but a 55 point accuracy collapse under noise reveals it as the less reliable choice under real conditions.

Quantization affects each architecture differently. It shrinks ViT significantly but slows it down, while barely touching ResNet-50 at all. Both results are reported transparently, with the technical reason explained above.

## 🔮 Future Work

- Investigate whether a hardware native quantization toolchain (for example Apple's Core ML on Apple Silicon, or NVIDIA's TensorRT on CUDA GPUs) avoids the latency regression seen with `torch.quantization`'s general purpose backend for ViT
- Determine why class 22 is the hardest class for both architectures
- Test additional corruption types (JPEG compression artifacts, motion blur specifically simulating vehicle speed)

## 👩‍💻 Author

Nirusha Mantralaya Ramesh

🔗 GitHub: Nirusha-19

## 📄 License

MIT. Free to use, fork, and build upon.