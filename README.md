# 🌿 Plant Disease Classification — EfficientNet-B3 Pipeline

A complete deep learning pipeline for classifying plant leaf diseases into 39 categories using EfficientNet-B3 with two-phase fine-tuning, CutMix/MixUp augmentation, and Test-Time Augmentation (TTA).

## Results

| Metric | Score |
|--------|-------|
| Validation Accuracy (single fold) | **99.54%** |
| Model | EfficientNet-B3 (ImageNet pretrained) |
| Training Time | ~20 min per fold on RTX PRO 6000 |
| Inference Time | ~5 min with TTA (10,976 images) |

## Dataset

- **Source:** [PlantVillage Dataset](https://huggingface.co/datasets/SmellsLikeAISpirit/plant-disease-train)
- **Train:** 43,729 images across 39 classes (14 crop species)
- **Test:** 10,976 images
- **Challenge:** Imbalanced classes (121 to 4,405 images per class) + "other" class

### Crops Covered

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. Some minority classes have as few as 121 samples (Potato healthy), while majority classes have 4,000+ (Orange Huanglongbing).

## Approach

### Architecture

**EfficientNet-B3** was chosen over MobileNetV2 based on published benchmarks:

| Model | PlantVillage Accuracy |
|-------|----------------------|
| MobileNetV2 | 93.7–97.3% |
| **EfficientNet-B3** | **99.5–99.8%** |
| EfficientNet-B4/B5 | 99.9% |

EfficientNet-B3 hits the sweet spot between accuracy and VRAM usage — fits comfortably on 8GB+ GPUs at batch size 32.

### Two-Phase Fine-Tuning

This was the single most impactful technique. A Stanford CS231N study showed it nearly doubles accuracy vs naive end-to-end training.

**Phase 1 — Classifier Head Only (5 epochs, lr=1e-3)**
- Freeze the entire pretrained backbone
- Train only the new classifier layer
- The model learns to map ImageNet features → 39 plant disease classes
- Fast convergence: reaches ~95% val accuracy

**Phase 2 — Full Fine-Tuning (up to 15 epochs, lr=2e-5)**
- Unfreeze all layers
- Much lower learning rate to avoid destroying pretrained features
- Cosine annealing LR schedule with warm restarts
- CutMix + MixUp augmentation kicks in
- Early stopping (patience=7) prevents overfitting
- This phase pushes accuracy from 95% → 99%+

### Data Augmentation

Training augmentations are aggressive on geometry but gentle on color — because **color is diagnostic** for plant diseases (yellowing = nutrient deficiency, brown spots = fungal infection, rust color = rust disease).

**Applied during training:**
- Random resize crop (scale 0.8–1.0)
- Horizontal & vertical flip
- Random rotation (±20°)
- Small translations (10%)
- Gentle color jitter (brightness/contrast ±10%, minimal hue shift)
- Gaussian blur
- Random erasing (cutout-style, 20% probability)

**Applied during Phase 2 only:**
- **CutMix** (p=0.5): Cuts a random patch from one image and pastes it onto another. Forces the model to recognize diseases from partial leaf views. Reported +1.2% accuracy on leaf disease benchmarks.
- **MixUp** (p=0.3): Blends two images with a random weight. Provides feature-space regularization.

**Why train accuracy is lower than val accuracy:** CutMix and MixUp make training deliberately harder (the model classifies blended/cut images). At validation time, clean unmodified images are used, so the model performs at its true capability.

### Handling Class Imbalance

- **WeightedRandomSampler:** Oversamples minority classes so every class appears roughly equally in each batch
- **Label Smoothing (0.1):** Prevents overconfident predictions, especially important for the ambiguous "other" class
- **Cross-Entropy Loss:** With label smoothing, outperformed focal loss on PlantVillage in published comparisons

### Test-Time Augmentation (TTA)

At inference, each test image is predicted on 7 variants:
- Original
- Horizontal flip
- Vertical flip
- 90°, 180°, 270° rotations

Softmax probabilities are averaged across all variants. This exploits the fact that leaf disease symptoms are orientation-invariant. Typically adds **0.5–1.5% accuracy**.

### Fold Ensemble

Stratified K-Fold cross-validation (5 folds, training 2 for speed). Each fold trains on 80% of data and validates on 20%. At inference, softmax probabilities from all fold models are averaged. This reduces variance and typically adds **0.3–1% accuracy**.

## Project Structure

```
├── plant_disease_training.ipynb   # Main notebook (19 cells, fully commented)
├── train_plant_disease.py         # Standalone training script
├── generate_submission.py         # Quick inference with existing checkpoints
├── test_predictions.py            # Inspect predictions on train/test samples
├── README.md                      # This file
├── data/                          # Auto-downloaded datasets (not in repo)
│   ├── plant-disease-train/       # 43,729 training images
│   └── plant-disease-test/        # 10,976 test images
└── outputs/                       # Generated during training
    ├── best_fold0.pth             # Best model checkpoint (fold 0)
    ├── best_fold1.pth             # Best model checkpoint (fold 1)
    ├── label_mapping.json         # Class name ↔ index mapping
    └── submission.csv             # Final predictions
```

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

Upload `plant_disease_training.ipynb` to RunPod / Colab / Kaggle and run cells sequentially.

### Option 2: Command Line

```bash
# Install dependencies
pip install timm ttach datasets huggingface_hub scikit-learn tqdm

# Run full pipeline (download → train → inference → predictions)
python train_plant_disease.py

# Or just generate predictions from existing checkpoints
python generate_submission.py
```

## Configuration

All hyperparameters are in the `CFG` class. Key settings to tweak:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_name` | `efficientnet_b3` | Also try: `mobilenetv2_120d`, `tf_efficientnet_b4` |
| `img_size` | 300 | Native resolution for EfficientNet-B3 |
| `batch_size` | 64 | Reduce to 32 for GPUs with < 16GB VRAM |
| `phase2_epochs` | 15 | Max epochs (early stopping usually triggers at ~8-10) |
| `phase2_lr` | 2e-5 | Lower = safer fine-tuning, higher = faster convergence |
| `train_folds` | [0, 1] | Train more folds for better ensemble (up to [0,1,2,3,4]) |
| `use_tta` | True | Disable for ~7x faster inference (costs ~0.5-1% accuracy) |
| `cutmix_prob` | 0.5 | Set to 0 to disable CutMix |
| `mixup_prob` | 0.3 | Set to 0 to disable MixUp |

## Hardware Requirements

| Setup | Batch Size | Time per Fold | Total (2 folds) |
|-------|-----------|---------------|------------------|
| RTX PRO 6000 (48GB) | 64 | ~15-20 min | ~40 min |
| RTX 3090/4090 (24GB) | 64 | ~20-25 min | ~50 min |
| RTX 3060 (12GB) | 32 | ~35-45 min | ~1.5 hr |
| MacBook M4 Pro (24GB) | 32 | ~2.5-3.5 hr | ~6 hr |
| CPU only (inference) | 1 | N/A | ~20 min |

## Key Techniques Summary

| Technique | Impact | How |
|-----------|--------|-----|
| EfficientNet-B3 over MobileNetV2 | +2-5% | Better architecture for accuracy |
| Two-phase fine-tuning | +2-4% | Freeze → unfreeze prevents catastrophic forgetting |
| CutMix + MixUp | +1-2% | Regularization through input-space augmentation |
| WeightedRandomSampler | +1-2% | Handles class imbalance (121 vs 4,405 samples) |
| Label smoothing (0.1) | +0.3-0.5% | Prevents overconfident predictions |
| TTA (7 augmentations) | +0.5-1.5% | Flip + rotate at test time, average predictions |
| Fold ensemble (2 folds) | +0.3-1% | Average softmax probabilities across models |

## Known Dataset Issues

- **Capture bias:** Background pixels alone can yield 49% accuracy due to camera/lighting artifacts ([Noyan, 2022](https://arxiv.org/abs/2206.04374))
- **Data leakage:** Multiple photos per physical leaf — train/test splits should respect leaf grouping
- **"other" class:** 300 heterogeneous out-of-distribution images — needs aggressive oversampling
- **Git LFS:** Dataset uses Git LFS for image storage — the pipeline auto-detects LFS pointers and pulls real images

## References

- Mohanty, Hughes & Salathé (2016). [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419). Frontiers in Plant Science.
- Noyan (2022). [Uncovering Bias in the PlantVillage Dataset](https://arxiv.org/abs/2206.04374). arXiv.
- Tan & Le (2019). [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946). ICML.
- Yun et al. (2019). [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899). ICCV.
- Zhang et al. (2018). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR.

## License

This project is for educational and research purposes. The PlantVillage dataset is publicly available for research use.
