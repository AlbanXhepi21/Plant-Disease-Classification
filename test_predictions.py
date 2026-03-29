#!/usr/bin/env python3
"""
Plant Disease — Test & Inspect Predictions
============================================
Run AFTER training is complete (needs model checkpoints in ./outputs/).

Usage:
    python test_predictions.py

What it does:
  1. Loads best model(s) from ./outputs/
  2. Predicts on random TEST images — shows filename, predicted class, confidence
  3. Predicts on random TRAIN images — compares prediction vs ground truth
  4. Shows lowest-confidence test predictions (potential errors)
  5. Saves a visual HTML report you can open in browser
"""

import os
import sys
import json
import random
import base64
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torchvision.transforms as T
import timm
import ttach as tta
from PIL import Image

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG — must match training config
# ============================================================================
class CFG:
    data_dir = "./data"
    output_dir = "./outputs"
    model_name = "efficientnet_b3"
    img_size = 300
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True
    num_test_samples = 20       # How many test images to preview
    num_train_samples = 20      # How many train images to verify
    use_tta = True


def get_val_transforms():
    return T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# LOAD MODEL + LABEL MAPPING
# ============================================================================
def load_label_mapping():
    mapping_path = Path(CFG.output_dir) / "label_mapping.json"
    if not mapping_path.exists():
        print(f"❌ {mapping_path} not found. Run training first!")
        sys.exit(1)

    with open(mapping_path) as f:
        data = json.load(f)

    label2idx = data["label2idx"]
    idx2label = {int(k): v for k, v in data["idx2label"].items()}
    print(f"✅ Loaded label mapping: {len(idx2label)} classes")
    return label2idx, idx2label


def load_model(fold=0):
    """Load a trained model checkpoint."""
    idx2label = load_label_mapping()[1]
    num_classes = len(idx2label)

    checkpoint_path = Path(CFG.output_dir) / f"best_fold{fold}.pth"
    if not checkpoint_path.exists():
        print(f"❌ {checkpoint_path} not found!")
        # Try to find any available fold
        for f in range(5):
            alt = Path(CFG.output_dir) / f"best_fold{f}.pth"
            if alt.exists():
                checkpoint_path = alt
                print(f"   Using {alt} instead")
                break
        else:
            print("❌ No model checkpoints found. Run training first!")
            sys.exit(1)

    model = timm.create_model(
        CFG.model_name, pretrained=False,
        num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.2,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=CFG.device))
    model = model.to(CFG.device)
    model.eval()

    print(f"✅ Loaded model from {checkpoint_path}")
    return model


def load_all_fold_models():
    """Load all available fold models for ensemble prediction."""
    idx2label = load_label_mapping()[1]
    num_classes = len(idx2label)
    models = []

    for fold in range(5):
        checkpoint_path = Path(CFG.output_dir) / f"best_fold{fold}.pth"
        if checkpoint_path.exists():
            model = timm.create_model(
                CFG.model_name, pretrained=False,
                num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.2,
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=CFG.device))
            model = model.to(CFG.device)
            model.eval()
            models.append((fold, model))

    print(f"✅ Loaded {len(models)} fold models: {[f for f, _ in models]}")
    return models


# ============================================================================
# PREDICTION
# ============================================================================
@torch.no_grad()
def predict_single(model, img_path, idx2label, use_tta=False):
    """Predict a single image. Returns top-5 predictions with confidence."""
    img = Image.open(img_path).convert("RGB")
    transform = get_val_transforms()
    img_tensor = transform(img).unsqueeze(0).to(CFG.device)

    if use_tta:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])
        tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode="mean")
        with autocast(enabled=CFG.mixed_precision):
            outputs = tta_model(img_tensor)
    else:
        with autocast(enabled=CFG.mixed_precision):
            outputs = model(img_tensor)

    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Top 5
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [(idx2label[i], float(probs[i])) for i in top5_idx]

    return top5, probs


@torch.no_grad()
def predict_ensemble(models, img_path, idx2label):
    """Predict using all fold models averaged."""
    transform = get_val_transforms()
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(CFG.device)

    all_probs = []
    for fold, model in models:
        if CFG.use_tta:
            transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ])
            tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode="mean")
            with autocast(enabled=CFG.mixed_precision):
                outputs = tta_model(img_tensor)
        else:
            with autocast(enabled=CFG.mixed_precision):
                outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    top5_idx = np.argsort(avg_probs)[::-1][:5]
    top5 = [(idx2label[i], float(avg_probs[i])) for i in top5_idx]
    return top5, avg_probs


# ============================================================================
# FIND IMAGES
# ============================================================================
def find_test_images():
    """Find test images in the repo."""
    base = Path(CFG.data_dir) / "plant-disease-test"
    test_images = []

    for d in [base / "test", base]:
        if not d.exists():
            continue
        for sub in ["images_0", "images_1"]:
            sub_dir = d / sub
            if sub_dir.exists():
                for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
                    test_images.extend(list(sub_dir.glob(ext)))
        if test_images:
            break
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
            test_images.extend(list(d.glob(ext)))
        if test_images:
            break

    if not test_images and base.exists():
        for ext in ["**/*.jpg", "**/*.JPG", "**/*.jpeg", "**/*.png"]:
            for p in base.glob(ext):
                if ".git" not in str(p):
                    test_images.append(p)

    return test_images


def find_train_images():
    """Find train images with their labels."""
    base = Path(CFG.data_dir) / "plant-disease-train"
    SKIP = {".git", ".github", "__pycache__", ".ipynb_checkpoints", ".hf", ".cache"}

    records = []
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir() or class_dir.name in SKIP:
            continue
        for img in class_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"} and img.is_file():
                # Verify it's a real image not LFS pointer
                try:
                    with open(img, "rb") as f:
                        h = f.read(30)
                    if not h.startswith(b"version https://git-lfs"):
                        records.append({"filepath": str(img), "label": class_dir.name})
                except Exception:
                    pass

    return records


# ============================================================================
# HTML REPORT
# ============================================================================
def img_to_base64(img_path, max_size=200):
    """Convert image to base64 for HTML embedding."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(test_results, train_results, output_path):
    """Generate an HTML report with image previews."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Plant Disease — Prediction Report</title>
<style>
    body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
    h1, h2 { color: #00d4aa; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
    .card { background: #16213e; border-radius: 12px; padding: 16px; border: 1px solid #2a2a4a; }
    .card img { width: 100%; border-radius: 8px; margin-bottom: 8px; }
    .pred { font-size: 14px; font-weight: bold; color: #00d4aa; }
    .conf { font-size: 13px; color: #a0a0c0; }
    .correct { border-left: 4px solid #00d4aa; }
    .wrong { border-left: 4px solid #ff6b6b; }
    .bar { height: 6px; background: #2a2a4a; border-radius: 3px; margin: 4px 0; }
    .bar-fill { height: 100%; border-radius: 3px; background: #00d4aa; }
    .top5 { font-size: 12px; color: #8888aa; margin-top: 8px; }
    .filename { font-size: 11px; color: #666; word-break: break-all; }
</style></head><body>
"""

    # Test predictions
    html += "<h1>🔮 Test Image Predictions</h1>\n"
    html += f"<p>Showing {len(test_results)} random test images with model predictions</p>\n"
    html += '<div class="grid">\n'

    for r in test_results:
        b64 = img_to_base64(r["filepath"])
        top1_class, top1_conf = r["top5"][0]
        html += f"""<div class="card">
    <img src="data:image/jpeg;base64,{b64}">
    <div class="pred">{top1_class}</div>
    <div class="bar"><div class="bar-fill" style="width:{top1_conf*100:.0f}%"></div></div>
    <div class="conf">Confidence: {top1_conf*100:.1f}%</div>
    <div class="top5">"""
        for cls, conf in r["top5"][1:3]:
            html += f"<br>  #{r['top5'].index((cls,conf))+1}: {cls} ({conf*100:.1f}%)"
        html += f'</div>\n<div class="filename">{r["filename"]}</div>\n</div>\n'

    html += "</div>\n"

    # Train verification
    html += "<h1>✅ Train Image Verification</h1>\n"
    html += f"<p>Showing {len(train_results)} random train images — comparing prediction vs ground truth</p>\n"

    correct = sum(1 for r in train_results if r["correct"])
    html += f"<p><b>{correct}/{len(train_results)}</b> correct ({100*correct/max(len(train_results),1):.0f}%)</p>\n"
    html += '<div class="grid">\n'

    for r in train_results:
        b64 = img_to_base64(r["filepath"])
        top1_class, top1_conf = r["top5"][0]
        css_class = "correct" if r["correct"] else "wrong"
        status = "✅" if r["correct"] else "❌"

        html += f"""<div class="card {css_class}">
    <img src="data:image/jpeg;base64,{b64}">
    <div class="pred">{status} Pred: {top1_class}</div>
    <div class="conf">True: {r['true_label']}</div>
    <div class="bar"><div class="bar-fill" style="width:{top1_conf*100:.0f}%"></div></div>
    <div class="conf">Confidence: {top1_conf*100:.1f}%</div>
</div>\n"""

    html += "</div>\n</body></html>"

    with open(output_path, "w") as f:
        f.write(html)

    print(f"📄 HTML report saved → {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("🔍 Plant Disease — Test & Inspect Predictions")
    print(f"   Device: {CFG.device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    _, idx2label = load_label_mapping()

    # Load models
    models = load_all_fold_models()
    if not models:
        print("❌ No models found. Run training first!")
        return

    use_ensemble = len(models) > 1
    single_model = models[0][1]  # Fallback to first fold

    # ========================================
    # 1. TEST IMAGE PREDICTIONS
    # ========================================
    print(f"\n{'='*60}")
    print("  TEST IMAGE PREDICTIONS")
    print(f"{'='*60}")

    test_images = find_test_images()
    print(f"📸 Found {len(test_images)} test images")

    if test_images:
        sample_test = random.sample(test_images, min(CFG.num_test_samples, len(test_images)))
        test_results = []

        for img_path in tqdm(sample_test, desc="Predicting test"):
            try:
                if use_ensemble:
                    top5, probs = predict_ensemble(models, img_path, idx2label)
                else:
                    top5, probs = predict_single(single_model, img_path, idx2label, use_tta=CFG.use_tta)

                test_results.append({
                    "filepath": str(img_path),
                    "filename": img_path.name,
                    "top5": top5,
                    "confidence": top5[0][1],
                })
            except Exception as e:
                print(f"   ⚠️ Skipped {img_path.name}: {e}")

        # Print results
        print(f"\n{'─'*80}")
        print(f"{'Filename':<25} {'Prediction':<50} {'Conf':>6}")
        print(f"{'─'*80}")
        for r in sorted(test_results, key=lambda x: -x["confidence"]):
            conf_bar = "█" * int(r["confidence"] * 20)
            print(f"{r['filename']:<25} {r['top5'][0][0]:<50} {r['confidence']*100:>5.1f}% {conf_bar}")

        # Low confidence alerts
        low_conf = [r for r in test_results if r["confidence"] < 0.5]
        if low_conf:
            print(f"\n⚠️  {len(low_conf)} low-confidence predictions (<50%):")
            for r in low_conf:
                print(f"   {r['filename']}: {r['top5'][0][0]} ({r['confidence']*100:.1f}%)")
                for cls, conf in r["top5"][1:3]:
                    print(f"      also could be: {cls} ({conf*100:.1f}%)")
    else:
        test_results = []
        print("⚠️ No test images found")

    # ========================================
    # 2. TRAIN IMAGE VERIFICATION
    # ========================================
    print(f"\n{'='*60}")
    print("  TRAIN IMAGE VERIFICATION")
    print(f"{'='*60}")

    train_records = find_train_images()
    print(f"📸 Found {len(train_records)} train images")

    if train_records:
        sample_train = random.sample(train_records, min(CFG.num_train_samples, len(train_records)))
        train_results = []

        for rec in tqdm(sample_train, desc="Verifying train"):
            try:
                if use_ensemble:
                    top5, probs = predict_ensemble(models, rec["filepath"], idx2label)
                else:
                    top5, probs = predict_single(single_model, rec["filepath"], idx2label, use_tta=CFG.use_tta)

                is_correct = top5[0][0] == rec["label"]
                train_results.append({
                    "filepath": rec["filepath"],
                    "true_label": rec["label"],
                    "top5": top5,
                    "confidence": top5[0][1],
                    "correct": is_correct,
                })
            except Exception as e:
                print(f"   ⚠️ Skipped: {e}")

        correct = sum(1 for r in train_results if r["correct"])
        wrong = [r for r in train_results if not r["correct"]]

        print(f"\n✅ {correct}/{len(train_results)} correct ({100*correct/len(train_results):.0f}%)")

        print(f"\n{'─'*100}")
        print(f"{'Status':<4} {'True Label':<45} {'Prediction':<45} {'Conf':>6}")
        print(f"{'─'*100}")
        for r in train_results:
            status = "✅" if r["correct"] else "❌"
            print(f" {status}  {r['true_label']:<45} {r['top5'][0][0]:<45} {r['confidence']*100:>5.1f}%")

        if wrong:
            print(f"\n❌ Misclassified ({len(wrong)}):")
            for r in wrong:
                print(f"   True: {r['true_label']}")
                print(f"   Pred: {r['top5'][0][0]} ({r['confidence']*100:.1f}%)")
                print(f"   Also: {r['top5'][1][0]} ({r['top5'][1][1]*100:.1f}%)")
                print()
    else:
        train_results = []
        print("⚠️ No train images found")

    # ========================================
    # 3. GENERATE HTML REPORT
    # ========================================
    if test_results or train_results:
        report_path = Path(CFG.output_dir) / "prediction_report.html"
        generate_html_report(test_results, train_results, report_path)
        print(f"\n💡 Open {report_path} in your browser to see visual results")

    # ========================================
    # 4. SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Models loaded: {len(models)} folds")
    print(f"  TTA: {'enabled' if CFG.use_tta else 'disabled'}")
    if test_results:
        confs = [r["confidence"] for r in test_results]
        print(f"  Test predictions: {len(test_results)} samples")
        print(f"  Test confidence: mean={np.mean(confs):.3f}  min={np.min(confs):.3f}  max={np.max(confs):.3f}")
    if train_results:
        print(f"  Train verification: {correct}/{len(train_results)} correct")


if __name__ == "__main__":
    main()
