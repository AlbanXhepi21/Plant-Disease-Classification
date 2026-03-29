#!/usr/bin/env python3
"""
Quick Submission Generator — Uses fold 0 model only
=====================================================
Run: python generate_submission.py
"""

import os
import sys
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import ttach as tta
from PIL import Image

warnings.filterwarnings("ignore")


class CFG:
    data_dir = "./data"
    output_dir = "./outputs"
    model_name = "efficientnet_b3"
    img_size = 300
    batch_size = 64
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True
    use_tta = True


def get_val_transforms():
    return T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TestDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG.img_size, CFG.img_size), (0, 0, 0))
        img = self.transform(img)
        return img, os.path.basename(fpath)


def main():
    print("🚀 Quick Submission Generator")
    print(f"   Device: {CFG.device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"   GPU: {torch.cuda.get_device_name(0)} ({vram/1e9:.1f} GB)")

    # --- Load label mapping ---
    print("\n📋 Loading label mapping...")
    mapping_path = Path(CFG.output_dir) / "label_mapping.json"
    with open(mapping_path) as f:
        data = json.load(f)
    idx2label = {int(k): v for k, v in data["idx2label"].items()}
    num_classes = len(idx2label)
    print(f"   {num_classes} classes")

    # --- Find all available fold models ---
    print("\n🧠 Loading model(s)...")
    models = []
    for fold in range(5):
        cp = Path(CFG.output_dir) / f"best_fold{fold}.pth"
        if cp.exists():
            model = timm.create_model(
                CFG.model_name, pretrained=False,
                num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.2,
            )
            model.load_state_dict(torch.load(cp, map_location=CFG.device))
            model = model.to(CFG.device)
            model.eval()
            models.append((fold, model))
            print(f"   ✅ Loaded fold {fold}")

    if not models:
        print("❌ No model checkpoints found in outputs/!")
        sys.exit(1)

    print(f"   Using {len(models)} model(s) for prediction")

    # --- Find test images ---
    print("\n📸 Finding test images...")
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

    print(f"   Found {len(test_images)} test images")

    # Check for LFS pointers in test images
    if test_images:
        lfs_count = 0
        for fp in test_images[:20]:
            try:
                with open(fp, "rb") as f:
                    h = f.read(30)
                if h.startswith(b"version https://git-lfs"):
                    lfs_count += 1
            except Exception:
                pass
        if lfs_count > 10:
            print("   ⚠️ Test images are LFS pointers. Pulling...")
            subprocess.run(["git", "lfs", "pull"], cwd=str(base), check=True)
            print("   ✅ LFS pull complete")

    # --- Find sample_submission.csv ---
    sample_df = None
    for sp in base.rglob("sample_submission.csv"):
        sample_df = pd.read_csv(sp)
        print(f"   📋 Loaded sample_submission.csv ({len(sample_df)} rows)")
        break

    # --- Run inference ---
    print(f"\n🔮 Running inference (TTA={'ON' if CFG.use_tta else 'OFF'})...")

    all_model_probs = []

    for fold, model in models:
        print(f"\n   Fold {fold}...")

        if CFG.use_tta:
            transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ])
            inference_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode="mean")
        else:
            inference_model = model

        inference_model.eval()
        ds = TestDataset(test_images, get_val_transforms())
        loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)

        fold_probs = []
        fold_fnames = []

        with torch.no_grad():
            for images, fnames in tqdm(loader, desc=f"   Fold {fold} predicting"):
                images = images.to(CFG.device, non_blocking=True)
                with autocast(enabled=CFG.mixed_precision):
                    outputs = inference_model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_probs.append(probs)
                fold_fnames.extend(fnames)

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_model_probs.append(fold_probs)
        print(f"   ✅ Fold {fold} done — {len(fold_fnames)} predictions")

    # --- Average across folds ---
    avg_probs = np.mean(all_model_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)

    print(f"\n📊 Building submission...")

    submission = pd.DataFrame({
        "id": fold_fnames,
        "label": [idx2label[p] for p in predictions],
    })

    # Use sample_submission as template
    if sample_df is not None:
        pred_map = dict(zip(submission["id"], submission["label"]))
        sample_df["label"] = sample_df["id"].map(pred_map)
        missing = sample_df["label"].isna().sum()
        if missing > 0:
            print(f"   ⚠️ {missing} test images not predicted — filling with 'other'")
            sample_df["label"] = sample_df["label"].fillna("other")
        submission = sample_df[["id", "label"]].copy()

    # --- Validate ---
    expected = 10976
    row_count = len(submission)
    cols = list(submission.columns)
    dupes = submission["id"].duplicated().sum()

    print(f"\n✅ Validation:")
    print(f"   Rows: {row_count} {'✅' if row_count == expected else '❌ EXPECTED ' + str(expected)}")
    print(f"   Columns: {cols} {'✅' if cols == ['id', 'label'] else '❌'}")
    print(f"   Duplicates: {dupes} {'✅' if dupes == 0 else '❌'}")
    print(f"   Classes used: {submission['label'].nunique()}")

    # --- Save ---
    save_path = Path(CFG.output_dir) / "submission.csv"
    submission.to_csv(save_path, index=False)
    print(f"\n💾 Saved → {save_path}")

    # --- Stats ---
    print(f"\n📊 Prediction distribution (top 10):")
    for cls, count in submission["label"].value_counts().head(10).items():
        print(f"   {cls}: {count}")

    max_probs = np.max(avg_probs, axis=1)
    print(f"\n🎯 Confidence:")
    print(f"   Mean:  {max_probs.mean():.3f}")
    print(f"   Min:   {max_probs.min():.3f}")
    print(f"   <50%:  {int((max_probs < 0.5).sum())} images")
    print(f"   <30%:  {int((max_probs < 0.3).sum())} images")
    print(f"   >99%:  {int((max_probs > 0.99).sum())} images")

    print(f"\n🎉 Done! Upload {save_path} to the competition page.")


if __name__ == "__main__":
    main()
