#!/usr/bin/env python3
"""
Plant Disease Classification — Competition Pipeline
=====================================================
Fully self-contained. Just run:
    python train_plant_disease.py

It will:
  1. Install missing dependencies
  2. Download train + test datasets via git clone
  3. Train 5-fold EfficientNet-B3 (two-phase fine-tuning)
  4. Run inference with TTA
  5. Output submission.csv ready for upload
"""

import os
import sys
import json
import random
import subprocess
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# ============================================================================
# 0. AUTO-INSTALL DEPENDENCIES
# ============================================================================
def ensure_installed(package, import_name=None):
    try:
        __import__(import_name or package)
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg, imp in [
    ("torch", "torch"), ("torchvision", "torchvision"), ("timm", "timm"),
    ("pandas", "pandas"), ("Pillow", "PIL"), ("tqdm", "tqdm"),
    ("scikit-learn", "sklearn"), ("ttach", "ttach"), ("huggingface_hub", "huggingface_hub"),
]:
    ensure_installed(pkg, imp)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
import timm
import ttach as tta
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG
# ============================================================================
class CFG:
    data_dir = "./data"
    output_dir = "./outputs"

    model_name = "efficientnet_b3"
    pretrained = True
    num_classes = 39

    img_size = 300
    batch_size = 64
    num_workers = 4

    phase1_epochs = 5
    phase1_lr = 1e-3

    phase2_epochs = 15
    phase2_lr = 2e-5
    weight_decay = 0.01

    label_smoothing = 0.1
    cutmix_prob = 0.5
    cutmix_alpha = 1.0
    mixup_prob = 0.3
    mixup_alpha = 0.4

    n_folds = 5
    train_folds = [0, 1]

    use_tta = True

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True
    early_stopping_patience = 7


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ============================================================================
# 1. DOWNLOAD DATA
# ============================================================================
def download_data():
    data_dir = Path(CFG.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing git-lfs...")
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "git-lfs"], capture_output=True)
        subprocess.run(["git", "lfs", "install"], check=True)

    train_repo = data_dir / "plant-disease-train"
    test_repo = data_dir / "plant-disease-test"

    if not train_repo.exists():
        print("📥 Cloning training dataset (may take a few minutes)...")
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/datasets/SmellsLikeAISpirit/plant-disease-train",
            str(train_repo)
        ], check=True)
        print("✅ Training data downloaded")
    else:
        print("✅ Training data already exists")

    if not test_repo.exists():
        print("📥 Cloning test dataset...")
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/datasets/SmellsLikeAISpirit/plant-disease-test",
            str(test_repo)
        ], check=True)
        print("✅ Test data downloaded")
    else:
        print("✅ Test data already exists")


# ============================================================================
# 2. DATA PREPARATION
# ============================================================================
def prepare_dataframe():
    SKIP_DIRS = {".git", ".github", "__pycache__", ".ipynb_checkpoints", ".hf", ".cache"}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    base = Path(CFG.data_dir) / "plant-disease-train"

    # Debug: print top-level repo contents
    print(f"\n🔍 Scanning repo: {base}")
    if base.exists():
        top_items = sorted(base.iterdir())
        dirs = [x.name for x in top_items if x.is_dir() and x.name not in SKIP_DIRS]
        files = [x.name for x in top_items if x.is_file()]
        print(f"   Directories: {dirs[:15]}{'...' if len(dirs) > 15 else ''}")
        print(f"   Files: {files[:10]}{'...' if len(files) > 10 else ''}")

    # Strategy 1: Use train_labels.csv if it exists
    for csv_path in [base / "train_labels.csv", base / "train" / "train_labels.csv"]:
        if csv_path.exists():
            print(f"   Found {csv_path}")
            labels_df = pd.read_csv(csv_path)
            print(f"   CSV columns: {list(labels_df.columns)}, rows: {len(labels_df)}")
            # Build filepath from CSV — find where the images actually live
            img_root = None
            sample_id = labels_df["id"].iloc[0]
            for candidate_root in [base / "train", base]:
                if not candidate_root.exists():
                    continue
                # Try finding the image by searching class subdirs
                for sub in candidate_root.iterdir():
                    if sub.is_dir() and sub.name not in SKIP_DIRS:
                        test_file = sub / sample_id
                        if test_file.exists():
                            img_root = candidate_root
                            break
                if img_root:
                    break
            if img_root:
                records = []
                for _, row in labels_df.iterrows():
                    fpath = img_root / row["label"] / row["id"]
                    if fpath.exists():
                        records.append({"filepath": str(fpath), "label": row["label"]})
                if records:
                    df = pd.DataFrame(records)
                    print(f"✅ Found {len(df)} training images via train_labels.csv")
                    _print_distribution(df)
                    return df

    # Strategy 2: Find directory containing class subdirectories with images
    candidates = [base / "train", base, Path(CFG.data_dir) / "train"]

    # Also add any non-hidden subdirectory of base that itself has subdirs
    if base.exists():
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name not in SKIP_DIRS and child not in candidates:
                candidates.append(child)

    for d in candidates:
        if not d.exists():
            continue
        records = []
        class_count = 0
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name in SKIP_DIRS:
                continue
            # Check if this subdirectory has images (making it a class directory)
            imgs = [f for f in sub.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.is_file()]
            if imgs:
                class_count += 1
                for img_path in imgs:
                    records.append({"filepath": str(img_path), "label": sub.name})

        if class_count >= 10:  # Expect at least 10 classes for PlantVillage
            df = pd.DataFrame(records)
            print(f"✅ Found {len(df)} training images across {class_count} classes in {d}")
            _print_distribution(df)
            return df
        elif class_count > 0:
            print(f"   ⚠️  {d} has only {class_count} class dirs, skipping...")

    # Nothing found — show full debug info
    print("❌ Could not find training images!")
    if base.exists():
        print("   Full repo structure (3 levels):")
        for item in sorted(base.iterdir()):
            if item.name in SKIP_DIRS:
                continue
            kind = "📁" if item.is_dir() else "📄"
            size = item.stat().st_size
            print(f"   {kind} {item.name} ({size} bytes)")
            if item.is_dir():
                for sub in sorted(list(item.iterdir())[:8]):
                    if sub.name in SKIP_DIRS:
                        continue
                    kind2 = "📁" if sub.is_dir() else "📄"
                    print(f"      {kind2} {sub.name}")
                    if sub.is_dir():
                        files_in = list(sub.iterdir())[:3]
                        for f in files_in:
                            print(f"         {f.name} ({f.stat().st_size} bytes)")
    raise FileNotFoundError("Cannot find training data. See debug output above.")


def _print_distribution(df):
    print("\n📊 Class distribution:")
    dist = df["label"].value_counts()
    for cls, count in dist.items():
        marker = "⚠️" if count < 300 else "  "
        print(f"  {marker} {cls}: {count}")


def build_label_mapping(df):
    classes = sorted(df["label"].unique())
    label2idx = {label: idx for idx, label in enumerate(classes)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    # Update num_classes dynamically
    CFG.num_classes = len(classes)
    print(f"\n🏷️  {len(classes)} classes mapped")

    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(Path(CFG.output_dir) / "label_mapping.json", "w") as f:
        json.dump({
            "label2idx": label2idx,
            "idx2label": {str(k): v for k, v in idx2label.items()},
        }, f, indent=2)

    return label2idx, idx2label


# ============================================================================
# 3. DATASET + AUGMENTATION
# ============================================================================
def get_train_transforms():
    return T.Compose([
        T.Resize((CFG.img_size + 32, CFG.img_size + 32)),
        T.RandomResizedCrop(CFG.img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(20),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.RandomGrayscale(p=0.02),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    return T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class PlantDiseaseDataset(Dataset):
    def __init__(self, df, label2idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["filepath"]).convert("RGB")
        except Exception:
            # If image is corrupted, return a random valid one instead
            return self.__getitem__(random.randint(0, len(self.df) - 1))
        label = self.label2idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


class PlantDiseaseTestDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            # Return a blank image if corrupted — will get low confidence
            img = Image.new("RGB", (CFG.img_size, CFG.img_size), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(fpath)


# ============================================================================
# 4. CUTMIX / MIXUP
# ============================================================================
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# 5. MODEL
# ============================================================================
def build_model():
    model = timm.create_model(
        CFG.model_name, pretrained=CFG.pretrained,
        num_classes=CFG.num_classes, drop_rate=0.3, drop_path_rate=0.2,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {CFG.model_name} | Params: {total:,}")
    return model.to(CFG.device)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name and "head" not in name:
            param.requires_grad = False
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"❄️  Backbone frozen. Trainable: {t:,}")


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Backbone unfrozen. Trainable: {t:,}")


# ============================================================================
# 6. TRAINING
# ============================================================================
def get_sampler(df_fold):
    class_counts = Counter(df_fold["label"].values)
    total = len(df_fold)
    weights = [total / (len(class_counts) * class_counts[df_fold.iloc[i]["label"]]) for i in range(len(df_fold))]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, phase="Phase 2"):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"  {phase} | Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        do_cutmix = phase == "Phase 2" and np.random.random() < CFG.cutmix_prob
        do_mixup = phase == "Phase 2" and not do_cutmix and np.random.random() < CFG.mixup_prob

        optimizer.zero_grad()
        with autocast(enabled=CFG.mixed_precision):
            if do_cutmix:
                images, ta, tb, lam = cutmix_data(images, labels, CFG.cutmix_alpha)
                loss = mixup_criterion(criterion, model(images), ta, tb, lam)
            elif do_mixup:
                images, ta, tb, lam = mixup_data(images, labels, CFG.mixup_alpha)
                loss = mixup_criterion(criterion, model(images), ta, tb, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        with torch.no_grad():
            if do_cutmix or do_mixup:
                outputs = model(images)
            _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)
        with autocast(enabled=CFG.mixed_precision):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def train_fold(fold, df_train, df_val, label2idx, idx2label):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}  |  Train: {len(df_train)}  |  Val: {len(df_val)}")
    print(f"{'='*60}")

    train_ds = PlantDiseaseDataset(df_train, label2idx, get_train_transforms())
    val_ds = PlantDiseaseDataset(df_val, label2idx, get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, sampler=get_sampler(df_train),
        num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size * 2, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True,
    )

    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
    scaler = GradScaler(enabled=CFG.mixed_precision)
    best_acc = 0.0
    patience = 0

    # ---- PHASE 1 ----
    print(f"\n📌 Phase 1: Head only ({CFG.phase1_epochs} epochs, lr={CFG.phase1_lr})")
    freeze_backbone(model)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.phase1_lr, weight_decay=CFG.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.phase1_epochs, eta_min=1e-6)

    for epoch in range(1, CFG.phase1_epochs + 1):
        train_one_epoch(model, train_loader, criterion, opt, scaler, epoch, "Phase 1")
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        sched.step()
        print(f"  → val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(CFG.output_dir) / f"best_fold{fold}.pth")

    # ---- PHASE 2 ----
    print(f"\n📌 Phase 2: Full fine-tune ({CFG.phase2_epochs} epochs, lr={CFG.phase2_lr})")
    unfreeze_backbone(model)
    opt = optim.AdamW(model.parameters(), lr=CFG.phase2_lr, weight_decay=CFG.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=CFG.phase2_epochs, T_mult=1, eta_min=1e-7)

    for epoch in range(1, CFG.phase2_epochs + 1):
        train_one_epoch(model, train_loader, criterion, opt, scaler, epoch, "Phase 2")
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion)
        sched.step(epoch)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  → val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), Path(CFG.output_dir) / f"best_fold{fold}.pth")
            print(f"  ✅ New best! acc={val_acc:.4f}")
        else:
            patience += 1
            if patience >= CFG.early_stopping_patience:
                print(f"  ⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n🏆 Fold {fold} best: {best_acc:.4f}")

    # Worst classes report
    model.load_state_dict(torch.load(Path(CFG.output_dir) / f"best_fold{fold}.pth"))
    _, _, preds, labels = validate(model, val_loader, criterion)
    report = classification_report(labels, preds, target_names=[idx2label[i] for i in range(CFG.num_classes)], output_dict=True)
    worst = sorted(
        [(n, m["f1-score"]) for n, m in report.items() if n in idx2label.values()],
        key=lambda x: x[1],
    )[:5]
    print("  ⚠️  5 worst classes:")
    for name, f1 in worst:
        print(f"     {name}: F1={f1:.3f}")

    del model
    torch.cuda.empty_cache()
    return best_acc


# ============================================================================
# 7. INFERENCE
# ============================================================================
def find_test_images():
    base = Path(CFG.data_dir) / "plant-disease-test"
    test_dirs = [base / "test", base, Path(CFG.data_dir) / "test"]

    test_images = []
    for d in test_dirs:
        if not d.exists():
            continue
        # Check images_0/, images_1/ subdirs
        for sub in ["images_0", "images_1"]:
            sub_dir = d / sub
            if sub_dir.exists():
                for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
                    test_images.extend(list(sub_dir.glob(ext)))
        if test_images:
            break
        # Check for images directly in this dir
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
            test_images.extend(list(d.glob(ext)))
        if test_images:
            break

    # Fallback: recursively find all images in the repo (skip .git)
    if not test_images and base.exists():
        print("   Scanning repo recursively for test images...")
        for ext in ["**/*.jpg", "**/*.JPG", "**/*.jpeg", "**/*.png"]:
            for p in base.glob(ext):
                if ".git" not in str(p):
                    test_images.append(p)

    print(f"📸 Found {len(test_images)} test images")

    # Check if test images are LFS pointers and pull if needed
    if test_images:
        lfs_test = 0
        for fp in test_images[:20]:
            try:
                with open(fp, "rb") as f:
                    h = f.read(30)
                if h.startswith(b"version https://git-lfs"):
                    lfs_test += 1
            except Exception:
                pass
        if lfs_test > 10:
            print("⚠️  Test images are LFS pointers. Running git lfs pull...")
            subprocess.run(["git", "lfs", "pull"], cwd=str(base), check=True)
            print("✅ Test LFS pull complete")

    # Find sample_submission.csv anywhere in the repo
    sample_df = None
    if base.exists():
        for sp in base.rglob("sample_submission.csv"):
            sample_df = pd.read_csv(sp)
            print(f"📋 Loaded {sp} ({len(sample_df)} rows)")
            break

    return test_images, sample_df


@torch.no_grad()
def predict_with_tta(model, test_images):
    if CFG.use_tta:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])
        model = tta.ClassificationTTAWrapper(model, transforms, merge_mode="mean")

    model.eval()
    ds = PlantDiseaseTestDataset(test_images, get_val_transforms())
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

    all_probs, all_fnames = [], []
    for images, fnames in tqdm(loader, desc="🔮 Predicting"):
        images = images.to(CFG.device, non_blocking=True)
        with autocast(enabled=CFG.mixed_precision):
            outputs = model(images)
        all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        all_fnames.extend(fnames)

    return all_fnames, np.concatenate(all_probs, axis=0)


def generate_submission(all_fold_probs, all_fnames, idx2label, sample_df=None):
    avg_probs = np.mean(all_fold_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)

    submission = pd.DataFrame({
        "id": all_fnames,
        "label": [idx2label[p] for p in predictions],
    })

    # Use sample_submission as template to guarantee correct IDs and order
    if sample_df is not None:
        pred_map = dict(zip(submission["id"], submission["label"]))
        sample_df["label"] = sample_df["id"].map(pred_map)
        missing = sample_df["label"].isna().sum()
        if missing > 0:
            print(f"⚠️  {missing} test images not predicted — filling with 'other'")
            sample_df["label"] = sample_df["label"].fillna("other")
        submission = sample_df[["id", "label"]].copy()

    # Validation checks
    expected_rows = 10976
    assert len(submission) == expected_rows, f"❌ Expected {expected_rows} rows, got {len(submission)}"
    assert list(submission.columns) == ["id", "label"], f"❌ Wrong columns: {list(submission.columns)}"
    assert submission["id"].duplicated().sum() == 0, "❌ Duplicate IDs!"

    save_path = Path(CFG.output_dir) / "submission.csv"
    submission.to_csv(save_path, index=False)

    print(f"\n✅ submission.csv saved → {save_path}")
    print(f"   Rows: {len(submission)}")
    print(f"   Classes: {submission['label'].nunique()}")
    print(f"\n📊 Top 10 predictions:")
    print(submission["label"].value_counts().head(10).to_string())

    max_probs = np.max(avg_probs, axis=1)
    print(f"\n🎯 Confidence: mean={max_probs.mean():.3f}  min={max_probs.min():.3f}  low(<0.5)={int((max_probs<0.5).sum())}")

    return submission


# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    print("🌿 Plant Disease Classification — Competition Pipeline")
    print(f"   Model: {CFG.model_name} | ImgSize: {CFG.img_size} | Batch: {CFG.batch_size}")
    print(f"   Folds: {CFG.train_folds} | Device: {CFG.device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"   GPU: {torch.cuda.get_device_name(0)} ({vram/1e9:.1f} GB)")
    print()

    seed_everything(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)

    download_data()

    df = prepare_dataframe()

    # Validate images — detect LFS pointers and pull if needed
    def count_valid_images(df):
        valid_mask = []
        bad = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Checking"):
            try:
                fpath = row["filepath"]
                with open(fpath, "rb") as f:
                    header = f.read(30)
                if header.startswith(b"version https://git-lfs"):
                    valid_mask.append(False)
                    bad += 1
                    continue
                img = Image.open(fpath)
                img.verify()
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
                bad += 1
        return valid_mask, bad

    print("\n🔍 Validating images...")
    valid_mask, bad_count = count_valid_images(df)

    # If too many bad files, force git lfs pull and re-validate
    if bad_count > len(df) * 0.1:
        print(f"\n⚠️  {bad_count}/{len(df)} files are invalid (likely Git LFS pointers)")
        print("   Running: git lfs pull (this will download ~2GB of images)...")
        repo_dir = Path(CFG.data_dir) / "plant-disease-train"
        subprocess.run(["git", "lfs", "pull"], cwd=str(repo_dir), check=True)
        print("✅ LFS pull complete. Re-validating...")
        valid_mask, bad_count = count_valid_images(df)

    df = df[valid_mask].reset_index(drop=True)
    print(f"✅ {len(df)} valid images ({bad_count} corrupted/skipped)")

    if bad_count > 1000:
        print(f"❌ Still too many bad files! Try manually:")
        print(f"   cd {Path(CFG.data_dir) / 'plant-disease-train'} && git lfs pull")
        sys.exit(1)

    label2idx, idx2label = build_label_mapping(df)

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    fold_accs = []
    for fold in CFG.train_folds:
        df_train = df[df["fold"] != fold].reset_index(drop=True)
        df_val = df[df["fold"] == fold].reset_index(drop=True)
        acc = train_fold(fold, df_train, df_val, label2idx, idx2label)
        fold_accs.append(acc)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Fold accs: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"  Mean CV:   {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"{'='*60}")

    test_images, sample_df = find_test_images()
    if not test_images:
        print("⚠️  No test images found! Skipping inference.")
        return

    all_fold_probs = []
    for fold in CFG.train_folds:
        print(f"\n🔮 Inference with fold {fold}...")
        model = build_model()
        model.load_state_dict(torch.load(Path(CFG.output_dir) / f"best_fold{fold}.pth"))
        fnames, probs = predict_with_tta(model, test_images)
        all_fold_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    generate_submission(all_fold_probs, fnames, idx2label, sample_df)

    print("\n🎉 Done! Upload outputs/submission.csv to the competition page.")


if __name__ == "__main__":
    main()
