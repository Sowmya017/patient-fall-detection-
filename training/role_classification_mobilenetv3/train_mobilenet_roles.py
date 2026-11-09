import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


ROLE_ORDER = ["medical_staff", "other", "patient"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(cfg):
    img_size = int(cfg["train"]["img_size"])
    aug = cfg["augment"]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=tuple(aug["random_crop_scale"])),
            transforms.RandomHorizontalFlip(p=float(aug["hflip"])),
            transforms.RandomRotation(degrees=float(aug["rotation_deg"])),
            transforms.RandomAffine(
                degrees=0,
                translate=(float(aug["translate"]), float(aug["translate"])),
            ),
            transforms.ColorJitter(
                brightness=float(aug["color_jitter"]),
                contrast=float(aug["color_jitter"]),
                saturation=float(aug["color_jitter"]),
                hue=min(0.1, float(aug["color_jitter"]) / 2.0),
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=float(aug["random_erasing"]),
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, val_tf


def make_model(cfg):
    name = cfg["train"]["model"]
    pretrained = bool(cfg["train"]["pretrained"])
    num_classes = int(cfg["train"]["num_classes"])

    if name != "mobilenet_v3_large":
        raise ValueError(f"Unknown model: {name}")

    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def make_weighted_sampler(dataset):
    targets = np.array(dataset.targets)
    class_sample_count = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_sample_count, 1.0)
    sample_weights = torch.from_numpy(class_weights[targets]).double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights(dataset, device):
    targets = np.array(dataset.targets)
    counts = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return {
        "val_acc": accuracy_score(all_labels, all_preds),
        "val_macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "val_weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "labels": all_labels,
        "preds": all_preds,
    }


def save_confusion_matrix(labels, preds, class_names, out_path, normalize=False):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sums, 1), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", color="black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_checkpoint(path, model, dataset, epoch, cfg):
    idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    torch.save(
        {
            "model": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "idx_to_class": idx_to_class,
            "epoch": epoch,
            "cfg": cfg,
        },
        path,
    )


def train(cfg, train_dir, val_dir, out_dir):
    set_seed(int(cfg["project"]["seed"]))
    device = "cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms(cfg)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    if train_ds.classes != ROLE_ORDER:
        print(f"Warning: expected class folders {ROLE_ORDER}, found {train_ds.classes}")

    sampler = make_weighted_sampler(train_ds) if cfg["train"]["use_weighted_sampler"] else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg["project"]["num_workers"]),
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["project"]["num_workers"]),
        pin_memory=device == "cuda",
    )

    model = make_model(cfg).to(device)
    class_weights = compute_class_weights(train_ds, device) if cfg["train"]["use_class_weights"] else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(cfg["train"]["label_smoothing"]),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"]["epochs"]),
        eta_min=float(cfg["train"]["min_lr"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device == "cuda")

    best_f1 = -1.0
    best_state = None
    history = []
    patience = int(cfg["train"]["early_stopping_patience"])
    stale_epochs = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        start = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"]) and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / max(len(train_ds), 1)
        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": metrics["val_acc"],
            "val_macro_f1": metrics["val_macro_f1"],
            "val_weighted_f1": metrics["val_weighted_f1"],
            "lr": scheduler.get_last_lr()[0],
            "time_sec": time.time() - start,
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        if metrics["val_macro_f1"] > best_f1:
            best_f1 = metrics["val_macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(out_dir / "best.pt", model, train_ds, epoch, cfg)
            save_confusion_matrix(metrics["labels"], metrics["preds"], train_ds.classes, out_dir / "confusion_matrix.png")
            save_confusion_matrix(
                metrics["labels"],
                metrics["preds"],
                train_ds.classes,
                out_dir / "confusion_matrix_normalized.png",
                normalize=True,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(out_dir / "last.pt", model, train_ds, epoch, cfg)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if stale_epochs >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 role classifier.")
    parser.add_argument("--config", default="config_used.yaml", help="Training config YAML path.")
    parser.add_argument(
        "--train-dir",
        required=True,
        help="ImageFolder train directory with class folders medical_staff/other/patient.",
    )
    parser.add_argument(
        "--val-dir",
        required=True,
        help="ImageFolder validation directory with class folders medical_staff/other/patient.",
    )
    parser.add_argument("--out-dir", default="runs/mobilenet_role", help="Output run directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, Path(args.train_dir), Path(args.val_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
