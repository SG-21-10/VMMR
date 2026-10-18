"""
Stage 3 — training.

Usage:
    python train.py --data data --epochs 40 --backbone mobilenet_v3_small

Auto-selects Apple Silicon MPS if available, else CPU. Logs per-epoch
train/val accuracy + loss to outputs/train_log.csv so you can reproduce
a Fig.-9-style accuracy/loss curve for your paper.
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_datasets
from model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                     choices=["mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    train_ds, val_ds, _ = get_datasets(args.data)
    num_classes = len(train_ds.classes)
    print(f"{num_classes} classes, {len(train_ds)} train / {len(val_ds)} val images")

    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(train_ds.classes))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes, backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, patience_ctr = 0.0, 0
    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, val_loss, val_acc, dt])
        print(f"epoch {epoch:03d} | train_acc {tr_acc:.4f} loss {tr_loss:.4f} "
              f"| val_acc {val_acc:.4f} loss {val_loss:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "classes": train_ds.classes,
                        "backbone": args.backbone}, "checkpoints/best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    print(f"done. best val_acc={best_val_acc:.4f}. checkpoint: checkpoints/best.pt")


if __name__ == "__main__":
    main()
