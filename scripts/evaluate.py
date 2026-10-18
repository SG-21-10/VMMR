"""
Stage 4 — evaluation for your Results section.

Produces exactly the artifacts you need to write a DeepCar-style Results
section:
  - outputs/accuracy_loss_curves.png   (compare to paper's Fig. 9)
  - outputs/confusion_matrix.png
  - outputs/classification_report.csv  (per-class precision/recall/F1)
  - printed top-1 / top-5 accuracy      (compare to paper's Table IV/VI)

Usage:
    python evaluate.py --data data --ckpt checkpoints/best.pt
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import build_model
from train import get_device


def plot_curves(log_csv: str, out_path: str):
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df.epoch, df.train_acc, label="train")
    axes[0].plot(df.epoch, df.val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(df.epoch, df.train_loss, label="train")
    axes[1].plot(df.epoch, df.val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]

    _, _, test_ds = get_datasets(args.data)
    assert test_ds.classes == classes, "class list mismatch between checkpoint and data folder"
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model = build_model(len(classes), backbone=ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = logits.softmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    import numpy as np
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = all_probs.argmax(1)

    top1 = (all_preds == all_labels).mean()
    top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, len(classes)),
                                 labels=list(range(len(classes))))
    print(f"\nTest top-1 accuracy: {top1*100:.2f}%")
    print(f"Test top-5 accuracy: {top5*100:.2f}%")
    print("(Report both alongside DeepCar 5.0's manual 96.72% / automated 92.14% "
          "as context, NOT as a direct apples-to-apples comparison — different "
          "dataset size and class count. Say this explicitly in your paper.)")

    report = classification_report(all_labels, all_preds, target_names=classes,
                                    output_dict=True, zero_division=0)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.out, "classification_report.csv"))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(max(8, len(classes) * 0.25), max(6, len(classes) * 0.25)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150)
    print(f"saved {args.out}/confusion_matrix.png and classification_report.csv")

    log_csv = os.path.join(args.out, "train_log.csv")
    if os.path.exists(log_csv):
        plot_curves(log_csv, os.path.join(args.out, "accuracy_loss_curves.png"))


if __name__ == "__main__":
    main()
