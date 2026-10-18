"""
Stage 0b — Clean the scraped dataset and split train/val/test.

- Drops corrupt / unreadable / too-small images
- Removes near-duplicates (perceptual hash) which web crawlers produce a lot of
- Enforces a minimum images-per-class threshold (drops classes below it,
  logs them so you can note "effective class count" honestly in the paper)
- Splits each surviving class 70/15/15 into data/train, data/val, data/test

Usage:
    python 02_clean_dataset.py --src data/raw --dst data --min_per_class 40
"""
import argparse
import hashlib
import os
import random
import shutil

from PIL import Image


def phash(img: Image.Image, hash_size: int = 8) -> str:
    img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--min_per_class", type=int, default=40,
                     help="classes with fewer surviving images are dropped")
    ap.add_argument("--min_side", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    dropped_classes, kept_classes = [], []
    total_kept, total_seen = 0, 0

    for class_name in sorted(os.listdir(args.src)):
        class_dir = os.path.join(args.src, class_name)
        if not os.path.isdir(class_dir):
            continue

        seen_hashes = set()
        good_files = []
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            total_seen += 1
            try:
                with Image.open(fpath) as im:
                    im.verify()
                with Image.open(fpath) as im:
                    im = im.convert("RGB")
                    if min(im.size) < args.min_side:
                        continue
                    h = phash(im)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    good_files.append(fpath)
            except Exception:
                continue

        if len(good_files) < args.min_per_class:
            dropped_classes.append((class_name, len(good_files)))
            continue

        kept_classes.append(class_name)
        total_kept += len(good_files)

        random.shuffle(good_files)
        n = len(good_files)
        splits = {
            "train": good_files[: int(0.70 * n)],
            "val": good_files[int(0.70 * n): int(0.85 * n)],
            "test": good_files[int(0.85 * n):],
        }
        for split, files in splits.items():
            out_dir = os.path.join(args.dst, split, class_name)
            os.makedirs(out_dir, exist_ok=True)
            for f in files:
                shutil.copy(f, os.path.join(out_dir, os.path.basename(f)))

    print(f"\nSeen {total_seen} raw files.")
    print(f"Kept {len(kept_classes)} classes, {total_kept} clean images.")
    print(f"Dropped {len(dropped_classes)} classes for having < {args.min_per_class} clean images:")
    for name, n in dropped_classes:
        print(f"  - {name}: {n}")
    print("\nReport this 'effective class count' in your Dataset section — "
          "it's exactly the kind of honest limitation DeepCar 5.0 itself discusses "
          "(their Table III comparison of dataset quality).")


if __name__ == "__main__":
    main()
