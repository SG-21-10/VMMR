"""
Stage 0 (v2) — Dataset acquisition via DuckDuckGo image search.

Replaces 01_scrape_images.py after discovering icrawler's BingImageCrawler
was silently returning unrelated content (likely a stale HTML parser no
longer matching Bing's current page structure). DuckDuckGo results were
manually spot-checked and confirmed to be genuine, relevant car photos.

Usage:
    python 01b_scrape_images_ddg.py --csv ../indian_cars_pilot.csv --out ../data/raw --per_class 120
"""
import argparse
import os
import time
import random
import requests
import pandas as pd
from ddgs import DDGS
from PIL import Image
from io import BytesIO


def build_query(row: pd.Series) -> str:
    return f"{row['make']} {row['model']} {row['generation']} {row['year_from']} front view car"


def download_image(url: str, dest_path: str, min_size=(200, 200), timeout=10) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return False
        img = Image.open(BytesIO(resp.content))
        if img.width < min_size[0] or img.height < min_size[1]:
            return False
        img.convert("RGB").save(dest_path, "JPEG")
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../indian_cars_pilot.csv")
    ap.add_argument("--out", default="../data/raw")
    ap.add_argument("--per_class", type=int, default=120)
    ap.add_argument("--classes_subset", type=int, default=None)
    ap.add_argument("--status_filter", choices=["all", "Current", "Discontinued"], default="all")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.status_filter != "all":
        df = df[df["status"] == args.status_filter]
    if args.classes_subset:
        df = df.head(args.classes_subset)

    os.makedirs(args.out, exist_ok=True)
    print(f"Scraping {len(df)} classes, up to {args.per_class} images each -> {args.out}")

    for _, row in df.iterrows():
        class_dir = os.path.join(
            args.out, f"{int(row['label_index']):03d}_{row['make']}_{row['model']}_{row['generation']}"
        ).replace(" ", "_")

        if os.path.isdir(class_dir) and len(os.listdir(class_dir)) >= args.per_class * 0.5:
            print(f"skip (already has images): {class_dir}")
            continue

        os.makedirs(class_dir, exist_ok=True)
        query = build_query(row)
        print(f"-> {query}")

        results = []
        for attempt in range(5):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(query, max_results=args.per_class))
                break
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"   search failed (attempt {attempt+1}/5): {e} -- waiting {wait}s")
                time.sleep(wait)
        if not results:
            print("   giving up on this class after 5 attempts")
            continue

        saved = 0
        for i, r in enumerate(results):
            url = r.get("image")
            if not url:
                continue
            dest = os.path.join(class_dir, f"{saved:04d}.jpg")
            if download_image(url, dest):
                saved += 1
            if saved >= args.per_class:
                break

        print(f"   saved {saved}/{len(results)} for this class")
        time.sleep(random.uniform(6, 12))  # be polite between classes

    print("Done. Next: python 02_clean_dataset.py")


if __name__ == "__main__":
    main()
