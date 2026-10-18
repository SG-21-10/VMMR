"""
Stage 0 — Dataset acquisition.

Run this ON YOUR MAC (not in a sandbox) because it needs real internet
access. It reads indian_cars.csv and downloads front-view images for
every (make, model, generation) row using Bing image search.

Usage:
    python 01_scrape_images.py --csv indian_cars.csv --out data/raw --per_class 120

Notes / honesty check for your report:
- Web-scraped images are noisy (wrong angle, wrong generation, watermarks,
  interior shots). Budget real time for manual pruning — do NOT skip
  02_clean_dataset.py.
- 167 classes x 120 images is ~20k downloads. That will take hours and
  will hit rate limits. If you're short on time, use --classes_subset
  to scrape a smaller, still-defensible subset first (see README).
- This mirrors what the DeepCar 5.0 paper explicitly criticises about
  web-crawled datasets (Table III) — say so in your Limitations section,
  it's a legitimate, citable point, not a flaw to hide.
"""
import argparse
import os
import time
import pandas as pd
from icrawler.builtin import BingImageCrawler


def build_query(row: pd.Series) -> str:
    # Front three-quarter / front view queries recognise the same regions
    # (headlight, grille, bumper) that DeepCar 5.0's agents rely on.
    return f"{row['make']} {row['model']} {row['generation']} {row['year_from']} front view car"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="indian_cars.csv")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--per_class", type=int, default=120,
                     help="images requested per class (expect ~60-70% survive cleaning)")
    ap.add_argument("--classes_subset", type=int, default=None,
                     help="only scrape the first N rows of the csv (for a quick pilot run)")
    ap.add_argument("--status_filter", choices=["all", "Current", "Discontinued"], default="all")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.status_filter != "all":
        df = df[df["status"] == args.status_filter]
    if args.classes_subset:
        df = df.head(args.classes_subset)

    os.makedirs(args.out, exist_ok=True)
    print(f"Scraping {len(df)} classes, {args.per_class} images each -> {args.out}")

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
        try:
            crawler = BingImageCrawler(storage={"root_dir": class_dir}, log_level=50)
            crawler.crawl(keyword=query, max_num=args.per_class, min_size=(200, 200))
        except Exception as e:
            print(f"   failed: {e}")
        time.sleep(1)  # be polite, avoid hammering the search endpoint

    print("Done. Next: python 02_clean_dataset.py")


if __name__ == "__main__":
    main()
