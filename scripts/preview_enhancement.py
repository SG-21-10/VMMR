"""
Utility — generate a before/after CLAHE+Retinex preview for the report/paper.

Usage:
    python3 preview_enhancement.py --image ../data/train/006_Maruti_Suzuki_Swift_4th_gen/0000.jpg
    python3 preview_enhancement.py --image <any_image_path> --out ../outputs
    python3 preview_enhancement.py --random_class 006_Maruti_Suzuki_Swift_4th_gen
"""
import argparse
import os
import random
import cv2
from preprocessing import clahe_enhance, single_scale_retinex


def enhance(img_bgr):
    clahe_img = clahe_enhance(img_bgr)
    retinex_img = single_scale_retinex(clahe_img, sigma=30.0)
    retinex_bgr = (
        cv2.cvtColor(retinex_img, cv2.COLOR_GRAY2BGR)
        if retinex_img.ndim == 2 else retinex_img
    )
    fused = cv2.addWeighted(clahe_img, 0.7, retinex_bgr, 0.3, 0)
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to a specific image")
    ap.add_argument("--random_class", default=None,
                     help="class folder name under ../data/train to pick a random image from")
    ap.add_argument("--data_root", default="../data/train")
    ap.add_argument("--out", default="../outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.image:
        img_path = args.image
    elif args.random_class:
        class_dir = os.path.join(args.data_root, args.random_class)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            raise SystemExit(f"no images found in {class_dir}")
        img_path = os.path.join(class_dir, random.choice(files))
    else:
        raise SystemExit("pass either --image <path> or --random_class <class_folder_name>")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise SystemExit(f"could not read image: {img_path}")

    fused = enhance(img_bgr)

    base = os.path.splitext(os.path.basename(img_path))[0]
    before_path = os.path.join(args.out, f"{base}_before.jpg")
    after_path = os.path.join(args.out, f"{base}_after.jpg")

    cv2.imwrite(before_path, img_bgr)
    cv2.imwrite(after_path, fused)

    print(f"source image: {img_path}")
    print(f"before -> {before_path}")
    print(f"after  -> {after_path}")


if __name__ == "__main__":
    main()
