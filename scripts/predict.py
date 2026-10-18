"""
User input, single image -> prediction + evidence.

Usage:
    python predict.py --image path/to/photo.jpg --ckpt checkpoints/best.pt

Pipeline: (optional) YOLO vehicle crop -> CLAHE/Retinex enhancement ->
VMMRNet -> top-3 make/model + confidence -> Grad-CAM overlay saved next
to the input image as <name>_gradcam.jpg.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from dataset import eval_tf
from gradcam import GradCAM, overlay_heatmap
from model import build_model
from preprocessing import enhance
from train import get_device


def try_yolo_crop(img_bgr: np.ndarray):
    """Best-effort vehicle localisation with a pretrained (COCO) YOLO — no
    training needed since 'car'/'truck' are COCO classes. Falls back to the
    full frame if ultralytics isn't installed or nothing is detected.
    Closes part of gap G3 (Stage 1: Acquire) without needing a custom
    per-part detector trained from scratch."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model.predict(img_bgr, classes=[2, 3, 5, 7], verbose=False)  # car/moto/bus/truck
        boxes = results[0].boxes
        if len(boxes) == 0:
            return img_bgr
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best = boxes.xyxy[areas.argmax()].cpu().numpy().astype(int)
        x1, y1, x2, y2 = best
        pad = int(0.05 * max(x2 - x1, y2 - y1))
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return img_bgr[y1:y2, x1:x2]
    except Exception as e:
        print(f"(YOLO crop skipped: {e})")
        return img_bgr


def predict(image_path: str, ckpt_path: str, use_yolo: bool = True, topk: int = 3):
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes), backbone=ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_bgr = cv2.imread(image_path)
    if use_yolo:
        img_bgr = try_yolo_crop(img_bgr)
    img_bgr = enhance(img_bgr, mode="auto")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = eval_tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(dim=1)[0]
    top_probs, top_idx = probs.topk(min(topk, len(classes)))

    predictions = [(classes[i], float(p)) for p, i in zip(top_probs, top_idx)]

    # Grad-CAM evidence for the #1 prediction
    cam_model = build_model(len(classes), backbone=ckpt["backbone"]).to(device)
    cam_model.load_state_dict(ckpt["model"])
    cam_model.eval()
    gradcam = GradCAM(cam_model, cam_model.target_layer())
    tensor.requires_grad_(True)
    cam, class_idx, conf = gradcam(tensor, class_idx=int(top_idx[0]))
    overlay = overlay_heatmap(img_bgr, cam)

    out_path = os.path.splitext(image_path)[0] + "_gradcam.jpg"
    cv2.imwrite(out_path, overlay)

    return predictions, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--no_yolo", action="store_true")
    args = ap.parse_args()

    preds, gradcam_path = predict(args.image, args.ckpt, use_yolo=not args.no_yolo)
    print("\nTop predictions:")
    for i, (name, conf) in enumerate(preds, 1):
        print(f"  {i}. {name}  —  {conf*100:.1f}%")
    print(f"\nEvidence heatmap saved to: {gradcam_path}")


if __name__ == "__main__":
    main()
