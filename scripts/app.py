"""
Interactive demo — the "user input" deliverable.

Usage:
    python app.py --ckpt checkpoints/best.pt

Opens a local web page (Gradio) where a user uploads/drags a car photo and
gets back the top-3 make/model predictions plus the Grad-CAM evidence
overlay. Good for your viva demo and screenshots in the paper.
"""
import argparse
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from predict import predict

CKPT_PATH = "checkpoints/best.pt"


def infer(pil_image: Image.Image):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        preds, gradcam_path = predict(tmp.name, CKPT_PATH)
    overlay_rgb = cv2.cvtColor(cv2.imread(gradcam_path), cv2.COLOR_BGR2RGB)
    label_text = "\n".join(f"{i+1}. {name} — {conf*100:.1f}%" for i, (name, conf) in enumerate(preds))
    return label_text, overlay_rgb


def main():
    global CKPT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    args = ap.parse_args()
    CKPT_PATH = args.ckpt

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Image(type="pil", label="Upload a front-view car photo"),
        outputs=[gr.Textbox(label="Top-3 predictions"), gr.Image(label="Grad-CAM evidence")],
        title="VMMR — Indian Vehicle Make & Model Recognition",
        description="Attention-based, explainable extension of DeepCar 5.0.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
