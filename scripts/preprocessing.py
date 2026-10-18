"""
Stage 1 (part 2) — Low-light / glare enhancement.

Implements the "CLAHE + Retinex" stage from the proposal (closes gap G2:
degraded inputs are selected but never repaired). This is applied BEFORE
the image is handed to the classifier, on both training and inference.

Single-scale Retinex here is a lightweight, dependency-free approximation
(log-domain illumination removal) — good enough to demonstrate and measure
a night/glare accuracy gain, and honest to describe as "single-scale"
rather than claiming the full multi-scale Retinex from the literature.
"""
import cv2
import numpy as np


def single_scale_retinex(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    img = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log(img) - np.log(blur + 1.0)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def clahe_enhance(img_bgr: np.ndarray, clip_limit: float = 2.5, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def enhance(img_bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    """mode: 'auto' decides from mean brightness; 'always' forces enhancement; 'off' passthrough."""
    if mode == "off":
        return img_bgr

    mean_lum = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean()
    if mode == "auto" and mean_lum > 90:  # already well-lit, don't over-process
        return img_bgr

    clahe_img = clahe_enhance(img_bgr)
    retinex_img = single_scale_retinex(clahe_img, sigma=30.0)
    # blend: keep CLAHE colour fidelity, add retinex-recovered detail
    fused = cv2.addWeighted(clahe_img, 0.7, cv2.cvtColor(retinex_img, cv2.COLOR_GRAY2BGR)
                             if retinex_img.ndim == 2 else retinex_img, 0.3, 0)
    return fused
