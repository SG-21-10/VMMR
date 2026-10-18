# Indian VMMR — Attention-Based Explainable Vehicle Make & Model Recognition

Extension of DeepCar 5.0 (Amirkhani & Barshooi, 2023), built to match your
Review-1 proposal, sized to what one person/team can actually collect and
train on a Mac before the deadline.

## What's really different from the slides (read this first)

Your slides propose 5 extensions: Grad-CAM explainability, CLAHE+Retinex
enhancement, lightweight backbones, Indian make/model coverage, jointly
trained YOLO+agents. All five are here — but scaled to be buildable without
a research lab's dataset:

| Proposal item | What's implemented |
|---|---|
| 4 hand-labelled ROI agents, 4 heavy backbones | 2 agents (global view + an attention-discovered region), 1 shared lightweight backbone (MobileNetV3-Small / EfficientNet-B0) — no manual bounding-box labels needed |
| Blackboard fusion | Learnable weighted fusion of the two agents' softmax outputs |
| Grad-CAM | Implemented, hooked to the shared backbone's last conv block |
| CLAHE + Retinex | Implemented (`preprocessing.py`) |
| Joint YOLO training | A **pretrained** COCO YOLOv8 crops the vehicle before classification (no training needed — `car`/`truck` are already COCO classes). Not jointly trained with the agents — that needs far more data/time than a course project allows |
| Indian dataset (40k images, 480 classes) | You have 167 classes with metadata (`indian_cars.csv`), **zero images**. See the data plan below — this is the actual bottleneck, be realistic about it |

Say this scoping explicitly in your paper's Methodology section — reviewers
respect "here's what we simplified and why" far more than an unsupported
claim of full parity with a published IEEE-TITS paper.

## The real constraint: you have no images yet

167 classes × even 100 images/class = ~17,000 images to scrape, clean, and
verify by eye. That is the single biggest risk to your deadline, not the
model code. Two honest paths:

**Path A — pilot dataset (recommended given grading deadline).**
Pick ~25–30 classes: the `Current` status models from your 5–6 highest-volume
makes (Maruti Suzuki, Hyundai, Tata, Mahindra, Kia — check with
`df[df.status=='Current'].make.value_counts()`). Scrape ~120 images/class
(~3,000 raw images, ~1,800–2,000 after cleaning). This is trainable to a
genuinely good accuracy on a MacBook in a few hours, and is exactly the
scale of a defensible course-project dataset.

**Path B — full 167 classes.** Only attempt this if you have >1 week and can
split scraping + manual pruning across your 3-person team. Expect ~60–70%
of scraped images to be usable after `02_clean_dataset.py`.

Either way, **report the "effective dataset"** you ended up with (N classes,
M images, images/class distribution) exactly like the paper's Fig. 2 — that
table is expected by any reviewer of a VMMR paper.

## Setup (Mac)

```bash
cd vmmr_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

If you're on Apple Silicon (M1/M2/M3/M4), PyTorch will automatically use
the `mps` backend (`train.py` and `predict.py` auto-detect it) — training
a MobileNetV3-Small on ~2,000–20,000 images is very feasible locally.
On an Intel Mac, everything still runs on CPU, just slower — for the full
167-class Path B on Intel, consider training on Google Colab's free GPU
instead (copy the `scripts/` folder + `data/` there) and bring the
checkpoint back for the demo app.

## Pipeline, step by step

```bash
# 1. Scrape (run locally — sandboxes/CI can't reach image search engines)
python scripts/01_scrape_images.py --csv indian_cars.csv --out data/raw \
    --per_class 120 --classes_subset 30   # drop --classes_subset for full run

# 2. Clean + split
python scripts/02_clean_dataset.py --src data/raw --dst data --min_per_class 40

# 3. Train
python scripts/train.py --data data --epochs 40 --backbone mobilenet_v3_small

# 4. Evaluate (produces figures/tables for your Results section)
python scripts/evaluate.py --data data --ckpt checkpoints/best.pt

# 5a. Predict on one photo (CLI)
python scripts/predict.py --image test_photo.jpg --ckpt checkpoints/best.pt

# 5b. Or launch the interactive demo (user uploads a photo in the browser)
python scripts/app.py --ckpt checkpoints/best.pt
```

## Mapping outputs to your journal sections

- **Dataset section** → `data/` class counts + a Fig.-2-style bar chart (bar
  chart of images per class — quick to make with `pandas`/`matplotlib` from
  `outputs/classes.txt` and folder counts).
- **Methodology** → `model.py`'s docstring table above, plus `preprocessing.py`
  for the enhancement stage and `predict.py`'s `try_yolo_crop` for the
  acquisition stage.
- **Results** → `outputs/accuracy_loss_curves.png` (cf. paper Fig. 9),
  `outputs/confusion_matrix.png`, `outputs/classification_report.csv`,
  and the printed top-1/top-5 accuracy (cf. paper Table IV/VI) — **do not**
  present these as beating 96.72%/92.14%; your class count and dataset size
  are different, so frame it as "comparable methodology, smaller-scale
  validation," which is a completely normal and accepted framing for a
  student paper extending a larger published benchmark.
- **Explainability results (your novel G1 contribution)** → several
  `predict.py` Grad-CAM overlays, ideally one where the heatmap visibly
  sits on the headlight/grille, discussed against DeepCar 5.0's claim that
  those regions carry the most class information.
- **Night/occlusion subset gain (G2)** → run `evaluate.py` twice: once with
  `enhance(mode="off")` and once with `mode="auto"` (edit `dataset.py`
  temporarily) on a manually curated low-light subset of your test set, and
  report the accuracy delta.
- **Edge deployment (G3)** → report parameter counts:
  `sum(p.numel() for p in model.parameters())` for your MobileNetV3 model
  vs. citing the paper's ResNet101+VGG19+Xception+DenseNet201 parameter sum,
  as your "≥60% fewer parameters" claim.

## Honesty notes for your write-up (these strengthen, not weaken, the paper)

1. Two agents + shared backbone, not four agents + four backbones — a
   data/compute-driven design choice, stated as a limitation/future work.
2. Web-scraped images, not manually verified with bounding boxes like
   DeepCar 5.0's dataset — exactly the dataset-quality gap the base paper
   itself criticises in prior work (their Table III). Note it as a known
   limitation of your extension, and as future work (manual annotation,
   or a small verified subset).
3. YOLO vehicle-cropping uses pretrained COCO weights, not a jointly
   trained detector — still closes practical value (garbage backgrounds
   removed before classification) without needing detector training data.
