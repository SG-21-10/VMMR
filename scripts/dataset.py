"""
Stage 1 dataset loader.

Folder layout expected (produced by 02_clean_dataset.py):
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg
    data/test/<class_name>/*.jpg

Augmentations mirror the DeepCar 5.0 list (rotation up to 30 deg, horizontal
flip about a vertical axis, affine/perspective warps, contrast jitter,
zoom, blur, noise) so your results are directly comparable in kind, even
though your compute/data budget is smaller.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from preprocessing import enhance

IMG_SIZE = 224

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), shear=8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),  # ~ "dropout of regions"
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EnhancedImageFolder(ImageFolder):
    """ImageFolder + CLAHE/Retinex enhancement applied before the torchvision transform,
    so night / low-light frames get restored contrast (closes gap G2)."""

    def __init__(self, root, transform=None, enhance_mode="auto"):
        super().__init__(root, transform=None)
        self.tv_transform = transform
        self.enhance_mode = enhance_mode

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # fallback for unusual encodings PIL can read but cv2 can't
            from PIL import Image
            img_pil = Image.open(path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        img_bgr = enhance(img_bgr, mode=self.enhance_mode)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        from PIL import Image
        img = Image.fromarray(img_rgb)
        if self.tv_transform:
            img = self.tv_transform(img)
        return img, target


def get_datasets(data_root: str, enhance_mode: str = "auto"):
    train_ds = EnhancedImageFolder(f"{data_root}/train", transform=train_tf, enhance_mode=enhance_mode)
    val_ds = EnhancedImageFolder(f"{data_root}/val", transform=eval_tf, enhance_mode=enhance_mode)
    test_ds = EnhancedImageFolder(f"{data_root}/test", transform=eval_tf, enhance_mode=enhance_mode)
    assert train_ds.classes == val_ds.classes == test_ds.classes, "class mismatch across splits"
    return train_ds, val_ds, test_ds
