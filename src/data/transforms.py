"""
Image preprocessing and light online augmentation for IVF embryo datasets.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torchvision import transforms

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


class ConditionalCLAHE:
    """
    Apply CLAHE only when grayscale mean < threshold.
    Operates on a PIL image and returns a PIL image.
    """

    def __init__(self, threshold: float = 60.0, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.threshold = threshold
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        if cv2 is None:
            print("Warning: OpenCV not available; CLAHE will be skipped.")

    def __call__(self, img):
        if cv2 is None:
            return img
        import numpy as np

        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mean_intensity = float(gray.mean())
        if mean_intensity >= self.threshold:
            return img

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        eq = clahe.apply(gray)
        eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        return transforms.functional.to_pil_image(eq_rgb)


def build_train_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    use_clahe: bool = False,
    clahe_threshold: float = 60.0,
) -> transforms.Compose:
    aug: list[Callable] = []
    if use_clahe:
        aug.append(ConditionalCLAHE(threshold=clahe_threshold))

    aug.extend(
        [
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(aug)


def build_eval_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    use_clahe: bool = False,
    clahe_threshold: float = 60.0,
) -> transforms.Compose:
    ops: list[Callable] = []
    if use_clahe:
        ops.append(ConditionalCLAHE(threshold=clahe_threshold))
    ops.extend(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)


def augmentation_summary() -> str:
    return (
        "Images were resized to 224×224 and normalized using mean/std from the public train split. "
        "During training, light online augmentations (±15° rotation, horizontal/vertical flips, mild brightness/contrast "
        "jitter, and occasional Gaussian blur) were applied; evaluation used resizing and normalization only."
    )
