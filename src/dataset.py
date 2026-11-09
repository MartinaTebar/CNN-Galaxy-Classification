"""
dataset.py
----------
The Galaxy10Dataset class and the transform pipelines.

MENTAL MODEL
  A torch Dataset answers exactly two questions:
    len(dataset)      -> how many samples are there?
    dataset[i]        -> give me sample i as (tensor, label)
  That's the whole interface. It knows nothing about batches, shuffling,
  epochs, or the model. It is a *formatting adapter*: it turns one raw
  numpy row into one correctly-shaped, correctly-typed torch tensor.

  Batching and shuffling are the DataLoader's job, not the Dataset's.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Human-readable class names, in label order (0..9).
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]


class Galaxy10Dataset(Dataset):
    """
    Parameters
    ----------
    images : np.ndarray, shape (N, 256, 256, 3), dtype uint8
        Keep it as uint8. Do NOT pre-convert to float32.
    labels : np.ndarray, shape (N,), integer class indices 0..9
    transform : callable or None
        A torchvision transform applied to each image tensor.
    """

    def __init__(self, images, labels, transform=None):
        assert len(images) == len(labels), "images and labels must align"
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # --- 1. grab one raw image -----------------------------------
        # shape (256, 256, 3), dtype uint8. .copy() is here because if
        # `self.images` is a memory-mapped array, torch.from_numpy on a
        # non-writable buffer warns; copying one image is cheap.
        img = np.asarray(self.images[idx]).copy()

        # --- 2. numpy -> torch, and fix the axis order ---------------
        # numpy / matplotlib use HWC (height, width, channels).
        # PyTorch conv layers expect CHW (channels, height, width).
        # permute() just reorders axes; it does not change any pixel value.
        img = torch.from_numpy(img).permute(2, 0, 1)   # (3, 256, 256) uint8

        # --- 3. resize / cast to float / normalise -------------------
        # This is where the uint8 -> float32 conversion happens, for ONE
        # image at a time. See the transform pipelines below.
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# --------------------------------------------------------------------
# TRANSFORM PIPELINES
# --------------------------------------------------------------------
# transforms.Compose just chains callables: output of one feeds the next.
#
# Two notes that trip people up:
#
# 1. Do NOT use transforms.ToTensor() here. ToTensor() expects a PIL image
#    or an HWC numpy array; we already have a CHW uint8 tensor. The right
#    tool is ConvertImageDtype, which casts uint8 -> float32 AND divides
#    by 255 so pixels land in [0, 1].
#
# 2. Normalize does  (x - mean) / std  per channel. Networks train much
#    better when inputs are roughly zero-mean and unit-variance.
#      - For SimpleCNN (trained from scratch) mean=std=0.5 maps [0,1] to
#        [-1,1]. Fine as a starting point.
#      - For ResNet18 you MUST use the ImageNet statistics, because the
#        pretrained weights were learned on data normalised that way.
#        Feeding differently-scaled inputs throws away much of the benefit.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CROP  = 140

def build_transforms(resolution: int, pretrained: bool, train: bool):
    """
    resolution : 140 for SimpleCNN (the 140 px crop, no resize), 224 for ResNet18
    pretrained : True -> ImageNet normalisation, False -> [-1, 1]
    train      : True -> include random augmentation
    """
    # steps = []
    steps = [
        transforms.CenterCrop(CROP),
        transforms.Resize((resolution, resolution), antialias=True),
    ]

    # --- augmentation: TRAIN SPLIT ONLY ------------------------------
    # Never augment val/test. Those must be a fixed, deterministic
    # yardstick, otherwise your metric wobbles for no reason.
    #
    # Galaxies are a lovely case for flips and rotations: a spiral galaxy
    # photographed upside-down is still the same spiral galaxy. There is no
    # canonical "up" in the sky. So these augmentations create genuinely
    # valid new training examples and are one of the cheapest accuracy wins
    # available to you.
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ]

    steps += [
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.ConvertImageDtype(torch.float32),   # uint8 -> float [0,1]
        transforms.Normalize(
            mean=IMAGENET_MEAN if pretrained else [0.5, 0.5, 0.5],
            std=IMAGENET_STD if pretrained else [0.5, 0.5, 0.5],
        ),
    ]
    return transforms.Compose(steps)


def load_split(split: str, data_dir=None, mmap=True):
    """
    Load one saved split. mmap=True keeps RAM usage low.

    The default path is resolved relative to THIS FILE, not the current
    working directory. That matters because "data/splits" only works if you
    happen to be sitting in the repo root - a notebook launched from
    anywhere else silently gets a FileNotFoundError. Anchoring to __file__
    makes it work from a terminal, a notebook, or a subdirectory alike.
    """
    from pathlib import Path

    if data_dir is None:
        # src/dataset.py -> src/ -> repo root -> data/splits
        d = Path(__file__).resolve().parent.parent / "data" / "splits"
    else:
        d = Path(data_dir)

    img_path = d / f"{split}_images.npy"
    if not img_path.exists():
        raise FileNotFoundError(
            f"{img_path} not found.\n"
            f"Run `python src/prepare_data.py` from the repo root first, "
            f"or pass data_dir= explicitly."
        )

    mode = "r" if mmap else None
    images = np.load(img_path, mmap_mode=mode)
    labels = np.load(d / f"{split}_labels.npy")
    return images, labels