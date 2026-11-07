"""
prepare_data.py
---------------
One-off script: read the Galaxy10 HDF5 file, make a stratified
train/val/test split, and save the pieces to disk as .npy files.

Run this ONCE, then never touch the HDF5 file again during training.

    python src/prepare_data.py

WHY SAVE TO .npy?
  - Reloading a 3.5 GB HDF5 and re-splitting at the top of every notebook
    session is slow and, if you don't fix the random seed, gives you a
    DIFFERENT split each time. That silently leaks test data into training
    across sessions.
  - .npy files can be memory-mapped (np.load(..., mmap_mode='r')), so the
    OS pages images in from disk on demand instead of holding 3.5 GB in RAM.

WHY uint8, NOT float32?
  17736 * 256 * 256 * 3 bytes
    = 3.5 GB as uint8
    = 13.9 GB as float32   <-- this is what your notebook currently does
  The conversion to float happens per-image inside the Dataset, so at any
  moment only one batch (~32 images) exists in float form.
"""

from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- config
H5_PATH = Path("data/Galaxy10_DECals_NoDuplicated.h5")
OUT_DIR = Path("data/splits")
SEED = 42          # fixed seed => the same split every time you run this
VAL_FRAC = 0.10    # fraction of the FULL dataset
TEST_FRAC = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ load
    with h5py.File(H5_PATH, "r") as f:
        images = np.array(f["images"])   # (N, 256, 256, 3) uint8
        labels = np.array(f["ans"])      # (N,) int

    # int64 is what torch's CrossEntropyLoss wants for class indices.
    # NOTE: no to_categorical / one-hot here. PyTorch does not want one-hot.
    labels = labels.astype(np.int64)

    print(f"loaded {images.shape[0]} images, dtype={images.dtype}")
    print("class counts:", np.bincount(labels))

    # ----------------------------------------------------------- split
    # We split INDICES rather than the big arrays, so we only make one
    # copy of the image data at the very end.
    #
    # stratify=labels tells sklearn to preserve the class proportions in
    # each split. Galaxy10 is imbalanced (the rarest class has a few
    # hundred images), so without this your test set can end up with
    # almost none of a rare class and the accuracy for it is meaningless.
    idx = np.arange(len(labels))

    # Step 1: peel off the test set.
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_FRAC,
        stratify=labels,
        random_state=SEED,
    )

    # Step 2: peel the validation set off what remains.
    # VAL_FRAC is a fraction of the FULL set, but train_test_split works on
    # the subset it's given, so we rescale: 0.10 / 0.90 = 0.111...
    val_share_of_remainder = VAL_FRAC / (1.0 - TEST_FRAC)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_share_of_remainder,
        stratify=labels[train_val_idx],
        random_state=SEED,
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # ------------------------------------------------------------ save
    for name, split_idx in splits.items():
        # Sorting the indices makes the fancy-indexing read sequential,
        # which is noticeably faster and keeps peak memory lower.
        split_idx = np.sort(split_idx)
        np.save(OUT_DIR / f"{name}_images.npy", images[split_idx])  # stays uint8
        np.save(OUT_DIR / f"{name}_labels.npy", labels[split_idx])
        print(
            f"{name:5s}: {len(split_idx):6d} images | "
            f"class counts {np.bincount(labels[split_idx], minlength=10)}"
        )

    print(f"\nsaved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()