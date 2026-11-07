"""
diagnose.py
-----------
Load the saved checkpoint and find out WHERE the errors are.

    python src/diagnose.py

WHY THIS EXISTS
  Overall accuracy is a single number hiding ten very different stories.
  On an imbalanced dataset like Galaxy10, a model can score 0.60 overall
  while being at 0.00 on the rarest class and confusing three spiral
  classes with each other. Those are completely different problems with
  completely different fixes, and you cannot tell them apart from one
  number.

  This runs on the VALIDATION split, not the test split. Keep the test
  split sealed until you have stopped changing the model.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Galaxy10Dataset, build_transforms, load_split, CLASS_NAMES
from model_simple import SimpleCNN
from model_resnet import build_resnet18

# RESOLUTION = 128
RESOLUTION = 224
BATCH_SIZE = 64
# CKPT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "simplecnn.pt"
CKPT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "resnet18.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the whole split and return (true_labels, predicted_labels)."""
    model.eval()
    all_true, all_pred = [], []

    for images, labels in loader:
        logits = model(images.to(device))
        all_pred.append(logits.argmax(dim=1).cpu())
        all_true.append(labels)

    return torch.cat(all_true).numpy(), torch.cat(all_pred).numpy()


def main() -> None:
    device = get_device()

    images, labels = load_split("val")
    ds = Galaxy10Dataset(
        images,
        labels,
        # train=False: NO augmentation. Evaluation must be deterministic.
        # transform=build_transforms(RESOLUTION, pretrained=False, train=False),
        transform=build_transforms(RESOLUTION, pretrained=True, train=False),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
    model = build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    y_true, y_pred = collect_predictions(model, loader, device)

    # ---------------------------------------------------------- baselines
    n = len(y_true)
    counts = np.bincount(y_true, minlength=10)
    print(f"validation images: {n}")
    print(f"overall accuracy      : {(y_true == y_pred).mean():.4f}")
    print(f"random-guess baseline : {1/10:.4f}")
    print(f"majority-class baseline: {counts.max()/n:.4f}"
          f"  (always predict '{CLASS_NAMES[counts.argmax()]}')")

    # ------------------------------------------------------- per class
    # RECALL    = of the real class-k galaxies, what fraction did we catch?
    # PRECISION = of the ones we CALLED class k, what fraction were right?
    # A class with recall 0.00 is one the model has effectively given up on
    # - usually a rare class it can safely ignore to maximise overall
    # accuracy. That is the single most common failure on Galaxy10.
    print(f"\n{'class':28s} {'n':>6s} {'recall':>8s} {'precis':>8s}")
    print("-" * 54)
    for k, name in enumerate(CLASS_NAMES):
        real_k = y_true == k
        called_k = y_pred == k
        recall = (y_pred[real_k] == k).mean() if real_k.sum() else float("nan")
        precision = (y_true[called_k] == k).mean() if called_k.sum() else float("nan")
        print(f"{name:28s} {real_k.sum():6d} {recall:8.3f} {precision:8.3f}")

    # ------------------------------------------------- confusion matrix
    # Row = true class, column = predicted class. The diagonal is correct
    # answers. Large OFF-diagonal entries are the interesting part: they
    # tell you which specific pairs the model cannot separate. Expect the
    # three spiral classes to bleed into each other, and the two edge-on
    # classes likewise - those are genuinely hard distinctions that even
    # human labellers disagree on.
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print("\nconfusion matrix (row = true, col = predicted)")
    print("     " + "".join(f"{j:6d}" for j in range(10)))
    for i in range(10):
        print(f"{i:3d}  " + "".join(f"{cm[i, j]:6d}" for j in range(10)))

    # The three worst confusions, named. This is your to-do list.
    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("\nworst confusions:")
    for idx in np.argsort(off.ravel())[::-1][:3]:
        i, j = divmod(idx, 10)
        print(f"  {off[i, j]:4d}x  '{CLASS_NAMES[i]}' called '{CLASS_NAMES[j]}'")


if __name__ == "__main__":
    main()