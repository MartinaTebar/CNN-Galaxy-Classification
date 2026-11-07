"""
compare_models.py
-----------------
Is the pretrained model ACTUALLY better than the random-init one, or is
the 1.64-point gap noise?

    python src/compare_models.py

WHY NOT JUST COMPARE THE TWO ACCURACIES
  Both models were evaluated on the SAME 1768 images, so the two scores
  are not independent samples - they are paired. Comparing them as if
  they were independent throws away that pairing and loses power.

  McNemar's test uses the pairing. It ignores every image both models got
  right and every image both got wrong (those carry no information about
  which is better) and looks only at the DISAGREEMENTS:

      b = pretrained right, random wrong
      c = pretrained wrong, random right

  Under the null hypothesis "the two models are equally good", each
  disagreement is a coin flip, so b and c should be about equal. The
  p-value is the probability of seeing a split at least this lopsided
  from a fair coin.

CAVEAT
  Both checkpoints were selected on this validation split (saved at the
  best-scoring epoch), so both scores are slightly optimistic. That
  inflates both equally, so the comparison is still fair - but do not
  quote these numbers as final performance. That is what the test split
  is for.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader

from dataset import Galaxy10Dataset, build_transforms, load_split, CLASS_NAMES
from model_resnet import build_resnet18

RESOLUTION = 224
BATCH_SIZE = 64
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

MODELS = {
    "pretrained": CKPT_DIR / "resnet18_pretrained.pt",
    "random":     CKPT_DIR / "resnet18_random.pt",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predictions(ckpt_path: Path, loader, device) -> np.ndarray:
    # pretrained=False here only skips a pointless 45 MB download - the
    # next line overwrites every weight with the trained values anyway.
    model = build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds = [model(imgs.to(device)).argmax(dim=1).cpu() for imgs, _ in loader]
    return torch.cat(preds).numpy()


def main() -> None:
    device = get_device()

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Did you copy the checkpoint after each run?\n"
                f"  cp checkpoints/resnet18.pt checkpoints/resnet18_{name}.pt"
            )

    images, labels = load_split("val")
    ds = Galaxy10Dataset(
        images, labels,
        # Must match training preprocessing EXACTLY: 224px, ImageNet
        # normalisation, no augmentation.
        transform=build_transforms(RESOLUTION, pretrained=True, train=False),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    y_true = np.asarray(labels).astype(np.int64)
    pred = {name: predictions(path, loader, device) for name, path in MODELS.items()}

    ok_a = pred["pretrained"] == y_true
    ok_b = pred["random"] == y_true

    both_right = int((ok_a & ok_b).sum())
    both_wrong = int((~ok_a & ~ok_b).sum())
    b = int((ok_a & ~ok_b).sum())    # only pretrained right
    c = int((~ok_a & ok_b).sum())    # only random right

    print(f"validation images: {len(y_true)}")
    print(f"pretrained accuracy: {ok_a.mean():.4f}")
    print(f"random     accuracy: {ok_b.mean():.4f}")
    print(f"difference         : {ok_a.mean() - ok_b.mean():+.4f}\n")

    print("agreement table")
    print(f"  both right              : {both_right:5d}")
    print(f"  both wrong              : {both_wrong:5d}")
    print(f"  only pretrained right (b): {b:5d}")
    print(f"  only random right     (c): {c:5d}")
    print(f"  total disagreements     : {b + c:5d}")

    # Exact McNemar: of the b+c disagreements, is b significantly more
    # than half? Two-sided, because we should be willing to conclude
    # either direction.
    res = binomtest(b, b + c, 0.5, alternative="two-sided")
    print(f"\nMcNemar exact test: p = {res.pvalue:.4f}")
    if res.pvalue < 0.05:
        print("  -> the difference is unlikely to be chance (p < 0.05)")
    else:
        print("  -> NOT significant at the 0.05 level: on this validation")
        print("     split you cannot claim one model is better than the other")

    # Where do they disagree? If the disagreements pile up in one or two
    # classes, that is more interesting than the overall p-value.
    print(f"\n{'class':28s} {'b':>5s} {'c':>5s}")
    print("-" * 40)
    for k, name in enumerate(CLASS_NAMES):
        mask = y_true == k
        print(f"{name:28s} {int((ok_a & ~ok_b & mask).sum()):5d} "
              f"{int((~ok_a & ok_b & mask).sum()):5d}")


if __name__ == "__main__":
    main()