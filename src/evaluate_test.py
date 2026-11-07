"""
evaluate_test.py
----------------
The FINAL evaluation: all three trained models on the held-out TEST split.
Run from the repo root:

    python src/evaluate_test.py | tee logs/test_results.txt

WHY THIS IS A SEPARATE SCRIPT
  Every decision so far (architecture, learning rate, crop size, which
  epoch to keep) was made by looking at the VALIDATION split, so validation
  scores are slightly optimistic. The test split has never influenced any
  choice, which makes it the only honest estimate of how the models do on
  galaxies they have never seen.

  That only stays true if you run this ONCE, after you have stopped
  changing the models. If you look at these numbers and then go back and
  tweak things, the test set has become a second validation set.

WHAT IT REPORTS
  - accuracy           : fraction of galaxies classified correctly
  - balanced accuracy  : mean of the per-class recalls. Every class counts
                         equally, so a model cannot score well by ignoring
                         the rare classes.
  - macro F1           : mean per-class F1 (balance of precision and recall)
  - per-class recall, and a confusion matrix per model
  - McNemar's exact test for every pair of models (see compare_models.py
    for why a paired test is the right one here)

It also writes the numbers to logs/test_metrics.json for plotting.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import Galaxy10Dataset, build_transforms, load_split, CLASS_NAMES
from model_simple import SimpleCNN
from model_resnet import build_resnet18

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
OUT_JSON = ROOT / "logs" / "test_metrics.json"
BATCH_SIZE = 64

# Each model must be evaluated with EXACTLY the preprocessing it was trained
# with: same resolution, same normalisation, no augmentation.
#   imagenet_norm=True  -> ImageNet mean/std (both ResNets)
#   imagenet_norm=False -> scale to [-1, 1]  (SimpleCNN)
MODELS = {
    "SimpleCNN": dict(
        ckpt=CKPT_DIR / "simplecnn.pt",
        build=lambda: SimpleCNN(num_classes=len(CLASS_NAMES)),
        resolution=128, imagenet_norm=False,
    ),
    "ResNet18 (random init)": dict(
        ckpt=CKPT_DIR / "resnet18_random.pt",
        build=lambda: build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False),
        resolution=224, imagenet_norm=True,
    ),
    "ResNet18 (ImageNet pretrained)": dict(
        ckpt=CKPT_DIR / "resnet18_pretrained.pt",
        # pretrained=False only skips the ImageNet download: the next step
        # overwrites every weight with our trained checkpoint anyway.
        build=lambda: build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False),
        resolution=224, imagenet_norm=True,
    ),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(spec: dict, images, labels, device) -> np.ndarray:
    ds = Galaxy10Dataset(
        images, labels,
        transform=build_transforms(spec["resolution"], pretrained=spec["imagenet_norm"],
                                   train=False),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = spec["build"]().to(device)
    model.load_state_dict(torch.load(spec["ckpt"], map_location=device))
    model.eval()

    preds = [model(imgs.to(device)).argmax(dim=1).cpu() for imgs, _ in loader]
    return torch.cat(preds).numpy()


def main() -> None:
    for name, spec in MODELS.items():
        if not spec["ckpt"].exists():
            raise FileNotFoundError(f"{spec['ckpt']} not found (needed for {name}).")

    device = get_device()
    images, labels = load_split("test")
    y_true = np.asarray(labels).astype(np.int64)
    n = len(y_true)
    counts = np.bincount(y_true, minlength=len(CLASS_NAMES))

    print(f"device: {device}")
    print(f"test images: {n}")
    print(f"random-guess baseline  : {1 / len(CLASS_NAMES):.4f}")
    print(f"majority-class baseline: {counts.max() / n:.4f}"
          f"  (always predict '{CLASS_NAMES[counts.argmax()]}')")

    preds, results = {}, {}
    for name, spec in MODELS.items():
        y_pred = predict(spec, images, labels, device)
        preds[name] = y_pred
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
        recall = cm.diagonal() / cm.sum(axis=1)
        results[name] = {
            "accuracy": float((y_pred == y_true).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "per_class_recall": dict(zip(CLASS_NAMES, map(float, recall))),
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------ summary table
    print(f"\n{'model':32s} {'accuracy':>9s} {'bal. acc':>9s} {'macro F1':>9s}")
    print("-" * 62)
    for name, r in results.items():
        print(f"{name:32s} {r['accuracy']:9.4f} {r['balanced_accuracy']:9.4f} "
              f"{r['macro_f1']:9.4f}")

    # -------------------------------------------------- per-class recall
    short = ["SimpleCNN", "ResNet rand", "ResNet pre"]
    print(f"\nper-class recall\n{'class':28s} {'n':>5s} " +
          " ".join(f"{s:>11s}" for s in short))
    print("-" * (35 + 12 * len(short)))
    for k, cls in enumerate(CLASS_NAMES):
        row = " ".join(f"{results[m]['per_class_recall'][cls]:11.3f}" for m in MODELS)
        print(f"{cls:28s} {counts[k]:5d} {row}")

    # ------------------------------------------------- confusion matrices
    for name, r in results.items():
        cm = np.array(r["confusion_matrix"])
        print(f"\nconfusion matrix - {name} (row = true, col = predicted)")
        print("     " + "".join(f"{j:6d}" for j in range(len(CLASS_NAMES))))
        for i in range(len(CLASS_NAMES)):
            print(f"{i:3d}  " + "".join(f"{cm[i, j]:6d}" for j in range(len(CLASS_NAMES))))

    # ------------------------------------------------- McNemar, pairwise
    # Three comparisons, so with a Bonferroni correction a difference is
    # significant at the 5% level only if p < 0.05 / 3 = 0.0167.
    pairs = list(combinations(MODELS, 2))
    alpha = 0.05 / len(pairs)
    print(f"\nMcNemar exact test (Bonferroni-corrected threshold p < {alpha:.4f})")
    results["mcnemar"] = {}
    for a, b in pairs:
        ok_a, ok_b = preds[a] == y_true, preds[b] == y_true
        only_a = int((ok_a & ~ok_b).sum())
        only_b = int((~ok_a & ok_b).sum())
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        verdict = "significant" if p < alpha else "not significant"
        print(f"  {a} vs {b}\n"
              f"    only first right: {only_a:4d} | only second right: {only_b:4d} | "
              f"p = {p:.2e}  -> {verdict}")
        results["mcnemar"][f"{a} vs {b}"] = {
            "only_first_right": only_a, "only_second_right": only_b, "p_value": float(p),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"n_test": n, "class_counts": counts.tolist(),
                                    "results": results}, indent=2))
    print(f"\nmetrics written to {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
