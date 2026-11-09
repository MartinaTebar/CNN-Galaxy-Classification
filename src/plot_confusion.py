"""
plot_confusion.py
-----------------
Draw the test-set confusion matrix of one model as a PNG for the README.
Run from the repo root, after evaluate_test.py:

    python src/plot_confusion.py

No model is loaded: the confusion matrices are read from
logs/test_metrics.json, which evaluate_test.py writes.

Each row is divided by its total, so a cell shows the FRACTION of that
true class that got each prediction. The diagonal is then exactly the
per-class recall, and big and small classes can be compared directly.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- config
MODEL = "ResNet18 (ImageNet pretrained)"   # any key in test_metrics.json
ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "logs" / "test_metrics.json"
OUT_PATH = ROOT / "figures" / "confusion_resnet_pretrained.png"


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    result = metrics["results"][MODEL]

    counts = np.array(result["confusion_matrix"])
    class_names = list(result["per_class_recall"].keys())   # label order 0..9
    fractions = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)

    # write the percentage in every non-empty cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if counts[i, j] == 0:
                continue
            value = fractions[i, j]
            label = "<1" if 100 * value < 0.5 else f"{100 * value:.0f}"
            ax.text(
                j, i, label,
                ha="center", va="center", fontsize=9,
                color="white" if value > 0.5 else "black",
            )

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        f"{MODEL}: test-set confusion matrix\n"
        f"(% of each true class, test accuracy {100 * result['accuracy']:.1f}%)"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of true class")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
