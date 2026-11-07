"""
preview_crop.py
---------------
Eyeball the CenterCrop before committing to a 60-epoch run.

    python3 src/preview_crop.py

Top row = original 256x256. Bottom row = the same galaxy after
CenterCrop(180). If arms are being clipped, raise the number. If there's
still a wide empty margin, lower it.
"""

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from dataset import load_split, CLASS_NAMES

CROP = 140

images, labels = load_split("val")
idxs = range(0, 400, 50)          # 8 galaxies, arbitrary but spread out

fig, axes = plt.subplots(2, len(list(idxs)), figsize=(14, 4))
for col, i in enumerate(idxs):
    img = torch.from_numpy(images[i].copy()).permute(2, 0, 1)   # (3, 256, 256)

    axes[0, col].imshow(img.permute(1, 2, 0))
    axes[0, col].set_title(CLASS_NAMES[labels[i]], fontsize=7)

    axes[1, col].imshow(transforms.CenterCrop(CROP)(img).permute(1, 2, 0))

    axes[0, col].axis("off")
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig("crop_preview.png", dpi=120)
plt.show()