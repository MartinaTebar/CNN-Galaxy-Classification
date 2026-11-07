"""
model_resnet.py
---------------
ResNet18 with ImageNet-pretrained weights, re-headed for Galaxy10.

WHY THIS SHOULD BEAT SimpleCNN
  Your SimpleCNN learned its edge and texture detectors from ~14k galaxy
  images. ResNet18's were learned from 1.2M ImageNet photos. Early-layer
  features - edges, gradients, blobs, textures - are almost universal, so
  those million photos of dogs and cars bought you a feature extractor
  that no amount of galaxy data at your scale could match.

  It is also 11.2M parameters against your 391k, and has residual
  connections, which is what lets a network this deep train at all.

THE ONLY CLASS-AWARE PART GETS REPLACED
  torchvision ships resnet18 with fc = Linear(512, 1000) for ImageNet's
  1000 classes. We throw that away and bolt on Linear(512, 10). Everything
  before it is class-agnostic and gets reused as-is.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 10,
                   pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    """
    freeze_backbone
        False (default) -> FINE-TUNING. Every weight is trainable; the
            pretrained values are just a very good starting point. This is
            the right choice here, because galaxies look nothing like
            ImageNet photos, so the deeper layers genuinely need to change.

        True -> FEATURE EXTRACTION. Only the new fc layer trains; the
            backbone is a frozen function. Trains in a fraction of the time
            (no gradients for 11.2M parameters) and cannot overfit much,
            but caps out lower. Useful as a fast baseline, or as a warm-up
            stage before unfreezing.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # ORDER MATTERS: replace fc AFTER the freeze loop. A freshly created
    # Linear has requires_grad=True by default, so doing it in this order
    # leaves exactly the head trainable. Freeze first, re-head second.
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # 512 -> 10

    return model


def param_counts(model: nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for freeze in (False, True):
        m = build_resnet18(freeze_backbone=freeze)
        total, trainable = param_counts(m)
        out = m(torch.randn(2, 3, 224, 224))
        print(f"freeze_backbone={freeze!s:5s} | total {total:,} | "
              f"trainable {trainable:,} | out {tuple(out.shape)}")
    # This is the concrete case where "total" and "trainable" differ - the
    # distinction that looked pointless back when both were 391,466.