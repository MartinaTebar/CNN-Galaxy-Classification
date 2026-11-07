"""
train_resnet.py
---------------
The smallest honest training loop, for ResNet18. Run from the repo root:

    python src/train_resnet.py          | tee logs/resnet_pretrained.txt
    python src/train_resnet.py --random | tee logs/resnet_random.txt

The first starts from ImageNet weights and saves
checkpoints/resnet18_pretrained.pt; the second starts from random weights
and saves checkpoints/resnet18_random.pt.

MENTAL MODEL
  Training is one loop, repeated:

      for each epoch:
          for each batch in the training set:
              1. forward  : logits = model(images)
              2. loss     : how wrong were we?
              3. backward : loss.backward() fills every .grad
              4. step     : optimiser nudges every weight downhill
          then: run over the validation set WITHOUT step 3 or 4,
                and record the score.

  That's it. Everything else in this file is bookkeeping around those
  four lines.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Galaxy10Dataset, build_transforms, load_split, CLASS_NAMES
from model_simple import count_parameters
from model_resnet import build_resnet18

# ---------------------------------------------------------------- config
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
RESOLUTION = 224
NUM_WORKERS = 4          # set to 0 if you hit multiprocessing errors
SEED = 42                # fixes weight init, shuffling and augmentation
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def set_seed(seed: int) -> None:
    # Same seed -> same initial weights, same batch order, same random
    # flips/rotations. GPU kernels (especially MPS) are not bit-for-bit
    # deterministic, so reruns land very close but not always identical.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():     # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(split: str, train: bool) -> DataLoader:
    """
    The Dataset says "here is sample i". The DataLoader handles everything
    else: it picks which indices to fetch, stacks them into a batch, and
    (with num_workers > 0) does that in background processes so the GPU
    never sits idle waiting for JPEG-sized numpy slices.
    """
    images, labels = load_split(split)
    ds = Galaxy10Dataset(
        images,
        labels,
        # ImageNet normalisation for BOTH the pretrained and the random-init
        # run, so the starting weights are the only thing that differs
        # between the two experiments.
        transform=build_transforms(RESOLUTION, pretrained=True, train=train),
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=train,          # shuffle TRAIN only. Never shuffle val/test:
                                # it changes nothing about the score and only
                                # makes debugging harder.
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,        # drop a ragged final batch during training so
                                # BatchNorm never sees a batch of size 1.
    )


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()               # tells Dropout to drop and BatchNorm to use
                                # batch statistics. Forgetting this is the
                                # single most common PyTorch bug.
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1. forward
        logits = model(images)                 # (B, 10) raw scores

        # 2. loss. CrossEntropyLoss expects RAW LOGITS and INTEGER labels.
        loss = criterion(logits, labels)

        # 3. backward. zero_grad first, because PyTorch *accumulates*
        #    gradients by default - without this you would be summing the
        #    gradients of every batch you have ever seen.
        optimiser.zero_grad(set_to_none=True)
        loss.backward()

        # 4. step: weight -= lr * gradient (roughly; Adam is fancier)
        optimiser.step()

        # bookkeeping. .item() pulls a python float off the GPU.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()                # no gradients needed -> less memory, faster
def evaluate(model, loader, criterion, device):
    model.eval()                # Dropout off, BatchNorm uses running stats
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ResNet18 on Galaxy10.")
    ap.add_argument("--random", action="store_true",
                    help="start from random weights instead of ImageNet")
    args = ap.parse_args()
    pretrained = not args.random
    ckpt_path = CKPT_DIR / ("resnet18_pretrained.pt" if pretrained else "resnet18_random.pt")

    set_seed(SEED)
    device = get_device()
    print(f"device: {device} | seed: {SEED} | "
          f"init: {'ImageNet pretrained' if pretrained else 'random'}")

    train_loader = make_loader("train", train=True)
    val_loader = make_loader("val", train=False)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = build_resnet18(num_classes=len(CLASS_NAMES), pretrained=pretrained).to(device)
    print(f"parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    # optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        lr_now = optimiser.param_groups[0]["lr"]

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        flag = ""
        if va_acc > best_val_acc:
            # Save the weights whenever validation improves. The model at the
            # LAST epoch is usually not the best one - it has typically begun
            # to overfit. This is your "early stopping, poor man's edition".
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- saved"

        print(
            # f"epoch {epoch:2d}/{EPOCHS} | "
            f"epoch {epoch:2d}/{EPOCHS} | lr {lr_now:.2e} | " 
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}{flag}"
        )

    print(f"\nbest val accuracy: {best_val_acc:.4f}")
    print(f"weights saved to {ckpt_path}")
    print("Do NOT touch the test set until you have stopped changing the model.")


if __name__ == "__main__":
    main()