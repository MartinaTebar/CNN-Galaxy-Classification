"""
train.py
--------
The training loop, written out explicitly. In PyTorch you write this
yourself rather than calling model.fit(), which is the point: nothing is
hidden, and every number below is a choice you can inspect and change.

Run from the REPO ROOT (the data/ paths are relative to it):

    python src/train.py --model simplecnn --epochs 20
    python src/train.py --model resnet18  --epochs 10 --lr 1e-3 --freeze


=====================================================================
WHERE THE DEFAULT NUMBERS COME FROM
=====================================================================
Being honest about this, because "1e-3" with no justification is just
cargo-culting. Three tiers:

PRINCIPLED - derived from something real:
  lr=1e-4 when fine-tuning an unfrozen ResNet. Pretrained weights are
    already near a good solution; a large step destroys them in the first
    few batches. Roughly 10x smaller than a from-scratch LR is the
    standard rule.
  shuffle=True on train only. If the data sat sorted by class, the model
    would see 500 spirals in a row and the batch gradients would be
    wildly biased. Val/test order cannot affect a metric, so no shuffle.
  Checkpoint on best VAL accuracy, not the last epoch. Later epochs
    usually overfit; the final model is rarely the best one.

CONVENTIONAL - the field's default, works, not derived:
  lr=1e-3 for Adam. This is literally the default in the Adam paper and
    in torch. It is the right first guess for training from scratch.
  batch_size=64. Powers of 2 suit GPU memory layout. 32-128 is the usual
    band. Larger = smoother gradients but fewer updates per epoch and
    more memory; smaller = noisier gradients, which is mildly
    regularising. If you hit out-of-memory, halve this first.
  Adam over SGD. Adam adapts a per-parameter step size and is far more
    forgiving of a bad initial LR. Well-tuned SGD+momentum often ends up
    slightly better, but "well-tuned" is the expensive part.

ARBITRARY - I picked a plausible number; tune these if needed:
  epochs=20, dropout=0.3, scheduler factor=0.5 and patience=3,
  num_workers=4.

WHAT TO CHANGE, IN ORDER, WHEN THINGS GO WRONG:
  accuracy stuck at ~0.10 (= chance)  -> real bug, or LR far too high.
                                         Check the sanity-check output.
  train acc >> val acc                -> overfitting. Raise dropout, add
                                         weight_decay=1e-4, train fewer
                                         epochs.
  both accuracies still rising at end -> just train longer.
  loss becomes nan                    -> LR too high. Divide by 10.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Galaxy10Dataset, build_transforms, load_split, CLASS_NAMES
from model import SimpleCNN, build_resnet18


# ---------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Pick the fastest available backend. Tensors and the model must live on
    the SAME device or PyTorch raises an error - hence all the .to(device)
    calls below.

    'mps' is Apple Silicon (M1/M2/M3). Your notebook shows a Mac, so if
    you are on Apple Silicon this gives a large speedup over CPU. If you
    are on an older Intel Mac you will land on CPU, and SimpleCNN at
    128x128 is still perfectly trainable there - just slower.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One full pass over the training set. This is the only function in the
    project that changes the model's weights.
    """
    # model.train() does NOT train anything. It is a MODE FLAG. It tells
    # every submodule to use training behaviour:
    #   Dropout    -> actively zeroes activations
    #   BatchNorm  -> normalises using THIS BATCH's statistics, and
    #                 updates its stored running averages
    # Forgetting this (or its eval() counterpart) is the most common
    # PyTorch bug there is, and it fails silently - no error, just worse
    # numbers.
    model.train()

    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        # CPU-side copy of the labels, for the same MPS int64 reason
        # documented in evaluate() below.
        labels_cpu = labels.to(torch.int64)

        # The DataLoader yields CPU tensors; the model may be on GPU/MPS.
        # non_blocking=True pairs with pin_memory to overlap this copy with
        # computation on CUDA. Harmless elsewhere.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---------------- THE FIVE LINES ------------------------------
        # This exact sequence appears in essentially every PyTorch project.
        # Worth memorising.

        # 1. Clear gradients left over from the previous batch.
        #    PyTorch ACCUMULATES into .grad by default (useful for
        #    simulating large batches). Skip this and step N is applied
        #    using the SUM of gradients from steps 1..N. Silent, and
        #    ruinous.
        optimizer.zero_grad()

        # 2. Forward pass -> raw logits, shape (B, 10). Running this also
        #    BUILDS THE AUTOGRAD GRAPH as a side effect, which is what
        #    makes step 4 possible without you writing any calculus.
        outputs = model(images)

        # 3. Scalar loss. CrossEntropyLoss takes RAW LOGITS (B, 10) and
        #    INTEGER labels (B,) - not one-hot, not softmaxed.
        loss = criterion(outputs, labels)

        # 4. Walk the graph backwards, filling p.grad for every parameter.
        loss.backward()

        # 5. Apply the update rule, reading .grad off each parameter.
        #    The optimizer never talks to the model - it was handed the
        #    parameter list at construction and just walks it.
        optimizer.step()
        # ---------------------------------------------------------------

        # loss.item() pulls a plain Python float OFF the autograd graph.
        # Accumulating the tensor itself would keep every batch's graph
        # alive in memory and you would OOM partway through the epoch.
        #
        # Why "* labels.size(0)"? loss is the MEAN over the batch. The last
        # batch of an epoch is usually smaller than the rest, so a plain
        # average-of-averages would over-weight it. Multiplying back up to
        # a sum and dividing by the true total gives the correct per-image
        # mean.
        running_loss += loss.item() * labels_cpu.size(0)

        # argmax on the CPU copy of the logits, for the same MPS reason.
        # Softmax is monotonic, so the argmax of the logits equals the
        # argmax of the probabilities - no need to convert.
        preds_cpu = outputs.detach().float().cpu().argmax(dim=1)
        correct += (preds_cpu == labels_cpu).sum().item()
        total += labels_cpu.size(0)

    # CAVEAT ON THIS NUMBER: the weights changed after every batch, so this
    # is an average over a MOVING model, not a measurement of the model you
    # end the epoch with. It usually reads slightly worse than reality.
    # Do not over-interpret small train-vs-val gaps because of this.
    return running_loss / total, correct / total


# ---------------------------------------------------------------------
# @torch.no_grad() disables graph construction for everything inside.
# Roughly 2x faster and much lighter on memory. Nothing here needs
# gradients because nothing here updates weights.
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=10):
    """
    Measure performance on val or test. Never modifies the model.

    "WHY RECOMPUTE THE OUTPUTS - CAN'T WE REUSE THE TRAINING RESULTS?"
    Four independent reasons, and they stack:

      1. DIFFERENT DATA. train_one_epoch iterates the train loader; this
         iterates val or test. Those are disjoint sets of galaxies. The
         validation images were never passed through the model during
         training - that is the entire point of holding them out. There
         is literally nothing to reuse.

      2. DIFFERENT MODE, DIFFERENT MATHS. Under train(), dropout is
         randomly zeroing 30% of features and BatchNorm is using batch
         statistics. Under eval(), dropout is off and BatchNorm uses its
         stored running averages. Feed the SAME image through both modes
         and the logits differ substantially. Training-mode outputs are
         deliberately corrupted for regularisation; they are not the
         model's real predictions.

      3. STALE WEIGHTS. optimizer.step() fires after every batch, so the
         model that produced batch 1's outputs has been updated hundreds
         of times by the end of the epoch. There is no single "the model"
         during a training pass.

      4. IT IS CHEAP. Under no_grad this costs about half a training pass,
         on a set one eighth the size. The saving from reuse would be
         negligible even if it were possible.

    What we DO reuse: `outputs` below is computed once and serves both the
    loss and the accuracy count. No wasted forward passes inside the loop.
    """
    # Mode flag again - the counterpart to model.train(). Deterministic
    # behaviour: same input always gives the same output.
    model.eval()

    running_loss, correct, total = 0.0, 0, 0

    # confusion[true_class, predicted_class] += 1
    # This matters more than accuracy on an imbalanced set: it shows you
    # WHICH classes get confused with WHICH, not just how often.
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for images, labels in loader:
        # KEEP A CPU-SIDE COPY OF THE LABELS BEFORE THEY EVER TOUCH THE
        # DEVICE. On Apple Silicon (MPS), int64 support is incomplete, and
        # round-tripping an int64 tensor CPU -> MPS -> CPU can come back as
        # garbage (you get nonsense like index -7266999397788931644). The
        # labels arrive from the DataLoader already on CPU as int64, so we
        # simply never ask the device to give them back.
        labels_cpu = labels.to(torch.int64)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)                 # one forward pass...
        loss = criterion(outputs, labels)       # ...used for the loss...

        # Bring the LOGITS back as float32 (which MPS handles correctly)
        # and take the argmax on the CPU, so no int64 ever crosses the
        # device boundary in the wrong direction.
        preds_cpu = outputs.detach().float().cpu().argmax(dim=1)

        running_loss += loss.item() * labels_cpu.size(0)
        correct += (preds_cpu == labels_cpu).sum().item()
        total += labels_cpu.size(0)

        # Vectorised confusion accounting: flatten (true, pred) to a single
        # index, count with bincount, reshape back. Faster than a Python
        # loop over the batch, and there is no per-element indexing that
        # could blow up on a bad value.
        flat = labels_cpu * num_classes + preds_cpu
        confusion += (torch.bincount(flat, minlength=num_classes ** 2)
                      .numpy().reshape(num_classes, num_classes))

    return running_loss / total, correct / total, confusion


# ---------------------------------------------------------------------
def per_class_accuracy(confusion):
    """
    Recall per class: of all the true examples of class k, what fraction
    did we get right? = diagonal / row sum.

    Why this matters here: Galaxy10 is imbalanced. A model that completely
    ignores the rarest class can still post a respectable headline
    accuracy by doing well on the common ones. This number exposes that.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.diag(confusion) / confusion.sum(axis=1))


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simplecnn", "resnet18"], default="simplecnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="L2 penalty. Try 1e-4 if you see overfitting.")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n-convs", type=int, default=2,
                    help="convs per block in SimpleCNN. 1 = fast plumbing "
                         "run, 2 = the real thing (bigger receptive field)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--freeze", action="store_true",
                    help="resnet18 only: train just the new final layer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Fixing the seed makes runs comparable. Without this, two runs differ
    # by weight init, dropout masks and shuffle order, and you cannot tell
    # whether a change helped or you just got lucky. Note this does NOT
    # give bit-exact reproducibility on GPU, but it removes most variance.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    # --- model, and the input resolution it expects --------------------
    # These are coupled: SimpleCNN was designed around 128px (four
    # halvings -> 8x8 feature maps), while ResNet18's pretrained weights
    # were learned at 224px with ImageNet normalisation. Feeding ResNet
    # differently-scaled inputs throws away much of the transfer benefit.
    if args.model == "simplecnn":
        model = SimpleCNN(num_classes=10, dropout=args.dropout,
                          n_convs=args.n_convs)
        resolution, pretrained = 128, False
    else:
        model = build_resnet18(num_classes=10, freeze_backbone=args.freeze)
        resolution, pretrained = 224, True
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {args.model} | trainable params: {n_params:,}")

    # --- data ----------------------------------------------------------
    loaders = {}
    for split in ("train", "val", "test"):
        images, labels = load_split(split)
        ds = Galaxy10Dataset(
            images, labels,
            # train=True adds random flips/rotations. Val and test must
            # stay deterministic or the metric wobbles for no reason.
            transform=build_transforms(resolution, pretrained,
                                       train=(split == "train")),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            # Shuffle train only. Order must never be a signal the model
            # can exploit; for val/test, order cannot affect the result.
            shuffle=(split == "train"),
            # Background processes that prepare the NEXT batch while the
            # GPU works on the current one. Without this the GPU sits idle
            # waiting for JPEG-sized arrays to be resized and normalised.
            # Set to 0 if you hit multiprocessing errors on macOS/Windows.
            num_workers=args.workers,
            # Page-locked memory, which makes the CPU->GPU copy faster.
            # CUDA only; pointless overhead otherwise.
            pin_memory=(device.type == "cuda"),
        )
        print(f"{split:5s}: {len(ds)} images")

    # --- SANITY CHECK: always do this before a long run ----------------
    # Pull one batch and actually look at it. Thirty seconds here saves an
    # hour of debugging a run that was silently training on garbage.
    xb, yb = next(iter(loaders["train"]))
    print(f"\nbatch images {tuple(xb.shape)} {xb.dtype} "
          f"range [{xb.min():.2f}, {xb.max():.2f}]")
    print(f"batch labels {tuple(yb.shape)} {yb.dtype} e.g. {yb[:8].tolist()}")
    # EXPECT: (64, 3, 128, 128) float32, values roughly [-1, 1]
    #         (64,) int64, values 0..9  -- integers, NOT one-hot vectors
    # If dtype is uint8, normalisation did not run.
    # If the range is [0, 255], the transform pipeline is broken.
    # If labels are 2-D, you left Keras' to_categorical in somewhere.

    # --- loss ----------------------------------------------------------
    # CrossEntropyLoss = log_softmax + negative log-likelihood, fused into
    # one numerically-stable op. It takes RAW LOGITS and INTEGER labels.
    # This is why SimpleCNN.forward has no softmax: applying one here
    # would be applying it twice.
    #
    # SANITY: an untrained 10-class model should give loss ~= ln(10) = 2.30.
    # If your epoch-1 loss is far from that, something is wrong before
    # training even starts.
    criterion = nn.CrossEntropyLoss()

    # OPTIONAL - class weighting for the imbalance. Raises the loss
    # contribution of rare classes so the model cannot just ignore them.
    # Run once WITHOUT it, read the per-class accuracies, then decide.
    # Turning it on trades a little overall accuracy for better rare-class
    # recall, which for a morphology study is usually the right trade.
    #   counts = np.bincount(load_split("train", mmap=False)[1], minlength=10)
    #   w = torch.tensor(counts.sum() / (10 * counts), dtype=torch.float32)
    #   criterion = nn.CrossEntropyLoss(weight=w.to(device))

    # --- optimiser -----------------------------------------------------
    # Filter to parameters that actually require gradients, otherwise the
    # optimiser complains about the frozen ResNet backbone.
    params = [p for p in model.parameters() if p.requires_grad]

    # Adam keeps a per-parameter running estimate of gradient mean and
    # variance, and scales each step accordingly. Practical effect:
    # parameters with consistently small gradients still move. This makes
    # it much more forgiving of a badly chosen LR than plain SGD, which is
    # why it is the right default when you are not tuning carefully.
    #
    # weight_decay is L2 regularisation: it pulls weights toward zero,
    # discouraging the model from memorising. Default 0 here; turn it on
    # (1e-4) if you see the train/val gap widen.
    optimizer = torch.optim.Adam(params, lr=args.lr,
                                 weight_decay=args.weight_decay)

    # LEARNING RATE SCHEDULE
    # Big steps early to explore the loss landscape, small steps later to
    # settle into a minimum instead of bouncing around it.
    #
    # ReduceLROnPlateau watches a metric and cuts the LR when it stops
    # improving:
    #   mode="min"   we are watching val LOSS, so lower is better
    #   factor=0.5   halve the LR when triggered (0.1 is also common;
    #                0.5 is gentler and less likely to stall you early)
    #   patience=3   wait 3 epochs with no improvement before cutting.
    #                Too small and normal epoch-to-epoch noise triggers
    #                it; too large and you waste epochs at a stale LR.
    #
    # Watching val LOSS rather than val ACCURACY: loss is continuous and
    # responds to the model getting more confident, while accuracy is a
    # step function that can sit still for several epochs while the model
    # is genuinely improving underneath.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- the epoch loop -------------------------------------------------
    # One epoch = one pass over every training image. The outer loop does
    # nothing clever: it trains, measures, and records.
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, loaders["train"],
                                          criterion, optimizer, device)
        va_loss, va_acc, _ = evaluate(model, loaders["val"],
                                      criterion, device)

        # Feed the scheduler the metric it is watching. Must be called
        # AFTER the optimizer steps for this epoch, not before.
        scheduler.step(va_loss)

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc,
                            lr=optimizer.param_groups[0]["lr"]))

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"{time.time() - t0:.0f}s")

        # CHECKPOINTING
        # Save the weights from the best VALIDATION epoch, not the last
        # one. Training accuracy keeps climbing long after the model has
        # started overfitting, so the final epoch is rarely the best.
        #
        # state_dict() is a plain dict of tensors - just the weights, not
        # the code. To reload you must first construct the same
        # architecture, then load into it.
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_dir / f"{args.model}_best.pt")
            print(f"    saved (best val acc {va_acc:.3f})")

    np.save(ckpt_dir / f"{args.model}_history.npy", history)

    # --- final evaluation on the TEST set -------------------------------
    # Reload the best checkpoint, because `model` currently holds the
    # LAST epoch's weights, which may be worse.
    model.load_state_dict(torch.load(ckpt_dir / f"{args.model}_best.pt"))

    # This is the one and only time the test set is touched. The moment
    # you tune anything based on this number, it stops being an honest
    # estimate of generalisation and becomes just another val set.
    te_loss, te_acc, confusion = evaluate(model, loaders["test"],
                                          criterion, device)
    print(f"\nTEST loss {te_loss:.4f} | accuracy {te_acc:.4f}")

    print("\nper-class accuracy (recall):")
    for name, acc, n in zip(CLASS_NAMES,
                            per_class_accuracy(confusion),
                            confusion.sum(axis=1)):
        print(f"  {name:26s} {acc:.3f}  (n={n})")

    np.save(ckpt_dir / f"{args.model}_confusion.npy", confusion)
    print(f"\nconfusion matrix saved to {ckpt_dir}/{args.model}_confusion.npy")


if __name__ == "__main__":
    main()