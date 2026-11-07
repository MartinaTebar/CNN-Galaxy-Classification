"""
model.py
--------
SimpleCNN: the smallest architecture that is still a *real* CNN.

MENTAL MODEL
  A CNN is a stack of "blocks". Each block does two jobs:

      1. LOOK       - convolution: detect patterns in small neighbourhoods
      2. SHRINK     - pooling:     halve the height and width

  Every time you shrink, you can afford to double the number of channels,
  because the tensor is 4x smaller in area. So the tensor marches like:

      (3, 128, 128) -> (32, 64, 64) -> (64, 32, 32) -> (128, 16, 16) -> (256, 8, 8)
       ^ lots of pixels,                                                ^ few pixels,
         few channels                                                     many channels

  Early channels learn edges and blobs. Later channels, which see a much
  larger patch of the original image, learn things like "spiral arm",
  "central bulge", "elongated smooth blob".

  At the very end you collapse the 8x8 grid to a single vector and ask a
  plain linear layer: given these 256 numbers, which of the 10 classes?
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    One block: Conv -> BatchNorm -> ReLU -> MaxPool.

    Conv2d(kernel_size=3, padding=1)
        3x3 is the standard kernel size. padding=1 adds a one-pixel border
        of zeros so the output has the SAME height and width as the input.
        (out = in + 2*padding - kernel + 1 = in + 2 - 3 + 1 = in.)
        Keeping the size unchanged here means the only thing that changes
        spatial size is the pooling layer - much easier to reason about.

    bias=False
        BatchNorm immediately subtracts the mean, which cancels any bias
        the conv would have added. So the bias is dead weight; skip it.

    BatchNorm2d
        Re-centres and re-scales each channel using the statistics of the
        current batch. Not strictly required, but it makes training far
        more forgiving of a badly-chosen learning rate. Cheap win.

    ReLU
        max(x, 0). Without a nonlinearity, stacking conv layers would be
        pointless: a composition of linear maps is just another linear map.
        inplace=True saves a little memory.

    MaxPool2d(2)
        Takes the largest value in each 2x2 window -> halves H and W.
        Two purposes: it cuts compute 4x, and it grows the *receptive
        field* (how much of the original image one output pixel can see),
        which is how later layers get to reason about large structures.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """
    ~390k parameters. Small enough to train on a laptop GPU in minutes,
    big enough to get meaningfully above chance on Galaxy10.

    Parameters
    ----------
    num_classes : int
        10 for Galaxy10.
    dropout : float
        Fraction of features randomly zeroed before the final linear layer,
        during training only. A crude but effective regulariser.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(3, 32),      # (3, 128, 128) -> (32, 64, 64)
            conv_block(32, 64),     #               -> (64, 32, 32)
            conv_block(64, 128),    #               -> (128, 16, 16)
            conv_block(128, 256),   #               -> (256, 8, 8)
        )

        # GLOBAL AVERAGE POOLING
        # Averages each channel's whole 8x8 map down to a single number,
        # giving a 256-vector. The obvious alternative is Flatten(), but
        # that would produce 256*8*8 = 16384 features and a final linear
        # layer with 163k parameters - 40% of the whole network spent on
        # one layer, and a prime spot for overfitting. Averaging also means
        # the model no longer cares *where* in the frame a feature was,
        # only whether it was present, which is what you want when the
        # galaxy is roughly centred but not pixel-aligned.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B, 256, 1, 1) -> (B, 256)
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),     # -> (B, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 128, 128) float32, already normalised by the transform.

        Returns RAW LOGITS of shape (B, 10) - NOT probabilities.
        Do not put a softmax here. nn.CrossEntropyLoss applies log-softmax
        internally, and doing it twice quietly wrecks your gradients.
        To see probabilities at inference time, call softmax yourself.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape sanity check. Run this before you ever start a training job -
    # it catches 90% of architecture mistakes in half a second.
    model = SimpleCNN()
    dummy = torch.randn(4, 3, 128, 128)   # a fake batch of 4 images
    out = model(dummy)
    print(model)
    print(f"\ninput  {tuple(dummy.shape)}")
    print(f"output {tuple(out.shape)}   (expected (4, 10))")
    print(f"trainable parameters: {count_parameters(model):,}")