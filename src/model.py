# """
# model.py
# --------
# Two models, trained as two SEPARATE experiments on the same splits:

#   1. SimpleCNN  - built from scratch, so you understand what the pieces do.
#   2. ResNet18   - pretrained on ImageNet, fine-tuned. The comparison
#                   baseline; it should beat SimpleCNN, and the interesting
#                   question is by how much.

# They are not combined and one is not built from the other.
# """

# import torch
# import torch.nn as nn
# from torchvision import models


# # =====================================================================
# # 1. SimpleCNN
# # =====================================================================
# #
# # THE CORE IDEA
# #   A conv layer slides a small learned filter (here 3x3) across the image
# #   and records how strongly each position matches it. Early filters learn
# #   edges and blobs; later ones, built on top, learn things like "spiral
# #   arm" or "bright central bulge".
# #
# #   Two numbers control the shape of everything:
# #     in_channels  - depth of the incoming feature map (3 for RGB input)
# #     out_channels - how many different filters this layer learns
# #
# #   Each block halves the spatial size (MaxPool2d) and doubles the depth.
# #   So a 128x128x3 input becomes:
# #       block1 ->  64x 64x 32
# #       block2 ->  32x 32x 64
# #       block3 ->  16x 16x128
# #       block4 ->   8x  8x256
# #   Trading resolution for semantic richness is the standard CNN pattern.


# def conv_block(in_ch: int, out_ch: int, n_convs: int = 2) -> nn.Sequential:
#     """
#     (Conv -> BatchNorm -> ReLU) x n_convs, then one MaxPool.

#     WHY MORE THAN ONE CONV BEFORE POOLING?
#       Not to "try more filters" - the number of filters is out_ch, which is
#       independent of how many convs you stack. Stacking buys two things:

#       1. RECEPTIVE FIELD. One 3x3 conv lets an output pixel see a 3x3 input
#          patch; two stacked see 5x5. Compounded over four blocks:
#              n_convs=1  ->  46 px of a 128 px image  (only 36%!)
#              n_convs=2  ->  76 px of a 128 px image
#          Galaxy morphology is a GLOBAL property (arm tightness, elongation,
#          overall symmetry), so a neuron that has never seen most of the
#          galaxy is badly handicapped. This matters here.

#       2. NONLINEARITY PER PARAMETER. Two 3x3 convs = 18 weights per channel
#          pair and TWO ReLUs. One 5x5 conv covers the same spatial extent
#          with 25 weights and only ONE ReLU. Cheaper and more expressive.
#          This is the VGG argument, and it is why 3x3 became universal.

#       n_convs=1 is faster per epoch; use it for a quick plumbing check.
#     """
#     layers = []
#     for i in range(n_convs):
#         layers += [
#             # SHAPE ARITHMETIC:  H_out = floor((H_in + 2*pad - kernel)/stride) + 1
#             # With kernel=3, stride=1, pad=1:  H_out = H_in + 2 - 3 + 1 = H_in.
#             # H_in cancels, so this is resolution-INDEPENDENT. The rule for
#             # "same" padding at stride 1 is  pad = (kernel - 1) / 2:
#             #     kernel 3 -> pad 1,  kernel 5 -> pad 2,  kernel 7 -> pad 3
#             # (Kernels are odd so they have a centre pixel and can be padded
#             # symmetrically.) Conv2d's stride defaults to 1, which is what we
#             # want: convs should not downsample here.
#             nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
#                       kernel_size=3, padding=1, bias=False),

#             # BatchNorm renormalises activations within each batch. It makes
#             # training far more forgiving of a badly chosen learning rate.
#             # bias=False above because BatchNorm has its own shift term, so
#             # the conv bias would be redundant.
#             nn.BatchNorm2d(out_ch),

#             # ReLU: max(0, x). The nonlinearity. Without one, stacking layers
#             # would collapse into a single linear map and depth would buy
#             # you nothing.
#             nn.ReLU(inplace=True),
#         ]

#     # Halve height and width, keeping the strongest response in each 2x2
#     # window. NOTE: MaxPool2d's stride DEFAULTS TO kernel_size, so this is
#     # implicitly stride 2. It is the only downsampling in the network.
#     # (Pooling acts on each channel independently - it never changes the
#     # channel count. The conv layer alone decides depth.)
#     layers.append(nn.MaxPool2d(kernel_size=2))
#     return nn.Sequential(*layers)


# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes: int = 10, dropout: float = 0.3,
#                  n_convs: int = 2):
#         super().__init__()

#         # WHY DOES THE DEPTH DOUBLE EACH BLOCK?
#         #   Channels are a VOCABULARY of feature detectors, and each conv
#         #   kernel has shape (in_ch, 3, 3) and produces exactly one output
#         #   map. Asking for out_ch=64 is asking for 64 distinct detectors.
#         #
#         #   The useful vocabulary grows with abstraction. Layer 1 needs only
#         #   oriented edges and blobs - a few dozen covers it. Layer 4 needs
#         #   compositions ("bright core with diffuse halo", "curved arc of a
#         #   given tightness"), and the number of useful COMBINATIONS of
#         #   simpler parts explodes. So the vocabulary widens as it deepens.
#         #
#         #   It also balances the compute budget. Conv cost is roughly
#         #   H * W * in_ch * out_ch * 9. Halving H and W divides by 4;
#         #   doubling both channel counts multiplies by 4. Every block
#         #   therefore costs about the same, instead of the budget being
#         #   spent entirely on high-resolution early layers.
#         #
#         #   This is a convention (from VGG), not a law. Constant width
#         #   trains fine; it just tends to underperform per unit of compute.
#         self.features = nn.Sequential(
#             conv_block(3, 32, n_convs),      # 128x128 -> 64x64
#             conv_block(32, 64, n_convs),     #  64x64  -> 32x32
#             conv_block(64, 128, n_convs),    #  32x32  -> 16x16
#             conv_block(128, 256, n_convs),   #  16x16  ->  8x8
#         )

#         # GLOBAL AVERAGE POOLING
#         # Input here is (B, 256, 8, 8). This averages each 8x8 map down to a
#         # single number -> (B, 256, 1, 1). "Adaptive" means you specify the
#         # OUTPUT size and it derives the window, so a 224px input (7x7 maps)
#         # still yields 256 numbers. That is why this model does not crash at
#         # other resolutions.
#         #
#         # Each of the 256 numbers answers: "how strongly does feature k
#         # appear ANYWHERE in this image?" Position is discarded entirely -
#         # correct here, since where the galaxy sits in the frame carries no
#         # morphological information.
#         #
#         # The alternative, Flatten() + Linear(256*8*8, 10), is 16384 -> 10 =
#         # 163,850 parameters: ~40% more than the whole rest of the network
#         # (388,896), concentrated in one position-dependent layer that
#         # overfits readily, and it hard-codes the input resolution.
#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.classifier = nn.Sequential(
#             # Pure reshape, no maths: (B, 256, 1, 1) -> (B, 256)
#             nn.Flatten(),

#             # During TRAINING, zero 30% of the 256 features on every forward
#             # pass and scale the survivors by 1/0.7 so expected magnitude is
#             # unchanged. This stops the classifier leaning too hard on any
#             # single feature, since it might be missing next time.
#             # Automatically a no-op under model.eval().
#             nn.Dropout(dropout),

#             # y = xW^T + b, with W of shape (10, 256). Each of the 10 outputs
#             # is a learned weighted sum of the 256 feature strengths:
#             # "class 5 is likely when features 12, 88, 201 are strong and 40
#             # is weak." Only 2,570 parameters.
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x):
#         """
#         Defines the computation. Note this is a DECLARATION, re-executed on
#         every call - which is also when the autograd graph gets built (see
#         the note at the bottom of this file). You never write a backward().

#         Call it as model(x), NOT model.forward(x): nn.Module.__call__ runs
#         registered hooks and then dispatches here. Bypassing it silently
#         skips machinery some features rely on.

#         The batch dimension B is never mentioned by any layer. Every op
#         treats it as "do this independently, N times" - the one exception
#         being BatchNorm, which deliberately looks ACROSS the batch to
#         compute its statistics. That is exactly why it behaves differently
#         under train() and eval().
#         """
#         # x arrives as (B, 3, 128, 128) float32
#         x = self.features(x)      # -> (B, 256, 8, 8)   nn.Sequential pipes
#                                   #    its children in order; it is forward
#                                   #    all the way down.
#         x = self.pool(x)          # -> (B, 256, 1, 1)
#         return self.classifier(x) # -> (B, 10)
#         # IMPORTANT: returns raw LOGITS, not probabilities.
#         # Do NOT put a softmax here. nn.CrossEntropyLoss applies
#         # log_softmax internally; adding your own softmax first is a
#         # common bug that makes gradients tiny and training crawl.
#         # For probabilities at inference: torch.softmax(logits, dim=1).


# # =====================================================================
# # 2. ResNet18, fine-tuned
# # =====================================================================
# #
# # TRANSFER LEARNING IN ONE PARAGRAPH
# #   ResNet18's early layers already know edges, textures and shapes from
# #   1.2M ImageNet photos. Those features transfer to galaxies even though
# #   ImageNet contains no galaxies. So we keep the learned body, replace the
# #   final 1000-class layer with a fresh 10-class one, and train.
# #
# # TWO STRATEGIES
# #   freeze_backbone=True   Only the new final layer trains. Fast, needs
# #                          little data, but limited ceiling. Good first run.
# #   freeze_backbone=False  Everything trains, at a SMALL learning rate
# #                          (~1e-4). Usually the best final accuracy. Use a
# #                          large LR here and you will wreck the pretrained
# #                          weights in the first few steps.
# #
# #   A good recipe: freeze first, train a few epochs, then unfreeze and
# #   continue at a low LR.


# def build_resnet18(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
#     weights = models.ResNet18_Weights.IMAGENET1K_V1
#     model = models.resnet18(weights=weights)

#     if freeze_backbone:
#         # requires_grad=False -> no gradients computed, no updates applied.
#         for param in model.parameters():
#             param.requires_grad = False

#     # model.fc is the final Linear(512, 1000). Replacing it with a new
#     # module gives fresh weights that have requires_grad=True by default,
#     # so the head trains even when everything else is frozen.
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model


# # =====================================================================
# # WHERE IS BACKPROPAGATION?
# # =====================================================================
# # There is no backward() in this file, and there should not be. PyTorch
# # DERIVES the backward pass from your forward pass automatically. This is
# # autograd, and it is the central idea of the framework.
# #
# # THE MECHANISM
# #   Every nn.Parameter (Conv2d.weight, Linear.weight, ...) has
# #   requires_grad=True. Whenever an op touches a tensor that requires grad,
# #   PyTorch does not merely compute the result - it records WHICH op
# #   produced it and WHAT the inputs were. So running forward() builds a
# #   graph of the computation as a side effect. You can see it on any
# #   output tensor's .grad_fn attribute.
# #
# #   loss.backward() then walks that graph backwards from the scalar loss,
# #   applies the chain rule at each node (every node knows its own local
# #   derivative), and deposits d(loss)/d(p) into p.grad for every parameter
# #   it reaches.
# #
# # THE DIVISION OF LABOUR - four distinct jobs, easily confused:
# #   model                 holds parameters; each has an empty .grad slot
# #   loss.backward()       FILLS every .grad
# #   optimizer.step()      READS .grad and updates the weights
# #   optimizer.zero_grad() CLEARS .grad (they ACCUMULATE otherwise)
# #
# #   The optimizer never talks to the model. It was handed the parameter
# #   list at construction and just walks it reading .grad. That is why
# #   forgetting zero_grad() is so insidious: nothing errors, the gradients
# #   silently pile up across steps.
# #
# # You CAN hand-write a backward, via torch.autograd.Function, but only for
# # genuinely novel ops. For anything built from standard layers, never.


# # =====================================================================
# # sanity check: run `python src/model.py`
# # =====================================================================
# if __name__ == "__main__":
#     # A fake batch of 4 images. Always test with fake data first - it
#     # catches shape bugs in one second instead of after a 20-minute load.
#     dummy = torch.randn(4, 3, 128, 128)

#     for n in (1, 2):
#         m = SimpleCNN(n_convs=n)
#         print(f"SimpleCNN(n_convs={n}) out: {tuple(m(dummy).shape)} | "
#               f"params {sum(p.numel() for p in m.parameters()):,}")

#     m = SimpleCNN()
#     # Resolution-independence: works at 224 too, thanks to AdaptiveAvgPool2d
#     print("SimpleCNN out @224:", tuple(m(torch.randn(2, 3, 224, 224)).shape))

#     # ---- autograd demo: gradients appear without any backward() ------
#     w = m.features[0][0].weight        # first conv kernel
#     print("\ngrad BEFORE backward:", w.grad)

#     out = m(dummy)
#     print("out.grad_fn         :", out.grad_fn)      # the graph exists
#     loss = nn.CrossEntropyLoss()(out, torch.tensor([1, 7, 3, 0]))
#     loss.backward()                                  # walks it in reverse
#     print("grad AFTER backward :", tuple(w.grad.shape),
#           "| mean abs", f"{w.grad.abs().mean():.4f}")

#     dummy224 = torch.randn(4, 3, 224, 224)
#     r = build_resnet18(freeze_backbone=True)
#     print("\nResNet18 out:", tuple(r(dummy224).shape))      # expect (4, 10)
#     trainable = sum(p.numel() for p in r.parameters() if p.requires_grad)
#     print("ResNet18 trainable params:", trainable)   # expect 5130 = 512*10 + 10







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