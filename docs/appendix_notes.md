# Appendix Notes

This file records supplementary material moved out of the paper main text.

## Rank Capacity

SeMi-LoRA represents the update matrix as:

```text
Delta W = B^m (C^m + I) (R^m + I) A^m
```

where `A^m` and `B^m` are block-diagonal compression and decompression matrices, `R^m` is the Rank Mix matrix, and `C^m` is the Channel Mix matrix.

For any feasible parameters,

```text
rank(Delta W)
<= min(rank(B^m), rank(C^m + I), rank(R^m + I), rank(A^m)).
```

Since `A^m` and `B^m` are block diagonal, their ranks are the sums of the ranks of all channel-wise blocks. If each mini adapter has rank `r` and there are `c` channels, then:

```text
rank(A^m) = rank(B^m) = c r.
```

The residual mixers `(R^m + I)` and `(C^m + I)` operate in the same `c r`-dimensional latent space. Except for degenerate cancellations, they do not reduce the latent rank. Therefore:

```text
rank(Delta W) <= min(d_in, d_out, c r).
```

This is a rank-capacity statement: increasing the number of channels expands the maximum update rank capacity while keeping the base rank fixed.

## Complete Update

Expanding the update gives:

```text
Delta W
= B^m C^m R^m A^m
 + B^m R^m A^m
 + B^m C^m A^m
 + B^m A^m.
```

The final term `B^m A^m` corresponds to independent block-wise mini-LoRA updates. The leading cross-mixing term `B^m C^m R^m A^m` introduces non-zero off-diagonal channel blocks, enabling cross-channel interactions and a dense update pattern.

This distinguishes SeMi-LoRA from block-diagonal high-rank variants such as MELoRA, whose update is restricted to local independent groups.

## Complexity

For one adapted linear layer with input/output dimensions approximately `d`, LoRA has:

```text
Parameters: r(d_in + d_out)
FLOPs:      2r(d_in + d_out)
```

SeMi-LoRA adds latent mixing:

```text
Parameters: r(d_in + d_out) + c r^2 + r c^2
FLOPs:      2r(d_in + d_out) + 2c r^2 + 2r c^2
```

For LLaMA2-7B with `d=4096`, `r=32`, and `c=8`, the per-layer overhead is about `+3.9%` over standard LoRA.

## Hyperparameters Used in the Paper

### Commonsense Reasoning

```text
Optimizer: AdamW
Scheduler: Linear
Rank: 32
Alpha: 64
Learning rate: 2e-4
Batch size: 16
Warmup steps: 100
Epochs: 3
Target modules: q_proj, k_proj, v_proj, up_proj, down_proj
```

### Math Reasoning

```text
Optimizer: AdamW
Scheduler: Linear
Rank: 128
Alpha: 256
Learning rate: 2e-5
Batch size: 16
Warmup ratio: 0.03
Epochs: 1
Target modules: all linear projection modules
```

### GLUE

Common settings:

```text
Optimizer: AdamW
Warmup ratio: 0.06
Scheduler: Linear
Max sequence length: 256
```

Representative SeMi-LoRA setting:

```text
Rank x channels: 8 x 8
```

Task-specific learning rates and batch sizes should follow the scripts or paper settings.
