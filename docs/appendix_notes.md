# Appendix Notes

This file records the main supplementary material removed from the camera-ready paper. It keeps the method derivations, complexity analysis, and hyperparameter settings.

## A. High-Rank Capacity of SeMi-LoRA

### Notation

For a frozen linear layer, let:

```text
W_0 in R^{d_out x d_in}
x   in R^{d_in}
Delta W in R^{d_out x d_in}
```

SeMi-LoRA splits the feature space into `c` channels. For each channel:

```text
g = d_in / c
k = d_out / c
A_i in R^{r x g}
B_i in R^{k x r}
R_i in R^{r x r}
C_t in R^{c x c}, for t = 1, ..., r
```

Here `r` is the base rank per channel and `c` is the channel number.

The block-diagonal compression and decompression matrices are:

```text
A^m = diag(A_1, ..., A_c) in R^{cr x d_in}
B^m = diag(B_1, ..., B_c) in R^{d_out x cr}
```

The Rank Mix matrix is also block diagonal:

```text
R^m = diag(R_1, ..., R_c) in R^{cr x cr}
```

The Channel Mix matrix `C^m in R^{cr x cr}` mixes channels at each fixed rank index. Its `(i, j)` block is diagonal:

```text
C^m_{i,j} = diag(C_{:,i,j}) in R^{r x r}
```

The residual Rank Mix and Channel Mix correspond to multiplying by `(R^m + I)` and `(C^m + I)`.

### Mergeable Update

Because every stage is linear, SeMi-LoRA can be written as a single update matrix:

```text
Delta W = B^m (C^m + I) (R^m + I) A^m.
```

This update can be merged into the base weight:

```text
W = W_0 + Delta W.
```

Therefore, SeMi-LoRA preserves the same zero-latency inference property as LoRA after merging.

### Rank Bound

We use the following standard rank facts:

```text
rank(A) <= min(a, b), for A in R^{a x b}
rank(AB) <= min(rank(A), rank(B))
rank(A + B) <= rank(A) + rank(B)
rank(diag(M_1, ..., M_n)) = sum_i rank(M_i)
```

For SeMi-LoRA:

```text
rank(Delta W)
= rank(B^m (C^m + I) (R^m + I) A^m)
<= min(rank(B^m), rank(C^m + I), rank(R^m + I), rank(A^m)).
```

Since `A^m` and `B^m` are block diagonal:

```text
rank(A^m) = sum_{i=1}^c rank(A_i)
rank(B^m) = sum_{i=1}^c rank(B_i)
```

Under the standard low-rank setting `r << min(g, k)` and assuming the mini adapters are full rank in their low-dimensional subspaces:

```text
rank(A_i) = rank(B_i) = r
rank(A^m) = rank(B^m) = c r
```

The residual mixers satisfy:

```text
R^m + I in R^{cr x cr}
C^m + I in R^{cr x cr}
rank(R^m + I) <= cr
rank(C^m + I) <= cr
```

Since `I` is full rank, these residual mixers are typically full rank unless the learned matrices produce degenerate cancellations, such as `R^m = -I` or `C^m = -I`.

Combining the above:

```text
rank(Delta W) <= min(d_in, d_out, c r).
```

This is a rank-capacity statement. It shows that increasing the channel number `c` expands the maximum rank capacity of the update while keeping the base rank `r` fixed. It does not claim that the realized effective rank must always increase during training.

## B. Complete Update Pattern

SeMi-LoRA expands as:

```text
Delta W
= B^m (C^m + I) (R^m + I) A^m
= B^m (C^m R^m + R^m + C^m + I) A^m
= B^m C^m R^m A^m
 + B^m R^m A^m
 + B^m C^m A^m
 + B^m A^m.
```

The term `B^m A^m` is the independent block-wise mini-LoRA update:

```text
B^m A^m = diag(B_1 A_1, ..., B_c A_c).
```

This is block diagonal, so it only updates local channel groups.

The leading term `B^m C^m R^m A^m` introduces cross-channel blocks. Under the output-channel by input-channel block view, its `(i, j)` block has the form:

```text
B_i diag(C_{:,i,j}) R_j A_j.
```

Therefore:

```text
B^m C^m R^m A^m =
[
  B_1 diag(C_{:,1,1}) R_1 A_1   ...   B_1 diag(C_{:,1,c}) R_c A_c
  ...
  B_c diag(C_{:,c,1}) R_1 A_1   ...   B_c diag(C_{:,c,c}) R_c A_c
]
```

Here `C in R^{r x c x c}` stores channel-mixing weights, and `C_{:,i,j}` is used to build a diagonal matrix for mixing from source channel `j` to target channel `i` at each rank index.

When off-diagonal channel weights are non-zero, the update matrix is no longer restricted to independent blocks. This enables cross-group interactions and supports a dense, complete update pattern over `W`.

### Relation to MELoRA

If both mixers are disabled:

```text
R^m = 0
C^m = 0
```

then:

```text
Delta W = B^m A^m = diag(B_1 A_1, ..., B_c A_c).
```

This recovers a MELoRA-like block-diagonal update. Even if the total rank is large, the update remains local because input and output groups do not interact across channels.

### Relation to LoRA

Standard LoRA uses:

```text
Delta W = B A
A in R^{r x d_in}
B in R^{d_out x r}
```

Partition `A` by input channels and `B` by output channels:

```text
A = [A_1, ..., A_c]
B = [B_1^T, ..., B_c^T]^T
```

where:

```text
A_j in R^{r x g}
B_i in R^{k x r}
```

If `R^m = 0` and Channel Mix is configured so that off-diagonal blocks pass the shared rank space across channels, then `B^m (C^m + I) A^m` produces all blocks:

```text
(i, j) block = B_i A_j.
```

Thus SeMi-LoRA can recover the dense LoRA update under a suitable Channel Mix construction. In this sense, both MELoRA-like local updates and LoRA-like dense updates are contained in the SeMi-LoRA parameterization.

## D. Complexity and Efficiency Analysis

### Setup

We evaluate the training overhead of one adapted linear layer:

```text
W_0 in R^{d_out x d_in}
d_in ~= d_out ~= d
```

Rank `r`, channel number `c`, and MELoRA group number `n` are treated as independent hyperparameters. FLOPs count matrix-vector multiplications as `2mn` for an `m x n` matrix, ignoring lower-order element-wise operations.

### LoRA

LoRA parameterizes:

```text
Delta W = B A
A in R^{r x d_in}
B in R^{d_out x r}
```

Trainable parameters:

```text
N_LoRA = r(d_in + d_out)
```

FLOPs:

```text
FLOPs_LoRA ~= 2r(d_in + d_out)
```

### MELoRA

MELoRA splits the feature dimensions into `n` independent groups. For each group:

```text
A_i in R^{r x (d_in / n)}
B_i in R^{(d_out / n) x r}
```

Trainable parameters:

```text
N_MELoRA
= sum_{i=1}^n [r(d_in / n) + r(d_out / n)]
= r(d_in + d_out)
```

FLOPs:

```text
FLOPs_MELoRA
~= n * 2r(d_in / n + d_out / n)
= 2r(d_in + d_out)
```

MELoRA has LoRA-like asymptotic cost, but its block-diagonal structure restricts cross-group information flow.

### VeRA

VeRA freezes a shared random high-rank pair `(A, B)` and trains layer-specific scaling vectors:

```text
b in R^{d_out}
d in R^r
```

Trainable parameters:

```text
N_VeRA = d_out + r ~= O(d)
```

FLOPs:

```text
FLOPs_VeRA ~= 2r(d_in + d_out)
```

Although VeRA is parameter efficient, it often uses a larger rank to preserve expressivity, which increases compute.

### HiRA

HiRA applies a Hadamard update:

```text
W_new = W_0 + W_0 o (B A)
```

where `o` denotes element-wise multiplication.

Unlike LoRA, this update cannot generally be evaluated as `B(Ax)` because of the element-wise product with `W_0`. The dense matrix `BA` must be materialized during training.

Training overhead:

```text
FLOPs_HiRA
= 2 r d_in d_out + d_in d_out
~= O(r d^2)
```

This quadratic dependence makes HiRA computationally heavier for large models.

### SeMi-LoRA

SeMi-LoRA keeps the same compression/decompression pathway as LoRA and adds latent-space mixers:

```text
N_SeMi-LoRA = r(d_in + d_out) + c r^2 + r c^2
```

where:

```text
r(d_in + d_out): compression/decompression
c r^2:           Rank Mix
r c^2:           Channel Mix
```

FLOPs:

```text
FLOPs_SeMi-LoRA
~= 2r(d_in + d_out) + 2c r^2 + 2r c^2
```

Since the extra terms operate only in the compact `c x r` latent space, the overhead is small when `d` is large.

### LLaMA2-7B Layer-Level Example

For a representative LLaMA2-7B layer with:

```text
d = 4096
LoRA rank r = 32
VeRA rank r = 256
SeMi-LoRA channels c = 8
```

the per-layer training overhead is:

| Method | Config | Trainable params | Param order | FLOPs | FLOP order |
|---|---:|---:|---:|---:|---:|
| LoRA | r = 32 | 262k | O(rd) | 524k | O(rd) |
| MELoRA | r = 32, n = 8 | 262k | O(rd) | 524k | O(rd) |
| VeRA | r = 256 | 4.4k | O(d) | 4.2M | O(rd) |
| HiRA | r = 32 | 262k | O(rd) | ~1.07G | O(rd^2) |
| SeMi-LoRA | r = 32, c = 8 | 272k (+3.9%) | O(rd) | 545k (+3.9%) | O(rd) |

This explains why SeMi-LoRA can add global feature mixing while retaining LoRA-like training cost.

## E. Hyperparameter Settings

### GLUE

Common settings:

```text
Optimizer: AdamW
Warmup ratio: 0.06
LR schedule: Linear
Max sequence length: 256
```

Task-specific settings:

| Setting | Hyperparameter | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B |
|---|---|---:|---:|---:|---:|---:|---:|
| Rank 2, channels 4 | Epochs | 20 | 30 | 50 | 20 | 50 | 50 |
| Rank 2, channels 4 | Learning rate | 3e-4 | 5e-4 | 7e-4 | 9e-4 | 9e-4 | 3e-4 |
| Rank 2, channels 4 | Alpha | 16 | 32 | 16 | 8 | 16 | 32 |
| Rank 2, channels 4 | Batch size | 32 | 32 | 64 | 64 | 128 | 64 |
| Rank 8, channels 8 | Epochs | 20 | 30 | 50 | 20 | 50 | 50 |
| Rank 8, channels 8 | Learning rate | 7e-4 | 5e-4 | 5e-4 | 4e-4 | 5e-4 | 3e-4 |
| Rank 8, channels 8 | Alpha | 16 | 16 | 32 | 8 | 32 | 32 |
| Rank 8, channels 8 | Batch size | 64 | 32 | 128 | 32 | 128 | 128 |

### LLM Tuning

| Hyperparameter | Commonsense / ConvAI | Math |
|---|---:|---:|
| Optimizer | AdamW | AdamW |
| LR scheduler | Linear | Linear |
| Rank | 32 | 128 |
| Alpha | 64 | 256 |
| Learning rate | 2e-4 | 2e-5 |
| Batch size | 16 | 16 |
| Warmup steps / ratio | 100 steps | 0.03 ratio |
| Epochs | 3 | 1 |
| Placement | Q, K, V, UP, Down | All projection modules |

For concrete commands, see the task scripts and the repository README.
