# Appendix Notes

This file records supplementary material removed from the camera-ready paper. It keeps the method derivations, complexity analysis, and hyperparameter settings.

## High-Rank Capacity of SeMi-LoRA

### Notation

For a frozen linear layer, let $W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, $x \in \mathbb{R}^{d_{\text{in}}}$, and $\Delta W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$.

SeMi-LoRA splits the feature space into $c$ channels. For each channel, let

$$
g=\frac{d_{\text{in}}}{c},\quad
k=\frac{d_{\text{out}}}{c},\quad
A_i\in\mathbb{R}^{r\times g},\quad
B_i\in\mathbb{R}^{k\times r},\quad
R_i\in\mathbb{R}^{r\times r}.
$$

The Channel Mix matrices are $C_t\in\mathbb{R}^{c\times c}$ for $t=1,\ldots,r$. Here $r$ is the base rank per channel and $c$ is the number of channels.

The block-diagonal compression and decompression matrices are

$$
A^m=\mathrm{diag}(A_1,\ldots,A_c)\in\mathbb{R}^{cr\times d_{\text{in}}},
\qquad
B^m=\mathrm{diag}(B_1,\ldots,B_c)\in\mathbb{R}^{d_{\text{out}}\times cr}.
$$

The Rank Mix matrix is also block diagonal:

$$
R^m=\mathrm{diag}(R_1,\ldots,R_c)\in\mathbb{R}^{cr\times cr}.
$$

The Channel Mix matrix $C^m\in\mathbb{R}^{cr\times cr}$ mixes channels at each fixed rank index. Its $(i,j)$ block is diagonal:

$$
C^m_{i,j}=\mathrm{diag}(C_{:,i,j})\in\mathbb{R}^{r\times r}.
$$

The residual Rank Mix and Channel Mix correspond to multiplying by $(R^m+I)$ and $(C^m+I)$.

### Mergeable Update

Because every stage is linear, SeMi-LoRA can be written as a single update matrix:

$$
\Delta W = B^m(C^m+I)(R^m+I)A^m.
$$

This update can be merged into the base weight:

$$
W = W_0+\Delta W.
$$

Therefore, SeMi-LoRA preserves LoRA's no-extra-latency inference property after merging.

### Rank Bound

We use the following standard rank facts:

$$
\mathrm{rank}(A)\leq \min(a,b),\quad A\in\mathbb{R}^{a\times b},
$$

$$
\mathrm{rank}(AB)\leq \min(\mathrm{rank}(A),\mathrm{rank}(B)),
$$

$$
\mathrm{rank}(A+B)\leq \mathrm{rank}(A)+\mathrm{rank}(B),
$$

$$
\mathrm{rank}(\mathrm{diag}(M_1,\ldots,M_n))
=\sum_{i=1}^{n}\mathrm{rank}(M_i).
$$

For SeMi-LoRA,

$$
\begin{aligned}
\mathrm{rank}(\Delta W)
&=\mathrm{rank}\!\left(B^m(C^m+I)(R^m+I)A^m\right)\\
&\leq \min\{\mathrm{rank}(B^m),\mathrm{rank}(C^m+I),
\mathrm{rank}(R^m+I),\mathrm{rank}(A^m)\}.
\end{aligned}
$$

Since $A^m$ and $B^m$ are block diagonal,

$$
\mathrm{rank}(A^m)=\sum_{i=1}^{c}\mathrm{rank}(A_i),
\qquad
\mathrm{rank}(B^m)=\sum_{i=1}^{c}\mathrm{rank}(B_i).
$$

Under the standard low-rank setting $r\ll\min(g,k)$ and assuming the mini adapters are full rank in their low-dimensional subspaces,

$$
\mathrm{rank}(A_i)=\mathrm{rank}(B_i)=r,
\qquad
\mathrm{rank}(A^m)=\mathrm{rank}(B^m)=cr.
$$

The residual mixers satisfy

$$
R^m+I\in\mathbb{R}^{cr\times cr},\quad
C^m+I\in\mathbb{R}^{cr\times cr},
$$

and therefore

$$
\mathrm{rank}(R^m+I)\leq cr,\qquad
\mathrm{rank}(C^m+I)\leq cr.
$$

For generic mixer parameters, the residual mixers remain full rank. They can lose rank only under degenerate cancellations, such as $R^m=-I$ or $C^m=-I$.

Combining the above,

$$
\mathrm{rank}(\Delta W)\leq \min(d_{\text{in}},d_{\text{out}},cr).
$$

This is a rank-capacity statement. It shows that increasing the channel number $c$ expands the maximum rank capacity of the update while keeping the base rank $r$ fixed. It does not claim that the realized effective rank must always increase during training.

## Complete Update Pattern

SeMi-LoRA expands as

$$
\begin{aligned}
\Delta W
&=B^m(C^m+I)(R^m+I)A^m\\
&=B^m(C^mR^m+R^m+C^m+I)A^m\\
&=B^mC^mR^mA^m+B^mR^mA^m+B^mC^mA^m+B^mA^m.
\end{aligned}
$$

The term $B^mA^m$ is the independent block-wise mini-LoRA update:

$$
B^mA^m=\mathrm{diag}(B_1A_1,\ldots,B_cA_c).
$$

This is block diagonal, so it only updates local channel groups.

The leading term $B^mC^mR^mA^m$ introduces cross-channel blocks. Under the output-channel by input-channel block view, its $(i,j)$ block is

$$
B_i\mathrm{diag}(C_{:,i,j})R_jA_j.
$$

Therefore,

$$
B^mC^mR^mA^m=
\begin{bmatrix}
B_1D_{1,1}R_1A_1 & \cdots & B_1D_{1,c}R_cA_c\\
\vdots & \ddots & \vdots\\
B_cD_{c,1}R_1A_1 & \cdots & B_cD_{c,c}R_cA_c
\end{bmatrix},
\quad
D_{i,j}=\mathrm{diag}(C_{:,i,j}).
$$

Here $C\in\mathbb{R}^{r\times c\times c}$ stores channel-mixing weights, and $C_{:,i,j}$ is used to build a diagonal matrix for mixing from source channel $j$ to target channel $i$ at each rank index.

When off-diagonal channel weights are non-zero, the update matrix is no longer structurally restricted to independent blocks. This enables cross-group interactions and allows the update to cover all channel blocks of $W$.

### Relation to MELoRA

If both mixers are disabled, i.e., $R^m=0$ and $C^m=0$, then

$$
\Delta W=B^mA^m=\mathrm{diag}(B_1A_1,\ldots,B_cA_c).
$$

This recovers a MELoRA-like block-diagonal update. Even if the total rank is large, the update remains local because input and output groups do not interact across channels.

### Relation to LoRA

Standard LoRA uses $\Delta W=BA$ with $A\in\mathbb{R}^{r\times d_{\text{in}}}$ and $B\in\mathbb{R}^{d_{\text{out}}\times r}$. Partition $A$ by input channels and $B$ by output channels:

$$
A=[A_1,\ldots,A_c],
\qquad
B=[B_1^\top,\ldots,B_c^\top]^\top,
$$

where $A_j\in\mathbb{R}^{r\times g}$ and $B_i\in\mathbb{R}^{k\times r}$.

Set $R^m=0$. Configure Channel Mix so that

$$
\mathrm{diag}(C_{:,i,j})=I_r\quad (i\neq j),
\qquad
\mathrm{diag}(C_{:,i,i})=0.
$$

Because $(C^m+I)$ contributes identity blocks on the diagonal, $B^m(C^m+I)A^m$ then produces every block:

$$
\text{block}(i,j)=B_iA_j.
$$

Thus SeMi-LoRA can recover the dense LoRA update under a suitable Channel Mix construction. In this sense, both MELoRA-like local updates and LoRA-like dense updates are contained in the SeMi-LoRA parameterization.

## Complexity and Efficiency Analysis

### Setup

We evaluate the training overhead of one adapted linear layer with $W_0\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ and $d_{\text{in}}\approx d_{\text{out}}\approx d$. Rank $r$, channel number $c$, and MELoRA group number $n$ are treated as independent hyperparameters. FLOPs count matrix-vector multiplications as $2mn$ for an $m\times n$ matrix, ignoring lower-order element-wise operations.

### LoRA

LoRA parameterizes $\Delta W=BA$, where $A\in\mathbb{R}^{r\times d_{\text{in}}}$ and $B\in\mathbb{R}^{d_{\text{out}}\times r}$.

Trainable parameters:

$$
N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}}).
$$

FLOPs:

$$
\mathrm{FLOPs}_{\text{LoRA}}\approx 2r(d_{\text{in}}+d_{\text{out}}).
$$

### MELoRA

MELoRA splits the feature dimensions into $n$ independent groups. For each group, $A_i\in\mathbb{R}^{r\times(d_{\text{in}}/n)}$ and $B_i\in\mathbb{R}^{(d_{\text{out}}/n)\times r}$.

Trainable parameters:

$$
\begin{aligned}
N_{\text{MELoRA}}
&=\sum_{i=1}^{n}\left(r\frac{d_{\text{in}}}{n}+r\frac{d_{\text{out}}}{n}\right)\\
&=r(d_{\text{in}}+d_{\text{out}}).
\end{aligned}
$$

FLOPs:

$$
\mathrm{FLOPs}_{\text{MELoRA}}
\approx n\cdot 2r\left(\frac{d_{\text{in}}}{n}+\frac{d_{\text{out}}}{n}\right)
=2r(d_{\text{in}}+d_{\text{out}}).
$$

MELoRA has LoRA-like asymptotic cost, but its block-diagonal structure restricts cross-group information flow.

### VeRA

VeRA freezes a shared random high-rank pair $(A,B)$ and trains layer-specific scaling vectors $b\in\mathbb{R}^{d_{\text{out}}}$ and $d\in\mathbb{R}^r$.

Trainable parameters:

$$
N_{\text{VeRA}}=d_{\text{out}}+r\approx O(d).
$$

FLOPs:

$$
\mathrm{FLOPs}_{\text{VeRA}}\approx 2r(d_{\text{in}}+d_{\text{out}}).
$$

Although VeRA is parameter efficient, it often uses a larger rank to preserve expressivity, which increases compute.

### HiRA

HiRA applies a Hadamard update:

$$
W_{\text{new}}=W_0+W_0\odot(BA).
$$

Unlike LoRA, this update cannot generally be evaluated as $B(Ax)$ because of the element-wise product with $W_0$. The dense matrix $BA$ must be materialized during training.

Training overhead:

$$
\mathrm{FLOPs}_{\text{HiRA}}
=2rd_{\text{in}}d_{\text{out}}+d_{\text{in}}d_{\text{out}}
\approx O(rd^2).
$$

This quadratic dependence makes HiRA computationally heavier for large models.

### SeMi-LoRA

SeMi-LoRA keeps the same compression/decompression pathway as LoRA and adds latent-space mixers.

Trainable parameters:

$$
N_{\text{SeMi-LoRA}}
=r(d_{\text{in}}+d_{\text{out}})+cr^2+rc^2.
$$

Here $r(d_{\text{in}}+d_{\text{out}})$ comes from compression/decompression, $cr^2$ from Rank Mix, and $rc^2$ from Channel Mix.

FLOPs:

$$
\mathrm{FLOPs}_{\text{SeMi-LoRA}}
\approx 2r(d_{\text{in}}+d_{\text{out}})+2cr^2+2rc^2.
$$

Since the extra terms operate only in the compact $c\times r$ latent space, the overhead is small when $d$ is large.

### LLaMA2-7B Layer-Level Example

For a representative LLaMA2-7B layer, use $d=4096$, LoRA rank $r=32$, VeRA rank $r=256$, and SeMi-LoRA channels $c=8$.

| Method | Config | Trainable params | Param order | FLOPs | FLOP order |
|---|---|---:|---|---:|---|
| LoRA | $r=32$ | 262k | $O(rd)$ | 524k | $O(rd)$ |
| MELoRA | $r=32,n=8$ | 262k | $O(rd)$ | 524k | $O(rd)$ |
| VeRA | $r=256$ | 4.4k | $O(d)$ | 4.2M | $O(rd)$ |
| HiRA | $r=32$ | 262k | $O(rd)$ | ~1.09G | $O(rd^2)$ |
| SeMi-LoRA | $r=32,c=8$ | 272k (+3.9%) | $O(rd)$ | 545k (+3.9%) | $O(rd)$ |

This explains why SeMi-LoRA can add global feature mixing while retaining LoRA-like training cost.

## Hyperparameter Settings

### GLUE

Common settings:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Warmup ratio | 0.06 |
| LR schedule | Linear |
| Max sequence length | 256 |

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
