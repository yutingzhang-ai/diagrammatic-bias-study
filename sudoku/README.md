# Case Study B — Loss-Based DB on 4×4 Sudoku

This directory contains the code for the Sudoku case study: a *loss-based* diagrammatic bias (triangle-consistency penalty) on a Transformer backbone, in a controlled four-cell ablation against a Geometric Transformer architectural prior.

## What's here

| File | Purpose |
|---|---|
| `Sudoku_4Models.ipynb` | Main four-cell ablation: TF / TF+DB / TF+GT / TF+GT+DB. Produces Table 1 and Figure 4 in the report. |

## The problem

4×4 Sudoku completion with `k = 8` revealed cells. Each puzzle is a sequence of 16 cell tokens; the network predicts a digit (0–3) for every cell. The 16 cells partition into 12 constraint groups (4 rows + 4 columns + 4 2×2 boxes), each of size 4 — giving 12 × C(4,3) = **48 triangles** in total. Primary metric: puzzle accuracy (all 16 cells correct).

## The two biases

Both sit on top of the same Transformer backbone (3 blocks, `d = 64`, 4 heads):

**Geometric Transformer (GT) — architectural prior.** A 1-D convolution (kernel 5) is interleaved with each self-attention block, giving the model an explicit local receptive field over the cell sequence. Encodes positional locality, but does *not* enforce row/column/box equivalence.

**Diagrammatic Backpropagation (DB) — loss-based prior.** An auxiliary triangle-consistency loss:

$$\mathcal{L}_{\text{DB}} = \frac{1}{|\mathcal{T}|} \sum_{(i,j,k) \in \mathcal{T}} \big(\|h_i - \bar{m}_{ijk}\|^2 + \|h_j - \bar{m}_{ijk}\|^2 + \|h_k - \bar{m}_{ijk}\|^2\big)$$

where $\bar{m}_{ijk}$ is the centroid of the three final-layer cell embeddings. Total loss is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{DB}}\mathcal{L}_{\text{DB}}$ with $\lambda_{\text{DB}} = 0.05$.

## Three pre-registered hypotheses

If both biases individually help, what happens when combined?

- **H1 (additivity):** Δ_{GT+DB} ≈ Δ_GT + Δ_{DB on TF}
- **H2 (regime dominance):** Δ_{GT+DB} ≈ max(Δ_GT, Δ_{DB on TF})
- **H3 (interference):** Δ_{DB on GT} ≪ Δ_{DB on TF} *and* A(TF+DB) > A(TF+GT+DB)

The data **support H3**: combining DB with GT fails to improve over DB alone, and the gap persists asymptotically.

