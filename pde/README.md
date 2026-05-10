# Case Study A — Architectural DB for Reaction-Diffusion PDEs

This directory contains the code for the PDE case study: an *architectural* diagrammatic bias (DB-Fast) for a 2D reaction-diffusion surrogate, evaluated against three parameter-matched baselines.

## What's here

| File | Purpose |
|---|---|
| `experiment.py` | Main three-stage protocol: CNN / MaskedCNN / UNet-1L / DB-Fast at matched parameter count (~593k). Produces the slope, OOD-average, and worst-OOD numbers in the report. |
| `rd_unet2l_n16.py` | Extended ablation that adds UNet-2L (deeper bottleneck) to the comparison. |

## The problem

2D reaction-diffusion equation on Ω = (0,1)² with homogeneous Dirichlet BCs:

$$-D \Delta u + \sigma u = f, \qquad D = 0.01, \ \sigma = 1.0$$

Source terms `f` are random superpositions of four isotropic Gaussian bumps. Ground truth comes from the standard 5-point finite-difference discretization. We train at `N_train = 12` and evaluate on a sequence of OOD grids `N ∈ {12, …, 24}`.

## DB-Fast in one paragraph

Each layer decomposes a standard 3×3 convolution into two masked depthwise sub-operations:
- **`DBEdgeConv`** aggregates only the four axis-aligned neighbours (the cross of the 3×3 kernel — exactly the support of the 5-point Laplacian).
- **`DBTriConv`** aggregates only the four diagonal neighbours, modulated by a learned per-channel sigmoid gate.

The masks are fixed binary tensors independent of `N`, so the inductive bias persists exactly at OOD grid sizes — which is what motivates the flatter-generalization-slope hypothesis.

## Primary metric

Log-log slope of relative-ℓ₂ error vs. grid size `N`. Flatter (smaller) is better. This disentangles in-distribution skill from how fast the error grows as `N` increases.

## Quick start

```bash
# from the repo root
cd pde
pip install -r requirements.txt
python experiment.py                # default settings, single seed
```

That's it — the script generates its own training data, trains all four models, and writes results to the current directory.

## Running the full protocol

```bash
python experiment.py --seeds 0 1 --epochs 200 --db_epochs 400
```

Key flags (see `experiment.py` for the full list):
- `--seeds`: list of integer seeds (Stage 2 in the report uses `0 1`; Stage 3 uses `2`).
- `--epochs` / `--db_epochs`: training budget for baselines / DB-Fast respectively.
- `--n_train_samples`, `--n_val_samples`, `--n_eval_samples`, `--batch_size`.

Output: a `final_info.json` (in AI-Scientist format) plus per-seed rel-ℓ₂ curves and figures.

## Reproducing report numbers

| Run | Seeds | Eval grids |
|---|---|---|
| Stage 1 — Seq A | `[0]` | `[12, 14, 19, 24]` |
| Stage 2A — Seq A | `[0, 1]` | `[12, 14, 19, 24]` |
| Stage 2B — Seq B | `[0, 1]` | `[12, 16, 20, 23]` |
| Stage 3 — Seq C | `[2]` | `[12, 15, 18, 22]` |

DB-Fast achieves the lowest log-log slope on the seed mean in every run; the gap to all three baselines grows in Stage 3.
