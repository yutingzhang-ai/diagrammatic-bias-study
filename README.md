# Diagrammatic Inductive Biases in Neural Networks

**When do they help, and why?** A unified empirical study across PDE surrogates and constraint satisfaction.

This repository accompanies the report *Diagrammatic Inductive Biases in Neural Networks: When Do They Help, and Why?* (May 2026). It contains the full code for two case studies that probe a single conditional principle:

> A diagrammatic inductive bias yields a measurable improvement to the extent that the relational structure it encodes is not already captured by another component of the model.

## The two case studies

| | **Case Study A** ([`pde/`](pde/)) | **Case Study B** ([`sudoku/`](sudoku/)) |
|---|---|---|
| **Domain** | 2D reaction-diffusion PDE surrogate | 4×4 Sudoku completion |
| **DB instantiation** | Architectural (masked convolutions) | Loss-based (triangle consistency) |
| **Question** | Does the full method beat the partial prior? | Does the partial prior compose with the full method? |
| **Headline** | DB-Fast achieves the flattest log-log generalization slope in every run | TF+DB is the strongest model; TF+GT+DB fails to compose, underperforming DB alone |

## Three-tier structural-prior hierarchy

Both studies share the same experimental skeleton: a plain baseline, a *partial* structural prior, and the full diagrammatic method.

| Tier | PDE study | Sudoku study |
|---|---|---|
| Plain baseline (no structural prior) | CNN | TF |
| Partial structural prior | MaskedCNN | TF+GT |
| Full diagrammatic method | DB-Fast | TF+DB |
| Combined (partial + full) | — | TF+GT+DB |

The PDE study finds the partial prior **cannot absorb** the diagrammatic decomposition (full method beats partial prior). The Sudoku study finds the partial prior **fails to compose** with the diagrammatic method (combined model underperforms full method alone). Together, both directions support the conditional principle.

## Repository layout

```
.
├── README.md              ← this file
├── report.pdf             ← the accompanying report
├── pde/                   ← Case Study A: architectural DB on reaction-diffusion PDEs
│   ├── README.md
│   ├── experiment.py            # main multi-seed protocol (CNN / MaskedCNN / UNet-1L / DB-Fast)
│   └── rd_unet2l_n16.py         # extended ablation including UNet-2L
└── sudoku/                ← Case Study B: loss-based DB on 4×4 Sudoku
    ├── README.md
    ├── Sudoku_GT_DB_Colab.ipynb       # main TF / TF+DB / TF+GT / TF+GT+DB ablation
    └── Sudoku_GT_DB_5models_ablation.ipynb  # extended ablation
```

The two subdirectories are intentionally independent: each can be run in isolation, with its own dependencies and entry point. See the per-study READMEs for setup instructions.

## Reproducing the reported results

- **PDE study:** see [`pde/README.md`](pde/README.md). Stage 2 / Stage 3 results in the report come from `experiment.py` with the seed configurations listed there.
- **Sudoku study:** see [`sudoku/README.md`](sudoku/README.md). The four-cell ablation in Table 1 of the report comes from `Sudoku_GT_DB_Colab.ipynb`.

## Citation

If you use this code, please cite the report and the two underlying references on Diagrammatic Backpropagation:

```bibtex
@misc{mahadevan2024gaia,
  title={GAIA: Categorical foundations of generative AI},
  author={Mahadevan, Sridhar},
  year={2024},
  eprint={2402.18732},
  archivePrefix={arXiv}
}

@book{mahadevan2025categories,
  title={Categories for AGI},
  author={Mahadevan, Sridhar},
  year={2025},
  note={Course textbook}
}
```

## License

[choose one — MIT is the usual default for course/research code]
