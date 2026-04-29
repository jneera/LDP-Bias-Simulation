# 🔐 LDP-Bias-Simulation

**Reproduction code for the formal bias analysis and empirical comparison of Local Differential Privacy mechanisms in bounded-domain recommender systems.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jneera/LDP-Bias-Simulation)

---

This repository contains the complete source code for all experiments reported in:

> Neera, J. & Franca, L. (2026). *The Hidden Cost of Clipping: Privacy Mechanisms and Bounded Domain Bias in Recommender Systems*. Proceedings of ESORICS 2026.

## 📖 Study Overview

Recommendation systems increasingly rely on Local Differential Privacy (LDP) to protect user rating data. A common workaround for unbounded noise (Laplace/Gaussian) is to clip perturbed ratings back to the valid range — but this introduces **systematic estimator bias** that structurally disadvantages users who rate at the extremes of the scale. This paper provides the first formal proof of this bias and the first systematic head-to-head comparison of four LDP mechanisms for collaborative filtering.

### Key findings

| Finding | Detail |
|---|---|
| Bias is non-zero for all finite ε | Clipping bias cannot be eliminated by collecting more data |
| Bias is maximised at boundary ratings | Users giving 1-star or 5-star ratings are disproportionately affected |
| Piecewise mechanism dominates | Lowest RMSE and highest NDCG@10 across all datasets and privacy budgets tested |
| Gaussian performs worst | Despite offering a weaker (ε,δ)-LDP guarantee rather than pure ε-LDP |
| Exception at extreme sparsity | At 20% sparsity and ε=0.1, Bounded Laplace marginally outperforms Piecewise |

### Datasets used

| Dataset | Total Ratings | Users | Items | Rating Scale |
|---|---|---|---|---|
| MovieLens 100K | 100,000 | 943 | 1,682 | 1–5 |
| MovieLens 1M | 1,000,000 | 6,040 | 3,952 | 1–5 |
| Jester | 1.8 million | 24,983 | 100 | −10 to 10 (rescaled to 1–5) |

### Experimental conditions

- **Privacy budgets:** ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
- **Sparsity levels:** natural, 50% subsampled, 20% subsampled
- **Evaluation metrics:** RMSE and NDCG@10
- **Matrix factorization:** Truncated SVD with d = 20 latent factors
- **Validation:** 10-fold cross-validation (random_state=42)

---

## 🗂️ Repository Structure

```
LDP-Bias-Simulation/
├── 100k_Experiment_for_ESORICS.ipynb   # MovieLens 100K experiments (Figs 1, 2, 3)
├── 1M_Experiment_for_ESORICS.ipynb     # MovieLens 1M experiments (Fig 4, Table 2)
├── Jester_Experiment_for_ESORICS.ipynb # Jester experiments (Fig 4, Table 2)
├── Sparsity_Experiment.ipynb           # Sparsity level comparison (Fig 3)
├── mechanisms/                         # Standalone mechanism implementations
│   ├── clipped_laplace.py              # Clipped Laplace mechanism
│   ├── bounded_laplace.py              # Bounded Laplace (rejection sampling)
│   ├── clipped_gaussian.py             # Clipped Gaussian mechanism
│   └── piecewise.py                    # Piecewise mechanism
├── LICENSE
└── README.md
```

---

## 🚀 Reproducing the Analysis

All experiments were implemented in Python using Google Colaboratory. The notebooks are the canonical way to reproduce all results.

### Option A — Google Colab (recommended)

Click the Colab badge at the top of this README. No local setup required.

### Option B — Local (Jupyter)

**Prerequisites:** Python 3.x

```bash
git clone https://github.com/jneera/LDP-Bias-Simulation.git
cd LDP-Bias-Simulation
pip install -r requirements.txt
jupyter notebook
```

Open the relevant notebook for each experiment (see Analysis Sections below).

### Option C — Standalone scripts

The `mechanisms/` folder contains self-contained Python implementations of each LDP mechanism, extracted from the notebooks for reuse in other projects:

```bash
from mechanisms.piecewise import piecewise_mechanism
from mechanisms.bounded_laplace import bounded_laplace_mechanism
from mechanisms.clipped_laplace import clipped_laplace_mechanism
from mechanisms.clipped_gaussian import clipped_gaussian_mechanism
```

---

## 📊 Analysis Sections

| Notebook | Reproduces | Description |
|---|---|---|
| `100k_Experiment_for_ESORICS.ipynb` | Figures 1, 2 & Table 2 (ML-100K) | RMSE and NDCG@10 vs ε across four mechanisms on MovieLens 100K |
| `Sparsity_Experiment.ipynb` | Figure 3 | RMSE at natural, 50%, and 20% sparsity on MovieLens 100K |
| `1M_Experiment_for_ESORICS.ipynb` | Figure 4 & Table 2 (ML-1M) | Cross-dataset comparison on MovieLens 1M |
| `Jester_Experiment_for_ESORICS.ipynb` | Figure 4 & Table 2 (Jester) | Cross-dataset comparison on Jester |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations and noise sampling |
| `scipy` | Laplace and Gaussian distributions |
| `scikit-learn` | Truncated SVD, cross-validation, RMSE |
| `pandas` | Dataset loading and result aggregation |
| `matplotlib` | Figure generation |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 👥 Authors

| Role | Name | Affiliation |
|---|---|---|
| Lead author & corresponding author | Jeyamohan Neera | School of Computer Science, Northumbria University, UK |
| Co-author | Lucas Franca | School of Computer Science, Northumbria University, UK |

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@inproceedings{neera2026hiddencost,
  title     = {The Hidden Cost of Clipping: Privacy Mechanisms and Bounded Domain Bias in Recommender Systems},
  author    = {Neera, Jeyamohan and Franca, Lucas},
  booktitle = {Proceedings of ESORICS 2026},
  year      = {2026},
  publisher = {Springer}
}
```

---

## 📄 Licence

Released under the [GNU General Public License v3.0](./LICENSE).

Copyright © Jeyamohan Neera & Lucas Franca, 2026
