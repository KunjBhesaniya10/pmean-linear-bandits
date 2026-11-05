# Contextualizing Fairness: Extending $p$-Mean Social Welfare to Linear Bandits

**Course Project:** CS6106: Multi-Agent Machine Learning, IIT Bombay  
**Instructor:** Prof. Avishek Ghosh  
**Authors:** Kunj Bhesaniya (Roll No. 23B0995) & Dhruvraj Merchant (Roll No. 23B1041)  
**GitHub Repository:** [https://github.com/KunjBhesaniya10/pmean-linear-bandits](https://github.com/KunjBhesaniya10/pmean-linear-bandits)

---

## 1. Project Overview

Standard linear bandit algorithms maximize aggregate reward, leading to winner-take-all dynamics that starve under-represented or minority arms. $p$-mean social welfare interpolates continuously between utilitarian efficiency ($p=1$), Nash proportional fairness ($p \to 0$), and egalitarian fairness ($p \to -\infty$).

This project provides:
1. **Empirical Reproduction of Welfarist-UCB** ([Sarkar, Pandey & Ray Chowdhury, 2025](https://arxiv.org/abs/2510.21312)): Full multi-agent multi-armed bandit reproduction across Experiments (a)–(f).
2. **Novel Contextual Extension**: Addressing dynamic streaming contexts where static geometric designs (such as John ellipsoids or precomputed D-optimal cores) cannot be precomputed:
   - **Adaptation 1 (Baseline Uniform Transfer):** Uniform random candidate exploration with best-arm adaptive certificate, followed by Phase II LinUCB.
   - **Adaptation 2 (50/50 D-Optimal Active Exploration):** Interleaves uniform selection with greedy D-optimal uncertainty sampling ($\arg\max_{a \in \mathcal{C}_t} \|x_a\|_{V_t^{-1}}$), actively learning rare directions before transitioning adaptively to LinUCB.
3. **Empirical Validation**:
   - **Experiment 1 (Benign Feature Arms):** Verifies fast Phase I termination and near-zero regret.
   - **Experiment 2 (Rare Niche Features):** Demonstrates that active D-optimal sampling prevents over-exploration on mediocre arms, achieving **$1.8\times$ lower regret** than uniform exploration.
   - **MovieLens-1M Streaming Benchmark:** Validates real-world dynamic user streaming, demonstrating superior welfare scaling across $p$ and protecting vulnerable demographic groups (e.g., **$0.863$ vs $0.845$** on Farmers).

---

## 2. Environment Setup

### Requirements
- Python 3.10+
- Dependencies: `numpy>=2.0`, `numba>=0.61`, `matplotlib>=3.8`, `pandas>=2.0`, `pytest>=8`

### Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install pmlb package in editable mode
pip install -e .

# 3. Verify installation with the test suite (44 tests, under 1 minute)
python -m pytest
```

---

## 3. Dataset & External Links

Per submission guidelines, large raw datasets are not bundled inside this archive. You can fetch MovieLens-1M directly via our automated helper script or the external links below:

- **Automated Download Script:**
  ```bash
  python scripts/download_movielens.py
  ```
- **Official GroupLens Source:** [https://files.grouplens.org/datasets/movielens/ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip) (ml-1m.zip, 5.9 MB)
- **GitHub CSV Mirror:** [https://github.com/khanhnamle1994/movielens](https://github.com/khanhnamle1994/movielens) (used automatically as fallback if GroupLens is unreachable)

The script formats and extracts the files into `data/ml-1m/`.

---

## 4. How to Run the Experiments Presented in the Project Presentation

All experiments match the final presentation slides (`slides/CS6106_TeamXX_Slides.pdf`).

### Run All Experiments
To run all presented experiments in sequence:
```bash
python scripts/run_all_experiments.py
```

### Run Individual Components

#### Stage 1: MAB Reproduction Experiments (a)–(f) (Presentation Slides 8–11)
Reproduces multi-armed bandit results comparing Welfarist-UCB, NCB, and Explore-then-UCB:
```bash
# Run all MAB experiments
python -m experiments.mab_reproduction

# Or run specific experiments, e.g., (a) and (f)
python -m experiments.mab_reproduction a f
```
- Outputs: Figures in `plots/mab_reproduction/`, summary tables in `results/mab_reproduction/summary.md`.

#### Stage 2: Linear Synthetic Experiments 1 & 2 (Presentation Slides 16–17)
Evaluates Contextual Adaptation 1 vs Adaptation 2:
```bash
# Run both synthetic experiments
python -m experiments.linear_synthetic

# Or run individual experiments:
python -m experiments.linear_synthetic 1   # Experiment 1: Benign Feature Arms
python -m experiments.linear_synthetic 2   # Experiment 2: Rare Niche Features
```
- Outputs: `plots/linear_synthetic/expt1_variant1_wins.png`, `plots/linear_synthetic/expt2_variant2_wins.png`, and markdown summaries in `results/linear_synthetic/`.

#### Stage 3: MovieLens-1M Real-World Streaming Benchmark (Presentation Slides 18–19)
Evaluates dynamic user streaming personalization, regret trajectories, and demographic subgroup equity:
```bash
python -m experiments.movielens
```
- Outputs:
  - `plots/movielens/movielens_clean.png` (Regret dynamics & scaling across $p \in \{1, 0, -2\}$)
  - `plots/movielens/movielens_demographics.png` (Subgroup utility across all 30 demographic categories)
  - Summary tables in `results/movielens/summary.md` and CSV curves in `results/movielens/`.

---

## 5. Summary of Key Experimental Results

### Multi-Armed Bandit Reproduction (50 Monte Carlo Trials)
- **Expt (a) Bernoulli ($T=10^7, p=0$):** Welfarist-UCB achieves Nash regret **$0.062$** vs $0.225$ for NCB ($4\times$ faster Phase I exit).
- **Expt (b) Gaussian ($T=10^6, p=0$):** Welfarist-UCB achieves Nash regret **$0.18$** vs $1.75$ for NCB ($\tau=266$ vs $2,487$).
- **Expt (c)–(e) Explore-then-UCB vs Welfarist-UCB ($T=10^6$):**
  - $p=0.5$: Welfarist-UCB **$0.16$** vs Explore-then-UCB $45.3$
  - $p=-0.5$: Welfarist-UCB **$0.22$** vs Explore-then-UCB $18.1$
  - $p=-1.5$: Welfarist-UCB **$0.72$** vs Explore-then-UCB $3.85$
- **Expt (f) Fairness Scaling Across $p$ ($T=10^6$):** Phase I exploration scales with $p^2$ for $p < -1$ ($\tau=275$ at $p=-1$, $\tau=1,010$ at $p=-2$, $\tau=5,922$ at $p=-5$).

### Contextual Linear Synthetic Experiments
- **Experiment 1: Benign Feature Arms ($d=6, K=50, T=50,000$):** Both adaptations exit exploration early ($\tau \approx 2,850$) and achieve near-zero cumulative regret ($0.0053$ vs $0.0068$).
- **Experiment 2: Rare Niche Features ($d=6, K=100, T=30,000$):**
  - Adaptation 1 (Uniform): Common arms fail to clear the stopping threshold; gets trapped in exploration ($\tau \approx 16,000$, regret $0.428$).
  - Adaptation 2 (50/50 D-Optimal): Targets rare feature coordinates actively, exits early ($\tau \approx 11,000$), achieving **$1.8\times$ lower regret ($0.245$)**.

### MovieLens-1M Dynamic Streaming Benchmark ($d=180, K=20, T=50,000$)
- **Prompt Exit:** Both adaptations stop Phase I early ($\tau \approx 1,178$ vs $1,295$), smoothly delegating exploitation to LinUCB.
- **Superior Regret Scaling:** Adaptation 2 achieves Nash regret $0.0491$ (vs $0.0493$) and maintains low regret across fairness parameters ($0.045$ at $p=1$, $0.075$ at $p=-2$).
- **Demographic Equity:** Adaptation 2 delivers higher normalized utility to the rarest demographic group (**Farmers: $0.863$ vs $0.845$**) and the highest generalized group welfare ($M_{-2} = \mathbf{0.9361}$).

---

## 6. Codebase Structure

```
.
├── pyproject.toml              # Build metadata and test configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── scripts/
│   ├── download_movielens.py   # Dataset downloader with automatic fallback mirror
│   └── run_all_experiments.py  # Master runner for all presentation benchmarks
├── src/pmlb/
│   ├── envs.py                 # Multi-armed bandit environments and reward samplers
│   ├── plotting.py             # Matplotlib styling and curve generation utilities
│   ├── runner.py               # Parallel Monte Carlo trial executor
│   ├── welfare.py              # Generalized p-mean, running welfare, and regret metrics
│   ├── data/
│   │   └── movielens.py        # MovieLens-1M loader, user/item feature embeddings
│   ├── mab/
│   │   └── algorithms.py       # Numba JIT kernels for Welfarist-UCB, NCB, Explore-then-UCB
│   └── linear/
│       ├── algorithms.py       # Contextual Adaptation 1, Adaptation 2, and LinUCB kernels
│       ├── envs.py             # Linear contextual environments and feature generators
│       ├── fair_lin_bandit.py  # Sarkar et al. (2026) FairLinBandit baseline
│       └── lin_nash.py         # Sawarni et al. (2023) LinNash baseline
├── experiments/
│   ├── common.py               # Shared caching and reporting routines
│   ├── mab_reproduction.py     # Multi-armed bandit experiments (a)-(f)
│   ├── linear_synthetic.py     # Linear synthetic Experiments 1 & 2
│   └── movielens.py            # MovieLens-1M dynamic contextual benchmark
├── tests/
│   ├── test_linear.py          # Sherman-Morrison, widths, and linear algorithm tests
│   ├── test_mab.py             # MAB kernel and stopping rule tests
│   └── test_welfare.py         # p-mean properties and regret curve tests
├── results/                    # Summary markdown tables and curve CSVs
└── plots/                      # Generated publication-quality figures
```

---

## 7. References

1. A. Sarkar, A. Pandey, and S. Ray Chowdhury. *Revisiting Social Welfare in Bandits: UCB is (Nearly) All You Need*. arXiv:2510.21312, 2025.
2. D. Sarkar, A. Pandey, and S. Ray Chowdhury. *Improved Algorithms for Nash Welfare in Linear Bandits*. ICML, arXiv:2601.22969, 2026.
3. A. Sawarni, S. Pal, and S. Barman. *Nash Regret Guarantees for Linear Bandits*. NeurIPS, 2023.
4. S. Barman, A. Khan, A. Maiti, and A. Sawarni. *Fairness and Welfare Quantification for Regret in Multi-Armed Bandits*. AAAI, 2023.
5. A. Krishna, P. G. John, A. Barik, and V. Y. F. Tan. *$p$-Mean Regret for Stochastic Bandits*. AAAI, 2025.
6. Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. *Improved Algorithms for Linear Stochastic Bandits*. NeurIPS, 2011.
