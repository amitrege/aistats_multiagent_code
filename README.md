# Multi-Agent Lipschitz Bandits: Experimental Code

This repository contains the code used to generate the empirical figures for the experimental appendix of the paper `Multi-Agent Lipschitz Bandits`.

It contains:

- the synthetic experiment runner,
- lightweight setup instructions.

## Repository layout

```text
experiments/run_experiments.py
requirements.txt
```

## Setup

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper-mode figures

From the repository root:

```bash
python3 experiments/run_experiments.py --mode paper
```

This writes outputs to:

- `img/empirical_regret_main.png`
- `img/empirical_collisions.png`
- `img/empirical_pathology.png`
- `experiments/results/summary_paper.json`

## Experiments included

The script currently generates three appendix experiments:

1. Regret curves in 1D and 2D comparing the proposed coordination-first protocol against an independent single-agent baseline.
2. Collision traces showing that the proposed method concentrates collisions in the short coordination stage.
3. A boundary-peak pathology illustration showing why ranking cells by center values can fail, and why the local-peek stage is needed.

## Notes

- The baseline is simple: each player runs an independent global fixed-grid UCB routine and ignores the presence of the other players except through collision-censored feedback.
- For the empirical illustration, the protocol pools successful Phase I and Phase II identification samples when forming the common target set. This keeps the experiments focused on the coordination-versus-learning decomposition highlighted by the paper.
