# Contributing

Thanks for your interest in FlyRank AI — Machine Learning & Data Leakage Analysis.

This repository is a synthetic, educational portfolio project. The goal is to demonstrate data exploration, cleaning, feature engineering, leakage analysis, model evaluation, and reproducible artifact generation.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common commands

```bash
python -m pytest -q
python -m src.pipeline --generate
```

## Workflow

1. Keep the synthetic dataset deterministic and documented.
2. Keep the tests green before opening a pull request.
3. Regenerate artifacts only when the output workflow intentionally changes.
4. Explain any leakage-related experiment clearly in notebooks or docs.
