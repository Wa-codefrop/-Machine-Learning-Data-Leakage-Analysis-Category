# FlyRank AI — Real SEO Performance Analysis and Data Leakage Review

This project is a realistic machine learning and data-leakage portfolio exercise centered on a real SEO performance dataset: UrbanScape Apparel SEO Performance. The goal is not to build a production-grade SEO engine. The goal is to demonstrate disciplined ML reasoning: understand the data first, choose valid targets, engineer features carefully, compare models honestly, and explain where leakage can distort performance.

## Project purpose

The core learning objective is to show that a model is only as credible as its target definition and feature set. In this repository, the dataset is treated as the source of truth. We do not force a synthetic decline-label task onto real data. Instead, we work with what the data can legitimately support.

This project explores:
- SEO traffic and ranking behavior
- feature groups such as technical SEO, content, conversion, and acquisition signals
- time-aware model evaluation
- leakage-sensitive feature design
- honest interpretation of what can and cannot be predicted from the dataset

## At a glance

This project is intentionally designed to show both the valid and invalid ways of modeling SEO performance. The real-data comparison below comes from the leakage audit flow in the notebooks and demonstrates why feature validity matters.

| Feature set | Model | F1 | ROC-AUC |
| --- | --- | ---: | ---: |
| Safe | Logistic Regression | 0.9724 | 0.9997 |
| Safe | Random Forest | 1.0000 | 1.0000 |
| Leaky | Logistic Regression | 0.3566 | 0.5240 |
| Leaky | Random Forest | 0.0000 | 0.5097 |

The leaky configuration is included to illustrate how outcome-adjacent or future-window features can distort performance. The safe configuration is the honest estimate of the model's real predictive value on the UrbanScape dataset.

## Dataset

The active project uses the real UrbanScape Apparel SEO Performance dataset stored under the raw data directory. This is a publicly available SEO dataset used as a realistic benchmark for portfolio work.

Important note:
- The dataset is not a repeated-entity panel with one row per content item over many time points.
- It is real-world SEO data and does not naturally define a clean decline-detection target in the way a synthetic leakage project often does.
- Because of that, the project makes an honest decision: define valid prediction tasks from the real data, rather than inventing a synthetic target structure.

## Why this project matters

A model can look excellent if it uses leakage. In SEO settings, this often happens when features are derived from the same business window as the outcome, or when the data structure implicitly encodes the label. This project demonstrates that idea clearly by comparing:

- a leakage-safe feature set
- a leakage-prone feature set that exaggerates performance

The point is not to reward the leaky model. The point is to show why feature validity and temporal logic are essential.

## Valid ML tasks supported by this dataset

The real data supports realistic, honest tasks such as:
- high-performance classification using traffic thresholds
- traffic regression and performance ranking
- segmentation and exploratory SEO analysis
- technical and content-factor comparison

The dataset does not cleanly support a repeated-entity decline prediction task without constructing a pseudo-entity structure that would not be grounded in the underlying data.

## Repository structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── archive/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_real_leakage_comparison.ipynb
│   └── 04_findings_summary.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── utils/
│   └── pipeline.py
├── tests/
├── docs/
├── outputs/
├── experiments/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .github/
├── LICENSE
└── .gitignore
```

## Workflow

The project follows a realistic ML workflow:

1. audit the dataset and check schema, missingness, duplicates, and temporal structure
2. define valid prediction targets from the data
3. perform exploratory analysis on traffic, CTR, rank, technical performance, and acquisition signals
4. train time-aware baseline models
5. compare leakage-safe and leakage-prone features
6. document the findings honestly and clearly

## Project notebooks

- [notebooks/01_dataset_audit.ipynb](notebooks/01_dataset_audit.ipynb): dataset audit and data-quality checks
- [notebooks/02_exploratory_data_analysis.ipynb](notebooks/02_exploratory_data_analysis.ipynb): SEO exploration across traffic, rank, and performance metrics
- [notebooks/03_real_leakage_comparison.ipynb](notebooks/03_real_leakage_comparison.ipynb): safe-vs-leaky baseline comparison on the real dataset
- [notebooks/04_findings_summary.ipynb](notebooks/04_findings_summary.ipynb): final interpretation and business conclusion

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the checks

```bash
python -m pytest -q
```

## Key findings

The project’s central finding is straightforward:

- a real SEO dataset can support a valid business-oriented ML analysis
- the dataset should determine the target and the task
- leakage can inflate metrics dramatically when the feature set contains outcome-adjacent or future-window information
- a careful time-aware split and feature audit are essential for credible results

The project therefore demonstrates a portfolio-worthy lesson: strong metrics are not enough; valid task definition and leakage-safe feature engineering are what make the model trustworthy.

## Limitations

This is intentionally an educational and portfolio-focused project, not a production SEO platform. It avoids overclaiming and focuses on transparent analysis, realistic feature choices, and honest interpretation of model uncertainty.

## Current status

The repository is in a real-data phase, with a valid baseline flow and a leakage-aware comparison framework built around actual SEO metrics. The project remains intentionally honest about the limits of the dataset while still delivering a solid demonstration of ML workflow discipline.
