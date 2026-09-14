# FlyRank AI — Machine Learning & Data Leakage Analysis

A synthetic, internship-style machine learning project that demonstrates a realistic SEO and content performance classification workflow with an emphasis on time-aware feature separation and data leakage detection.

## Project Overview

This project demonstrates a machine learning workflow for identifying SEO/content records that may be experiencing declining performance. It focuses not only on model performance but also on detecting data leakage and making sure that the model uses only information available at prediction time.

## Problem

Many content and SEO teams need early signals to identify pages that are starting to lose performance. However, a model that uses future information can create misleadingly high performance and produce a false sense of success. This project demonstrates that problem and shows how to test for it.

## Solution

The project generates a synthetic SEO/content performance dataset, builds a feature window, performs baseline modeling with a rule-based approach, Logistic Regression, and Random Forest, then compares a leakage-prone experiment with a leakage-free experiment.

## Key Learning

The main lesson is that high model performance does not automatically mean a valid model. If a feature represents a future outcome or label window, it must not be allowed into the prediction-time feature set.

## Architecture

```text
Synthetic SEO Dataset
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Time-Aware Split
        ↓
Baseline Models
        ↓
Leakage Investigation
        ↓
Leakage-Free Model
        ↓
Evaluation
        ↓
Results
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter
- Matplotlib
- Seaborn
- Google Colab
- Git
- GitHub

## Project Structure

```text
flyrank-ai-ml-data-leakage/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_data_leakage_investigation.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── utils/
│   └── pipeline.py
├── experiments/
│   └── results.csv
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── docs/
│   └── data_contract.md
├── tests/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.pipeline --generate
```

Then open the notebooks in the notebooks directory using Jupyter.

## Results

This repository is designed to generate and compare model experiments. The experiment scores are written to `experiments/results.csv`.

## Data Leakage Findings

The project is designed to demonstrate a direct leakage scenario where future-derived values such as trend_direction, trend_pct, future_clicks, and future_position are associated with the target label. These fields must be removed from the feature matrix. After removal, model behavior becomes a more honest estimate of predictive value.

## Limitations

This project uses a synthetic dataset. It is not a production FlyRank dataset. The dataset size is limited and the business rules are simplified for educational purposes.

## Future Improvements

- Use a real SEO dataset with stronger privacy review
- Add model monitoring and drift reporting
- Add automated retraining
- Add an API deployment layer
- Add a feature store and model registry
- Add experiment tracking tools such as MLflow

## Key Findings

The first experiment with future-derived leakage fields can achieve unusually strong performance because those features directly encode the label definition. After removing the future feature window and enforcing prediction-time feature availability, the model becomes a more realistic and honest classifier.
