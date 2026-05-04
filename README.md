# HD2 — Fairness-Focused Heart Disease Prediction (CDC BRFSS 2024)

A machine learning project that evaluates whether common classification models
perform **equitably across age groups** when predicting heart disease risk using
the [CDC BRFSS 2024](https://www.cdc.gov/brfss/) dataset (~457,000 records,
345 variables).

## Motivation

Many ML studies report strong *average* accuracy while hiding serious failures
for specific demographic groups.  In a healthcare setting, a high false negative
rate for younger patients — who may appear lower-risk on aggregate — can delay
diagnosis and treatment.  This project treats **subgroup auditing** as a
necessary part of responsible model evaluation.

## What the project does

| Step | Description |
|------|-------------|
| **Preprocessing** | Feature selection, BRFSS missing-code removal, binary encoding, age-group stratification |
| **Modelling** | Logistic Regression, Random Forest, SVM, XGBoost — trained with and without class weighting |
| **Standard evaluation** | Accuracy, Recall, F1-score, AUC-ROC |
| **Fairness evaluation** | False Negative Rate (FNR), Recall, F1, AUC-ROC broken down by three broad age groups: *Young (18-39)*, *Middle-aged (40-64)*, *Older adults (65+)* |
| **Fairness gap** | Max − Min metric across groups; highlights which model/group combination is worst |
| **Mitigation** | Re-runs with `class_weight="balanced"` / XGBoost `scale_pos_weight` and compares whether weighting narrows or widens age disparities |

## Project structure

```
HD2/
├── main.py                  # CLI entry-point for the full pipeline
├── requirements.txt
├── src/
│   ├── preprocessing.py     # Data loading, encoding, train/test split
│   ├── models.py            # Model definitions and evaluation metrics
│   ├── fairness.py          # Subgroup metrics, gap analysis, visualisations
│   └── utils.py             # Synthetic data generator (for testing/demo)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_fairness.py
└── data/                    # Place BRFSS 2024 XPT file here (not committed)
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with synthetic data (no real dataset needed)

```bash
python main.py --synthetic --n-samples 50000 --output-dir results
```

### 3. Run with real BRFSS 2024 data

Download `LLCP2024.XPT` from the
[CDC BRFSS page](https://www.cdc.gov/brfss/annual_data/annual_2024.html) and
place it in the `data/` folder, then:

```bash
python main.py --data data/LLCP2024.XPT --output-dir results
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

## Output files (written to `results/`)

| File | Content |
|------|---------|
| `fnr_by_age_standard.png` | FNR by age group — standard models |
| `fnr_by_age_weighted.png` | FNR by age group — class-weighted models |
| `recall_by_age_standard.png` | Recall by age group — standard models |
| `overall_vs_young_fnr.png` | Overall vs. Young (18-39) FNR comparison |
| `fairness_gaps_standard.csv` | Numeric fairness gap table — standard models |
| `fairness_gaps_weighted.csv` | Numeric fairness gap table — weighted models |

## Key finding (synthetic data demo)

Even with balanced overall recall, **Young adults (18-39) consistently show
FNR values of 0.75–0.96** across all models, compared to 0.18–0.34 for Older
adults (65+).  Class weighting reduces overall FNR but does not eliminate the
age-group disparity, confirming that aggregate mitigation strategies alone are
insufficient without targeted subgroup auditing.

## CLI options

```
python main.py --help

  --data FILE        Path to BRFSS 2024 .XPT or .csv file
  --synthetic        Generate synthetic BRFSS-like data instead
  --n-samples N      Number of synthetic records  [default: 50000]
  --output-dir DIR   Where to write results       [default: results]
  --random-state N   Random seed                  [default: 42]
```
