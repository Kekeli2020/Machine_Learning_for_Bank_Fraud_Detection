# Bank Fraud Prediction — ML Classification

A machine learning pipeline to predict fraudulent account applications using behavioral and channel signals from the NeurIPS 2022 Bank Account Fraud dataset.

## Thesis

Fraudulent account applications are rare but costly. Fraud rings leave behavioral fingerprints:
- **Velocity bursts**: multiple similar applications in short time windows
- **Mismatched risk**: requesting high credit limits with poor risk profiles
- **Channel clustering**: preference for easier-to-automate channels/devices

This repo explores these signals through EDA, then compares tree-based models (Random Forest, XGBoost, KNN) to find a working fraud classifier.

## Setup

### Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Data
The full `Base.csv` (203 MB) is too large for GitHub. Two options:

**Option 1: Use the sample (recommended for dev)**
1. Create a `data/` folder in the repo root:
   ```bash
   mkdir data
   ```
2. Obtain or create a stratified 20% sample of `Base.csv` and save as `data/sample_base.csv`
3. The script will auto-load the sample for fast iteration

**Option 2: Use your local full dataset**
1. Edit the `FULL_PATH` variable in `ML_for_Bank_Fraud_Detection.py`:
   ```python
   FULL_PATH = Path(r"C:\path\to\your\Base.csv")
   ```
2. Set `USE_FULL_DATA = True` to use it

## Usage

Run the full pipeline (EDA + modeling):
```bash
python ML_for_Bank_Fraud_Detection.py
```

### Control runtime
Edit these variables at the top of the script:
- `USE_FULL_DATA`: `False` (sample, ~5 min) or `True` (full data, ~30–60 min)
- `SAMPLE_FRAC`: sample size if `USE_FULL_DATA = False` (default: 0.2)
- `N_REPEATS`: cross-validation repeats (default: 1 for speed; bump to 3 for stability)

## What it does

### 1. EDA (5 plots)
- Class balance (fraud is ~1.3% of data)
- Univariate distributions of key features
- Behavioral signals (velocity, risk scores by fraud label)
- Channel/device signals (fraud rate per source, device OS, etc.)
- Correlation matrix (multicollinearity check)
- Fraud rate over time (drift detection)
- Engineered ratios (credit request vs limit, short-term vs long-term velocity)

### 2. Modeling
Three candidate classifiers, compared via cross-validation:
- **Random Forest**: ensemble of decision trees, handles nonlinear interactions, native class weighting
- **XGBoost**: gradient boosting, usually higher AUC but slightly slower
- **KNN**: simple distance-based baseline (slowest; undersampled internally for speed)

All three are fit on the same folds to ensure fair comparison.

### 3. Evaluation
- **PR AUC & ROC AUC**: overall model quality
- **Threshold selection**: aims for precision ≥ 0.9 (minimize false positives in fraud queue)
- **Confusion matrices**: side-by-side comparison of all 3 models
- **Classification report**: precision, recall, F1 per class

## Output

- **PR-curve plot**: all 3 models overlaid for easy comparison
- **Confusion matrices**: side-by-side heatmaps at each model's chosen threshold
- **Console metrics**: AUC, threshold, detailed classification report per model

## Notes

### Why undersampling for KNN only?
KNN's prediction cost scales with training set size. Oversampling (SMOTE) would have blown up the training set and made KNN very slow. Instead:
- RF/XGBoost: native class weighting, no resampling
- KNN: undersampled training set (smaller, balanced, faster to predict with)

### Why preprocessing is shared?
Imputation, scaling, and one-hot encoding are identical regardless of model. Fitting once per fold and reusing saves 3x redundant computation.

### Why no feature importance?
On a GitHub-hosted project, feature importance plots can bloat the repo. Add them locally if needed:
```python
# Example: RF feature importance
import matplotlib.pyplot as plt
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-10:]
plt.barh(range(10), importances[top_idx])
plt.show()
```

## GitHub: Large File Handling

If you ever need the full `Base.csv` in the repo:
1. Install [Git LFS](https://git-lfs.github.com)
2. Track it:
   ```bash
   git lfs track "*.csv"
   ```
3. Commit `.gitattributes` and retry the push

For now, using `.gitignore` keeps the repo lean and focused on code.

## Author
Kekeli Tsoekewo  
Date: 09/02/2026
