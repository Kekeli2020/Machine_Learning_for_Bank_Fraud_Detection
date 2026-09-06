"""
Bank Fraud Prediction — EDA & Modeling
Author: Kekeli Tsoekewo
Date: 09/02/2026
Dataset: NeurIPS 2022 Bank Account Fraud (Base.csv)

Thesis:
Fraudulent account applications are rare but costly, and fraud rings tend to
leave behavioral fingerprints: bursts of similar applications in short time
windows (velocity), mismatches between requested credit and risk profile, and
a preference for channels/devices that are easier to automate or spoof. This
script explores those hypotheses with EDA, then tests whether tree-based
models can turn the signals into a working fraud classifier.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)


# Use the sample by default for development. Full dataset remains local and is not committed.
SAMPLE_PATH = Path("data") / "sample_base.csv"
FULL_PATH = Path(r"C:\Users\KEKELI\OneDrive\Desktop\PythonProg\Base.csv")

# Run mode: trade off speed vs. thoroughness.
USE_FULL_DATA = False
SAMPLE_FRAC = 0.2
MIN_ROWS_FOR_SAMPLING = 2000
N_SPLITS, N_REPEATS = 5, 1


# Load data (single, clear block; sample preferred)

def load_dataset():
    if SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)
        source = SAMPLE_PATH
    elif FULL_PATH.exists():
        df = pd.read_csv(FULL_PATH)
        source = FULL_PATH
    else:
        raise FileNotFoundError(f"Neither {SAMPLE_PATH} nor {FULL_PATH} was found.")
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns from {source}")
    return df

df = load_dataset()

# A handful of numeric columns use -1 as a "not available" sentinel.
# Swap it for NaN so it doesn't get treated as a real value in plots or stats.
SENTINEL_COLS = [
    'prev_address_months_count', 'current_address_months_count',
    'intended_balcon_amount', 'bank_months_count',
    'session_length_in_minutes', 'device_distinct_emails_8w'
]
present_sentinels = [c for c in SENTINEL_COLS if c in df.columns]
if present_sentinels:
    df.loc[:, present_sentinels] = df.loc[:, present_sentinels].replace(-1, np.nan)


# EDA — testing the hypotheses behind the model


# 1. Class balance
# Fraud is rare. This shapes everything downstream: accuracy is a bad metric,
# training needs SMOTE (or similar), and the decision threshold is critical
# than the raw prediction.
ax = df['fraud_bool'].value_counts(normalize=True).plot(kind='bar', color=['C0', 'C3'])
ax.set_xticklabels(['Legit (0)', 'Fraud (1)'])
ax.set_ylabel('Proportion')
ax.set_title('Class Balance: Fraud is Rare')
plt.tight_layout()
plt.show()

# 2. Univariate distributions
# Sanity-check the raw shape of key numeric features before comparing fraud vs legit transactions.
univariate_cols = [
    'income', 'customer_age', 'intended_balcon_amount', 'proposed_credit_limit',
    'velocity_6h', 'velocity_24h', 'velocity_4w', 'session_length_in_minutes'
]
for col in univariate_cols:
    if col not in df.columns:
        continue
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col].dropna(), bins=80, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# 3. Behavioral signals: do fraudulent transactions look different?
# Intuition: fraud rings submit applications in bursts, so velocity features
# and the credit risk score should separate the two classes.
behavior_cols = ['velocity_6h', 'velocity_24h', 'velocity_4w', 'zip_count_4w',
                  'session_length_in_minutes', 'credit_risk_score']
for col in behavior_cols:
    if col not in df.columns:
        continue
    plt.figure(figsize=(10, 4))
    sns.kdeplot(data=df, x=col, hue='fraud_bool', common_norm=False)
    plt.title(f'{col} by Fraud Label')
    plt.tight_layout()
    plt.show()

# 4. Channel & device signals
# Intuition: certain acquisition channels/devices are easier to automate or
# spoof, so fraud should cluster in specific categories rather than spread evenly.
channel_cols = ['source', 'device_os', 'payment_type', 'employment_status', 'housing_status']
for col in channel_cols:
    if col not in df.columns:
        continue
    order = df[col].value_counts().index
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]})

    sns.countplot(data=df, x=col, order=order, ax=axes[0])
    axes[0].set_title('Volume')
    axes[0].tick_params(axis='x', rotation=45)

    fraud_rate = df.groupby(col)['fraud_bool'].mean().reindex(order)
    sns.barplot(x=fraud_rate.index, y=fraud_rate.values, ax=axes[1])
    axes[1].set_ylabel('Fraud rate')
    axes[1].set_title('Fraud Rate')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle(col)
    plt.tight_layout()
    plt.show()

# 5. Correlation among risk features
# Checking for redundant/multicollinear features before modeling — if two
# features move together, the model doesn't gain much from having both.
corr_cols = ['income', 'customer_age', 'credit_risk_score', 'intended_balcon_amount',
             'proposed_credit_limit', 'velocity_6h', 'velocity_24h', 'velocity_4w',
             'zip_count_4w', 'bank_branch_count_8w', 'session_length_in_minutes']
corr_cols = [c for c in corr_cols if c in df.columns]
if len(corr_cols) >= 2:
    corr = df[corr_cols].corr()
    sns.clustermap(corr, annot=True, fmt=".2f", cmap='vlag', center=0)
    plt.show()

# 6. Fraud rate over time
# Fraud patterns drift as fraudsters adapt — useful for knowing whether the
# model will need periodic retraining or monitoring.
if 'month' in df.columns:
    monthly = df.groupby('month')['fraud_bool'].agg(['mean', 'count']).reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=monthly, x='month', y='mean', marker='o')
    plt.ylabel('Fraud rate')
    plt.title('Fraud Rate by Month')
    plt.tight_layout()
    plt.show()

# 7. Engineered ratios
# Raw values can hide anomalies that ratios reveal: asking for a balance far
# above the proposed credit limit, or a spike in short-term velocity relative
# to the longer-term baseline.
if {'intended_balcon_amount', 'proposed_credit_limit', 'velocity_6h', 'velocity_4w'}.issubset(df.columns):
    df['amount_to_limit'] = df['intended_balcon_amount'] / (df['proposed_credit_limit'] + 1e-9)
    df['vel_ratio_6h_4w'] = (df['velocity_6h'] + 1) / (df['velocity_4w'] + 1)

    n_plot = min(20000, len(df))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df.sample(n_plot, random_state=42), x='amount_to_limit',
                     y='vel_ratio_6h_4w', hue='fraud_bool', alpha=0.6, s=10)
    plt.title('Amount-to-Limit vs Velocity Ratio')
    plt.tight_layout()
    plt.show()

if 'velocity_24h' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x='fraud_bool', y='velocity_24h', density_norm='width', inner='quartile')
    plt.yscale('symlog')  # heavy right tail — log scale keeps the shape readable
    plt.title('Velocity 24h by Fraud Label')
    plt.xlabel('Fraud')
    plt.tight_layout()
    plt.show()


# Modeling — using these signals to predict fraud.


# Run mode: trade off speed vs. thoroughness.
# The three fixes below are what actually cut runtime (the old ~1hr version):
#   1. Class imbalance is handled per-model instead of one-size-fits-all SMOTE.
#      RF/XGBoost get imbalance handling built in (class_weight / scale_pos_weight),
#      so they train on the fold as-is — no oversampling blow-up in row count.
#      KNN has no such option, so it alone trains on an undersampled (smaller,
#      not bigger) balanced set — this also fixes KNN's real bottleneck, which
#      is prediction cost scaling with training-set size.
#   2. Preprocessing (impute/scale/encode) is fit ONCE per fold and reused by
#      all 3 models, instead of being redone from scratch inside each model's
#      own pipeline (that was 3x redundant work every fold).
#   3. CV repeats default to 1 — one pass across 5 folds is enough to compare
#      models; bump N_REPEATS up later only if you want tighter error bars.
USE_FULL_DATA = USE_FULL_DATA  # keep original variable name
SAMPLE_FRAC = SAMPLE_FRAC
N_SPLITS, N_REPEATS = N_SPLITS, N_REPEATS

if USE_FULL_DATA:
    print("Running on the full dataset — this is the slow option.")
else:
    # If the sample is small, skip additional downsampling to avoid tiny folds.
    if len(df) >= MIN_ROWS_FOR_SAMPLING:
        df = df.groupby('fraud_bool', group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_FRAC, random_state=42)
        )
    else:
        print("Sample is small; skipping additional downsampling.")

print(f"Modeling on {df.shape[0]:,} rows | {N_SPLITS}-fold CV x {N_REPEATS} repeat(s)")

y = df['fraud_bool'].astype(int)
X = df.drop(columns=['fraud_bool']).copy()

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing — fit fresh inside each fold (see loop below) so nothing from
# the validation split leaks into training.
# Keep OneHotEncoder compatibility across sklearn versions without changing style.
import sklearn
skl_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
ohe_kwargs = {'handle_unknown': 'ignore'}
if skl_ver >= (1, 2):
    ohe_kwargs['sparse_output'] = False
else:
    ohe_kwargs['sparse'] = False

preprocessor = ColumnTransformer([
    ('num', SkPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', SkPipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(**ohe_kwargs))
    ]), cat_cols)
])

# Candidate models
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric='logloss',
                             tree_method='hist', n_jobs=-1, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
under_sampler = RandomUnderSampler(random_state=42)

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
model_names = ['RandomForest', 'XGBoost', 'KNN']
results = {name: {'y_true': [], 'y_proba': []} for name in model_names}

fold = 0
for train_idx, val_idx in cv.split(X, y):
    fold += 1
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Transform once, reuse for every model this fold.
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # RF and XGBoost: fit directly, weighting handles the imbalance.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf.set_params(scale_pos_weight=scale_pos_weight)

    rf.fit(X_train_proc, y_train)
    xgb_clf.fit(X_train_proc, y_train)

    results['RandomForest']['y_true'].append(y_val.values)
    results['RandomForest']['y_proba'].append(rf.predict_proba(X_val_proc)[:, 1])
    results['XGBoost']['y_true'].append(y_val.values)
    results['XGBoost']['y_proba'].append(xgb_clf.predict_proba(X_val_proc)[:, 1])

    # KNN: fit on a small undersampled set — faster to train AND to predict with.
    X_train_bal, y_train_bal = under_sampler.fit_resample(X_train_proc, y_train)
    knn.fit(X_train_bal, y_train_bal)
    results['KNN']['y_true'].append(y_val.values)
    results['KNN']['y_proba'].append(knn.predict_proba(X_val_proc)[:, 1])

    print(f"Completed fold {fold}/{N_SPLITS * N_REPEATS}")

# Evaluate & pick a threshold
# Fraud review queues are precision-first: false positives waste analyst time
# and annoy legitimate customers. Aim for precision >= 0.9 if reachable,
# otherwise fall back to the threshold that maximizes F1.
TARGET_PRECISION = 0.9
summary = {}  # collects everything needed for the comparison plots below

for name in model_names:
    res = results[name]
    y_true_all = np.concatenate(res['y_true'])
    y_proba_all = np.concatenate(res['y_proba'])

    precision, recall, thresholds = precision_recall_curve(y_true_all, y_proba_all)
    pr_auc = auc(recall, precision)
    roc = roc_auc_score(y_true_all, y_proba_all)
    print(f"\n{name} — PR AUC: {pr_auc:.4f} | ROC AUC: {roc:.4f}")

    idxs = np.where(precision >= TARGET_PRECISION)[0]
    if len(idxs) > 0:
        chosen_idx = idxs[-1]
        chosen_threshold = thresholds[chosen_idx] if chosen_idx < len(thresholds) else 0.5
    else:
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        chosen_idx = np.nanargmax(f1_scores)
        chosen_threshold = thresholds[chosen_idx] if chosen_idx < len(thresholds) else 0.5
    print(f"Chosen threshold: {chosen_threshold:.4f}")

    y_pred = (y_proba_all >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_true_all, y_pred)
    print("Confusion matrix (tn, fp, fn, tp):", cm.ravel())
    print(classification_report(y_true_all, y_pred, digits=4))

    summary[name] = {
        'precision': precision, 'recall': recall, 'pr_auc': pr_auc,
        'cm': cm, 'threshold': chosen_threshold
    }

# One PR-curve plot with all 3 models overlaid — easiest way to see who wins.
plt.figure(figsize=(8, 6))
for name, s in summary.items():
    plt.plot(s['recall'], s['precision'], label=f"{name} (PR AUC={s['pr_auc']:.3f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve: Model Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrices for all 3 models side by side, at each model's own threshold.
fig, axes = plt.subplots(1, len(summary), figsize=(6 * len(summary), 5))
for ax, (name, s) in zip(axes, summary.items()):
    sns.heatmap(s['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_title(f"{name} @ {s['threshold']:.3f}")
plt.suptitle('Confusion Matrices by Model')
plt.tight_layout()
plt.show()
