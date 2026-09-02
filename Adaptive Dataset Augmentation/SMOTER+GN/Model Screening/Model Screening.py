"""
Supplementary Code - Model Screening with SMOTER + Gaussian Noise Augmentation

This script evaluates multiple models with SMOTER (extreme 10% oversampling)
followed by Gaussian noise augmentation for C14, C14 Alloy Screening, and TiFe phase datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

# ====================== Configuration ======================
SIGMA = 0.05
N_GAUSSIAN_COPIES = 5
SMOTER_REL_COEF = 0.1
SMOTER_OVER_SAMP_RATIO = 1.0
RANDOM_STATE = 42
CV_FOLDS = 5

# Dataset file paths (English column names)
data_files = {
    'C14': 'C14_data.csv',
    'C14 Alloy Screening': 'BCC_data.csv',
    'TiFe': 'TiFe_data.csv'
}

# Target column names (English)
target_cols = {
    'C14': 'Max_H2_Uptake_wt_pct',
    'C14 Alloy Screening': 'Max_H2_Uptake_wt_pct',
    'TiFe': 'Max_H2_Uptake_wt_pct'
}


# ====================== Helper Functions ======================
def add_gaussian_noise(df, sigma, numeric_cols, n_copies=1):
    """Generate augmented dataset with Gaussian noise (preserve original samples)."""
    noisy_dfs = [df]
    for _ in range(n_copies):
        df_noisy = df.copy()
        for col in numeric_cols:
            noise = np.random.normal(0, sigma, len(df))
            df_noisy[col] = df_noisy[col] + noise
        noisy_dfs.append(df_noisy)
    return pd.concat(noisy_dfs, ignore_index=True)


def get_numeric_columns(df, target_col):
    """Return numeric columns (exclude target and boolean/object columns)."""
    exclude = [target_col] if target_col is not None else []
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Skip binary columns (0/1) as they are categorical
            if df[col].nunique() <= 2 and set(df[col].unique()).issubset({0, 1}):
                continue
            numeric_cols.append(col)
    return numeric_cols


def smoter_extreme_oversample(X, y, rel_coef=0.1, over_samp_ratio=1.0, random_state=42):
    """
    Manual SMOTER oversampling for extreme target values (both low and high tails).

    Parameters:
        X: DataFrame, features
        y: array, target values
        rel_coef: proportion of extreme samples at each tail (e.g., 0.1 = bottom 10% and top 10%)
        over_samp_ratio: new samples = original extreme samples * over_samp_ratio
    Returns: X_resampled, y_resampled
    """
    np.random.seed(random_state)
    data = X.copy()
    data['target'] = y

    lower_threshold = np.percentile(y, rel_coef * 100)
    upper_threshold = np.percentile(y, (1 - rel_coef) * 100)

    low_mask = y <= lower_threshold
    high_mask = y >= upper_threshold
    extreme_mask = low_mask | high_mask
    extreme_indices = np.where(extreme_mask)[0]

    if len(extreme_indices) == 0:
        print("Warning: No extreme samples found, skipping SMOTER oversampling")
        return X, y

    n_extreme = len(extreme_indices)
    n_new = int(n_extreme * over_samp_ratio)
    if n_new == 0:
        return X, y

    X_extreme = X.iloc[extreme_indices]
    y_extreme = y[extreme_indices]
    X_other = X.drop(index=extreme_indices)
    y_other = np.delete(y, extreme_indices)

    new_samples = []
    new_targets = []
    for _ in range(n_new):
        idx1 = np.random.randint(0, n_extreme)
        idx2 = idx1
        while idx2 == idx1 and n_extreme > 1:
            idx2 = np.random.randint(0, n_extreme)
        lam = np.random.uniform(0, 1)
        x1 = X_extreme.iloc[idx1].values
        x2 = X_extreme.iloc[idx2].values
        x_new = x1 + lam * (x2 - x1)
        y_new = y_extreme[idx1] + lam * (y_extreme[idx2] - y_extreme[idx1])
        new_samples.append(x_new)
        new_targets.append(y_new)

    X_resampled = pd.concat([X_other, X_extreme, pd.DataFrame(new_samples, columns=X.columns)], ignore_index=True)
    y_resampled = np.concatenate([y_other, y_extreme, new_targets])

    print(f"SMOTER oversampling: Original extreme: {n_extreme}, New samples: {n_new}, Total: {len(X_resampled)}")
    return X_resampled, y_resampled


def evaluate_models(X, y, models, cv_folds=5):
    """Evaluate each model using cross-validation, return mean R² and std."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
        results[name] = (scores.mean(), scores.std())
    return results


# ====================== Main Pipeline ======================
all_results = {}

for name, path in data_files.items():
    print(f"\n{'=' * 50}")
    print(f"Processing Dataset: {name}")
    print(f"{'=' * 50}")

    # 1. Load data
    df = pd.read_csv(path, encoding='utf-8-sig')

    # 2. Get target column
    target_col = target_cols[name]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

    y = df[target_col].values

    # 3. Separate features
    X_raw = df.drop(columns=[target_col])

    # 4. Keep only numeric columns
    X_raw = X_raw.select_dtypes(include=[np.number])

    original_n = len(X_raw)
    print(f"Original samples: {original_n}")

    # 5. SMOTER extreme 10% oversampling
    X_smoter, y_smoter = smoter_extreme_oversample(
        X_raw, y, rel_coef=SMOTER_REL_COEF, over_samp_ratio=SMOTER_OVER_SAMP_RATIO, random_state=RANDOM_STATE
    )
    print(f"After SMOTER: {len(X_smoter)} samples")

    # 6. Gaussian augmentation (5x)
    X_smoter_df = pd.DataFrame(X_smoter, columns=X_raw.columns)
    numeric_cols = get_numeric_columns(X_smoter_df, target_col=None)
    print(f"Numeric features for noise addition: {numeric_cols[:10]}..." if len(numeric_cols) > 10 else f"Numeric features: {numeric_cols}")

    X_expanded = add_gaussian_noise(X_smoter_df, SIGMA, numeric_cols, n_copies=N_GAUSSIAN_COPIES)
    y_expanded = np.tile(y_smoter, N_GAUSSIAN_COPIES + 1)
    print(f"After Gaussian augmentation: {len(X_expanded)} samples ({N_GAUSSIAN_COPIES + 1}x of SMOTER output)")

    # 7. Define models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'MLP': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=RANDOM_STATE)
    }

    # 8. Evaluate
    results = evaluate_models(X_expanded, y_expanded, models, cv_folds=CV_FOLDS)

    # 9. Sort by CV R² (mean) and get top 3
    sorted_models = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    top3 = sorted_models[:3]

    print("\nModel Performance (CV R² Mean ± Std):")
    for model_name, (mean, std) in sorted_models:
        print(f"  {model_name:20s}: {mean:.4f} ± {std:.4f}")

    print(f"\nTop 3 Models (by CV R²):")
    for i, (model_name, (mean, _)) in enumerate(top3, 1):
        print(f"  {i}. {model_name} (CV R² = {mean:.4f})")

    all_results[name] = {
        'full_ranking': sorted_models,
        'top3': [model_name for model_name, _ in top3]
    }

# ====================== Summary ======================
print("\n\n" + "=" * 60)
print("Final Recommended Baseline Models (by CV R²)")
print("=" * 60)
for dataset, info in all_results.items():
    print(f"\n{dataset}:")
    for i, model_name in enumerate(info['top3'], 1):
        print(f"  {i}. {model_name}")