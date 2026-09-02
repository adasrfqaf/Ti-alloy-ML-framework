"""
Supplementary Code for C14 Alloy Screening Phase - Model Screening with SMOTER Augmentation

This script evaluates six candidate models (Random Forest, GBDT, MLP, SVR, XGBoost, LightGBM)
under SMOTER augmentation and selects the top 3 based on CV R² performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# All models to evaluate (n_jobs=1 for reproducibility)
MODELS = {
    'RandomForest': RandomForestRegressor(random_state=SEED, n_jobs=1),
    'GradientBoosting': GradientBoostingRegressor(random_state=SEED),
    'MLP': MLPRegressor(random_state=SEED, max_iter=1000, early_stopping=True, validation_fraction=0.1),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(random_state=SEED, verbosity=0, n_jobs=1),
    'LightGBM': LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1)
}


def load_bcc_data():
    """
    Load C14 Alloy Screening dataset with English column names.
    No column mapping is performed.
    """
    df = pd.read_csv('BCC_data.csv', encoding='utf-8-sig')

    # Convert boolean columns to 0/1 if any
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Separate features and target
    target = 'Max_H2_Uptake_wt_pct'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    X = df.drop(columns=[target])
    y = df[target].values

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    # Fill missing values with mean
    X = X.fillna(X.mean())

    return X, y


def manual_smoter_interpolate(X_train, y_train, mode='high', high_ratio=0.1, low_ratio=0.1, k=5):
    """
    Manual SMOTER interpolation (linear interpolation without noise).

    Supports modes:
    - 'high': augment high-value tail
    - 'low': augment low-value tail
    - 'both': augment both tails
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    # Determine minority samples based on mode
    if mode == 'high':
        threshold = np.percentile(y, 100 * (1 - high_ratio))
        minority_idx = np.where(y >= threshold)[0]
        desc = f"High-value tail (≥ {threshold:.3f}, top {high_ratio * 100:.0f}%)"
    elif mode == 'low':
        threshold = np.percentile(y, 100 * low_ratio)
        minority_idx = np.where(y <= threshold)[0]
        desc = f"Low-value tail (≤ {threshold:.3f}, bottom {low_ratio * 100:.0f}%)"
    elif mode == 'both':
        low_th = np.percentile(y, 100 * low_ratio)
        high_th = np.percentile(y, 100 * (1 - high_ratio))
        minority_idx = np.where((y <= low_th) | (y >= high_th))[0]
        desc = f"Both tails (low ≤ {low_th:.3f}, high ≥ {high_th:.3f})"
    else:
        raise ValueError("mode must be 'high', 'low', or 'both'")

    print(f"  Augmentation mode: {desc}")
    print(f"  Selected samples: {len(minority_idx)}")

    if len(minority_idx) < 2:
        print("  Warning: Insufficient samples for interpolation, returning original data")
        return X_train, y_train

    # Compute K-nearest neighbors for minority samples
    nbrs = NearestNeighbors(n_neighbors=min(k, len(minority_idx) - 1), metric='euclidean').fit(X[minority_idx])
    synthetic_X = []
    synthetic_y = []

    for idx in minority_idx:
        distances, indices = nbrs.kneighbors(X[idx].reshape(1, -1))
        # Randomly select a neighbor (excluding self)
        neighbor_local_idx = np.random.choice(indices[0][1:], 1)[0]
        neighbor_global_idx = minority_idx[neighbor_local_idx]
        # Linear interpolation
        gap = np.random.uniform(0, 1)
        synthetic_sample = X[idx] + gap * (X[neighbor_global_idx] - X[idx])
        synthetic_target = y[idx] + gap * (y[neighbor_global_idx] - y[idx])
        synthetic_X.append(synthetic_sample)
        synthetic_y.append(synthetic_target)

    X_aug = np.vstack([X, synthetic_X])
    y_aug = np.concatenate([y, synthetic_y])

    if isinstance(X_train, pd.DataFrame):
        X_aug = pd.DataFrame(X_aug, columns=X_train.columns)

    return X_aug, y_aug


def evaluate_models(X_train, X_test, y_train, y_test):
    """
    Evaluate all models and return DataFrame sorted by CV R².
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name, model in MODELS.items():
        # Cross-validation R²
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=1)
        cv_r2 = cv_scores.mean()

        # Train and test
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)

        results.append({
            'Model': name,
            'CV_R2': cv_r2,
            'Test_R2': test_r2,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae
        })
        print(f"    {name}: CV R² = {cv_r2:.4f}, Test R² = {test_r2:.4f}")

    # Sort by CV R² descending
    return pd.DataFrame(results).sort_values('CV_R2', ascending=False)


# ====================== Main Program ======================
print("=" * 70)
print("C14 Alloy Screening Dataset - Manual SMOTER Interpolation (All Models Evaluation)")
print("=" * 70)

X, y = load_bcc_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Original training set: {len(X_train)}, Test set: {len(X_test)}")

# Apply augmentation strategy (e.g., high-value tail top 10%)
print("\nAugmentation strategy: High-value tail (top 10%)")
X_train_aug, y_train_aug = manual_smoter_interpolate(X_train, y_train, mode='high', high_ratio=0.1, k=5)
print(f"Augmented training set: {len(X_train_aug)} ({len(X_train_aug) / len(X_train):.2f}x)")

print("\nEvaluating all models...")
results_df = evaluate_models(X_train_aug, X_test, y_train_aug, y_test)

print("\n" + "=" * 70)
print("Top 3 Models (sorted by CV R²):")
print(results_df.head(3).to_string(index=False))

# Save results
results_df.to_csv('BCC_all_models_SMOTER_results.csv', index=False)
print("\nFull results saved to: BCC_all_models_SMOTER_results.csv")