"""
Supplementary Code - Random Forest Fine-Tuning with Gaussian Noise Augmentation

This script performs fine-tuning for Random Forest with Gaussian noise augmentation
on the C14 Alloy Screening phase dataset. No files are saved; all results are printed to console.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')


# ====================== Set Global Random Seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = 'BCC_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 20

# Random Forest hyperparameter search space
PARAM_DIST = {
    'n_estimators': [100, 120, 150],
    'max_depth': [10, 12, 14],
    'min_samples_split': [5, 8, 10],
    'min_samples_leaf': [1, 2, 3],
    'max_features': [0.5, 0.6, 0.7],
    'bootstrap': [True],
    'max_leaf_nodes': [80, 100, 120],
    'min_impurity_decrease': [0.0005, 0.001, 0.002],
    'ccp_alpha': [0.0005, 0.001, 0.002]
}
N_ITER = 40

# Noise levels
SIGMA_RATIOS = [0.0, 0.02, 0.05, 0.08, 0.1]


# ====================== Data Loading & Preprocessing ======================
def load_data(path):
    """Load C14 Alloy Screening dataset with English column names."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    print(f"Number of numeric features retained: {X.shape[1]}")
    return X, y


def select_top_features_rf(X_train, y_train, top_n=15):
    """Select top features using Random Forest importance."""
    from sklearn.ensemble import RandomForestRegressor
    temp_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_depth=20
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_model.fit(X_scaled, y_train)
    importance = temp_model.feature_importances_
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    print(f"\nSelected top {top_n} features (by Random Forest importance):")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (importance: {importance[sorted_idx[i - 1]]:.4f})")
    return top_features


def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=3):
    """
    Add relative Gaussian noise: noise_std = sigma_ratio * feature_std.
    """
    X_noisy_list = [X]
    y_noisy_list = [y]
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            noise = np.random.normal(0, sigma_ratio * stds[col], len(X))
            X_copy[col] = X_copy[col] + noise
        X_noisy_list.append(X_copy)
        y_noisy_list.append(y)
    X_aug = pd.concat(X_noisy_list, ignore_index=True)
    y_aug = np.concatenate(y_noisy_list)
    return X_aug, y_aug


def evaluate_model(model, X, y):
    """Evaluate model and return R², MAE, RMSE."""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse


def calculate_overfitting(cv_r2_mean, train_r2, test_r2):
    """Calculate overfitting metrics."""
    train_test_gap = train_r2 - test_r2
    cv_test_gap = cv_r2_mean - test_r2
    overfitting_score = train_test_gap + max(0, cv_test_gap)

    return {
        'train_test_gap': train_test_gap,
        'cv_test_gap': cv_test_gap,
        'overfitting_score': overfitting_score
    }


# ====================== Main Pipeline ======================
print("Loading C14 Alloy Screening dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")

# Fixed train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection (based on original training set)
top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"After feature selection - Training set shape: {X_train_sel.shape}")

# Calculate feature standard deviations for relative noise
feature_stds = X_train_sel.std(axis=0)

# Store results
results = []

# Tune for each noise level
for sigma_ratio in SIGMA_RATIOS:
    print(f"\n{'=' * 50}")
    print(f"Noise level sigma_ratio = {sigma_ratio}")
    print('=' * 50)

    if sigma_ratio == 0.0:
        X_train_aug = X_train_sel
        y_train_aug = y_train
        print("Using original training set (no noise)")
    else:
        X_train_aug, y_train_aug = add_relative_gaussian_noise(
            X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=3
        )
        print(f"Augmented training set: {len(X_train_aug)} (original {len(X_train_sel)} + 3 noise copies)")

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_sel)

    # Random Forest model
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    random_search = RandomizedSearchCV(
        rf, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
        scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=1
    )
    random_search.fit(X_train_scaled, y_train_aug)

    best_params = random_search.best_params_
    best_cv_r2_mean = random_search.best_score_
    best_model = random_search.best_estimator_

    # Recalculate CV R² mean and std for the best model
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train_aug,
                                cv=CV_FOLDS, scoring='r2')
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()

    # Evaluate on training set
    r2_train, mae_train, rmse_train = evaluate_model(best_model, X_train_scaled, y_train_aug)

    # Evaluate on test set
    r2_test, mae_test, rmse_test = evaluate_model(best_model, X_test_scaled, y_test)

    # Calculate overfitting metrics
    overfitting_metrics = calculate_overfitting(cv_r2_mean, r2_train, r2_test)

    print(f"Best params: {best_params}")
    print(f"CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"Train R²: {r2_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
    print(f"Test  R²: {r2_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")
    print(f"Train-Test Gap: {overfitting_metrics['train_test_gap']:.4f}")
    print(f"Overfitting Score: {overfitting_metrics['overfitting_score']:.4f}")

    results.append({
        'sigma_ratio': sigma_ratio,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'train_r2': r2_train,
        'test_r2': r2_test,
        'train_mae': mae_train,
        'test_mae': mae_test,
        'train_rmse': rmse_train,
        'test_rmse': rmse_test,
        'train_test_gap': overfitting_metrics['train_test_gap'],
        'cv_test_gap': overfitting_metrics['cv_test_gap'],
        'overfitting_score': overfitting_metrics['overfitting_score'],
        'best_params': str(best_params)
    })

# ====================== Results Summary ======================
print("\n\n" + "=" * 80)
print("Random Forest Performance Comparison at Different Noise Levels (C14 Alloy Screening Dataset, Top 15 Features)")
print("=" * 80)

df_results = pd.DataFrame([
    {
        'σ_ratio': r['sigma_ratio'],
        'CV R² (mean±std)': f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}",
        'Train R²': r['train_r2'],
        'Test R²': r['test_r2'],
        'Train MAE': r['train_mae'],
        'Test MAE': r['test_mae'],
        'Train RMSE': r['train_rmse'],
        'Test RMSE': r['test_rmse'],
        'Train-Test Gap': r['train_test_gap'],
        'Overfitting Score': r['overfitting_score']
    }
    for r in results
])
print(df_results.to_string(index=False, float_format="%.4f"))

# Find best noise level (by Test R²)
best_idx = results[pd.Series([r['test_r2'] for r in results]).idxmax()]
best_sigma = best_idx['sigma_ratio']
print(f"\nBest Noise Level (by Test R²): σ_ratio = {best_sigma}")
print(f"   CV R² = {best_idx['cv_r2_mean']:.4f} ± {best_idx['cv_r2_std']:.4f}")
print(f"   Test R² = {best_idx['test_r2']:.4f}, "
      f"MAE = {best_idx['test_mae']:.4f}, "
      f"RMSE = {best_idx['test_rmse']:.4f}")

# Find best generalization (minimum overfitting score)
best_gen_idx = results[pd.Series([r['overfitting_score'] for r in results]).idxmin()]
best_gen_sigma = best_gen_idx['sigma_ratio']
print(f"\nBest Generalization (Minimum Overfitting Score): σ_ratio = {best_gen_sigma}")
print(f"   CV R² = {best_gen_idx['cv_r2_mean']:.4f} ± {best_gen_idx['cv_r2_std']:.4f}")
print(f"   Overfitting Score = {best_gen_idx['overfitting_score']:.4f}, "
      f"Test R² = {best_gen_idx['test_r2']:.4f}")

print("\n✅ Evaluation complete. No files were saved.")