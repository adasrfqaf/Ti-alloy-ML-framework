"""
Supplementary Code - Random Forest Tuning with Gaussian Noise + SMOTER Augmentation (C14 Dataset)

This script performs tuning for Random Forest with combined Gaussian noise
and SMOTER augmentation on the C14 Laves phase dataset.
"""

import pandas as pd
import numpy as np
import random
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
SEED = 49
def set_global_seed(seed=49):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = "C14_data.csv"
TARGET_COL = "Max_H2_Uptake_wt_pct"
TEST_SIZE = 0.2
CV_FOLDS = 5

# Augmentation parameters
NOISE_COPIES = 3
NOISE_RATIOS = [0.10]
SMOTER_RATIOS = [0.05]
EXTREME_FACTOR = 1.2

# Random Forest hyperparameter search space
param_grid = {
    'n_estimators': [180, 200, 230],
    'max_depth': [5, 6, 7],
    'min_samples_split': [6, 7],
    'min_samples_leaf': [4],
    'max_features': [0.7, 0.75, 0.8],
    'bootstrap': [False]
}

# ====================== Data Loading ======================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    print(f"Original features: {X.shape[1]}, samples: {len(X)}")
    return X, y

def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    from sklearn.ensemble import RandomForestRegressor
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    rf_temp.fit(X_train, y_train)
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    print(f"Selected {n_selected} features ({target_ratio*100}% cumulative importance)")
    print(f"Selected features: {list(selected_features)}")
    return selected_features

def add_gaussian_noise(X, y, sigma_ratio, n_copies):
    X_noisy_list, y_noisy_list = [], []
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            std_col = X[col].std()
            if std_col > 0:
                noise = np.random.normal(0, sigma_ratio * std_col, len(X))
                X_copy[col] += noise
        X_noisy_list.append(X_copy)
        y_noisy_list.append(y)
    X_noisy = pd.concat(X_noisy_list, ignore_index=True)
    y_noisy = np.concatenate(y_noisy_list)
    return X_noisy, y_noisy

def smoter_manual(X, y, smoter_ratio, extreme_factor=EXTREME_FACTOR, k=5, bins=10):
    np.random.seed(SEED)
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)
    if n_generate <= 0:
        return pd.DataFrame(columns=X.columns), np.array([])
    y_percentile = np.percentile(y, np.linspace(0, 100, bins+1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices-1]
    sample_weights /= sample_weights.sum()
    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)
    X_smoter_list, y_smoter_list = [], []
    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y[idx]
        distances, indices = nn.kneighbors(x_seed.reshape(1,-1), n_neighbors=k+1)
        neighbor_idx = np.random.choice(indices[0][1:])
        x_neighbor = X.iloc[neighbor_idx].values
        y_neighbor = y[neighbor_idx]
        lam = np.random.uniform()
        x_new = x_seed + lam*(x_neighbor - x_seed)
        y_new = y_seed + lam*(y_neighbor - y_seed)
        X_smoter_list.append(x_new)
        y_smoter_list.append(y_new)
    X_smoter = pd.DataFrame(X_smoter_list, columns=X.columns)
    y_smoter = np.array(y_smoter_list)
    return X_smoter, y_smoter

def augment_data(X, y, noise_ratio, smoter_ratio):
    if smoter_ratio > 0:
        X_smoter, y_smoter = smoter_manual(X, y, smoter_ratio)
    else:
        X_smoter, y_smoter = pd.DataFrame(columns=X.columns), np.array([])
    X_combined = pd.concat([X, X_smoter], ignore_index=True)
    y_combined = np.concatenate([y, y_smoter])
    if noise_ratio > 0:
        X_noise, y_noise = add_gaussian_noise(X_combined, y_combined, noise_ratio, NOISE_COPIES)
        X_final = pd.concat([X_combined, X_noise], ignore_index=True)
        y_final = np.concatenate([y_combined, y_noise])
    else:
        X_final, y_final = X_combined, y_combined
    return X_final, y_final

def evaluate(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return r2, rmse, mae

# ====================== Save Test Predictions ======================
def save_predictions(y_test, y_pred, filename):
    results = pd.DataFrame({
        'true_capacity': y_test,
        'pred_capacity': y_pred
    })
    results.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"   Samples: {len(results)}")
    print(f"   Capacity range: {results['true_capacity'].min():.3f} ~ {results['true_capacity'].max():.3f} wt.%")

# ====================== Main Program ======================
if __name__ == '__main__':
    print("Loading data...")
    X, y = load_data(FILE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"Original training: {X_train.shape}, Test: {X_test.shape}")

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Feature selection
    print("\nPerforming feature selection (95% cumulative importance)...")
    selected_features = select_features_by_importance(X_train_scaled, y_train, target_ratio=0.95)
    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]
    print(f"After feature selection: {X_train_selected.shape}")

    # Iterate over augmentation combinations
    all_results = []
    total_combos = len(NOISE_RATIOS) * len(SMOTER_RATIOS)
    combo_idx = 0

    for noise in NOISE_RATIOS:
        for smoter in SMOTER_RATIOS:
            combo_idx += 1
            print(f"\n{'=' * 60}")
            print(f"Combination {combo_idx}/{total_combos}: noise={noise}, smoter={smoter} (copies={NOISE_COPIES})")
            print('=' * 60)

            X_aug, y_aug = augment_data(X_train_selected, y_train, noise, smoter)
            print(f"Augmented training size: {len(X_aug)} (original {len(X_train_selected)})")

            # Randomized search
            rf_base = RandomForestRegressor(random_state=SEED, n_jobs=1)
            random_search = RandomizedSearchCV(
                rf_base, param_grid, n_iter=30, cv=CV_FOLDS, scoring='r2',
                random_state=SEED, n_jobs=1, verbose=1
            )
            random_search.fit(X_aug, y_aug)

            best_params = random_search.best_params_
            best_model = random_search.best_estimator_

            # Cross-validation
            cv_scores = cross_val_score(best_model, X_aug, y_aug, cv=CV_FOLDS, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            print(f"CV R² = {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"Best hyperparameters: {best_params}")

            train_r2, train_rmse, train_mae = evaluate(best_model, X_aug, y_aug)
            test_r2, test_rmse, test_mae = evaluate(best_model, X_test_selected, y_test)

            print(f"Augmented Training: R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
            print(f"Test (Original): R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

            all_results.append({
                'noise_ratio': noise,
                'smoter_ratio': smoter,
                'best_params': best_params,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'model': best_model
            })

    # Summary
    print("\n" + "=" * 80)
    print("Random Forest Results Summary (Gaussian Noise + SMOTER):")
    summary = []
    for res in all_results:
        summary.append({
            'noise': res['noise_ratio'],
            'smoter': res['smoter_ratio'],
            'n_estimators': res['best_params'].get('n_estimators'),
            'max_depth': res['best_params'].get('max_depth'),
            'min_samples_split': res['best_params'].get('min_samples_split'),
            'min_samples_leaf': res['best_params'].get('min_samples_leaf'),
            'max_features': res['best_params'].get('max_features'),
            'bootstrap': res['best_params'].get('bootstrap'),
            'CV_R2': f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
            'Train_R2': f"{res['train_r2']:.4f}",
            'Test_R2': f"{res['test_r2']:.4f}",
            'Overfit': f"{res['train_r2'] - res['test_r2']:.4f}"
        })

    # Best model
    best_test_idx = np.argmax([r['test_r2'] for r in all_results])
    best_overall = all_results[best_test_idx]
    print(f"\nBest Test R² Random Forest model: noise={best_overall['noise_ratio']}, smoter={best_overall['smoter_ratio']}")
    print(f"CV R² = {best_overall['cv_mean']:.4f} ± {best_overall['cv_std']:.4f}")
    print(f"Test R² = {best_overall['test_r2']:.4f}, RMSE = {best_overall['test_rmse']:.4f}, MAE = {best_overall['test_mae']:.4f}")

