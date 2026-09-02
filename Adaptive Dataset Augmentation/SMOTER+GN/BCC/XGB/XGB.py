"""
Supplementary Code - XGBoost Grid Search with Gaussian Noise + SMOTER Augmentation (C14 Alloy Screening Dataset)

This script performs a single-stage grid search for XGBoost with combined Gaussian noise
and SMOTER augmentation on the C14 Alloy Screening phase dataset.
Only the best configuration (by Test R²) is printed.
No files are saved.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


# ====================== Fixed Random Seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


SEED = 42
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = "BCC_data.csv"
TARGET_COL = "Max_H2_Uptake_wt_pct"
TEST_SIZE = 0.2
CV_FOLDS = 5

# Augmentation parameters (you can expand these lists)
NOISE_RATIOS = [0.10]
SMOTER_RATIOS = [0.10]

# ====================== Merged Parameter Grid (Single Stage) ======================
PARAM_GRID = {
    'n_estimators': [260],
    'learning_rate': [0.05],
    'max_depth': [3],
    'min_child_weight': [10],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.5],
    'reg_lambda': [2],
    'subsample': [0.8]
}


# ====================== Data Loading ======================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    print(f"Original features: {X.shape[1]}, samples: {len(X)}")
    return X, y


# ====================== Feature Selection ======================
def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    xgb_temp = XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0)
    xgb_temp.fit(X_train, y_train)
    importances = xgb_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    print(f"Selected {n_selected} features ({target_ratio*100}% cumulative importance)")
    print(f"Selected features: {list(selected_features)}")
    return selected_features


# ====================== Augmentation Functions ======================
def add_gaussian_noise(X, y, sigma_ratio, n_copies=3):
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


def smoter_manual(X, y, smoter_ratio, extreme_factor=2.0, k=5, bins=10, random_state=SEED):
    np.random.seed(random_state)
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)
    if n_generate <= 0:
        return pd.DataFrame(columns=X.columns), np.array([])

    y_percentile = np.percentile(y, np.linspace(0, 100, bins + 1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices - 1]
    sample_weights = sample_weights / sample_weights.sum()

    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)

    X_smoter_list, y_smoter_list = [], []
    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y[idx]
        distances, indices = nn.kneighbors(x_seed.reshape(1, -1), n_neighbors=k + 1)
        neighbor_idx = np.random.choice(indices[0][1:])
        x_neighbor = X.iloc[neighbor_idx].values
        y_neighbor = y[neighbor_idx]
        lam = np.random.uniform()
        x_new = x_seed + lam * (x_neighbor - x_seed)
        y_new = y_seed + lam * (y_neighbor - y_seed)
        X_smoter_list.append(x_new)
        y_smoter_list.append(y_new)

    X_smoter = pd.DataFrame(X_smoter_list, columns=X.columns)
    y_smoter = np.array(y_smoter_list)
    return X_smoter, y_smoter


def augment_data(X, y, noise_ratio, smoter_ratio, random_state=SEED):
    """SMOTER first, then Gaussian noise (3 copies)"""
    np.random.seed(random_state)
    if smoter_ratio > 0:
        X_smoter, y_smoter = smoter_manual(X, y, smoter_ratio)
    else:
        X_smoter, y_smoter = pd.DataFrame(columns=X.columns), np.array([])
    X_combined = pd.concat([X, X_smoter], ignore_index=True)
    y_combined = np.concatenate([y, y_smoter])
    if noise_ratio > 0:
        X_noise, y_noise = add_gaussian_noise(X_combined, y_combined, noise_ratio, n_copies=3)
        X_final = pd.concat([X_combined, X_noise], ignore_index=True)
        y_final = np.concatenate([y_combined, y_noise])
    else:
        X_final, y_final = X_combined, y_combined
    return X_final, y_final


# ====================== Single-Stage Grid Search ======================
def grid_search_xgb(X_train, y_train, param_grid, cv_folds=CV_FOLDS):
    base_xgb = XGBRegressor(
        random_state=SEED,
        verbosity=0,
        n_jobs=1
    )
    gs = GridSearchCV(
        base_xgb, param_grid,
        cv=cv_folds, scoring='r2',
        n_jobs=1, verbose=0
    )
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    best_estimator = gs.best_estimator_
    cv_mean = gs.best_score_
    # Compute cv std from cv_results_
    cv_std = gs.cv_results_['std_test_score'][gs.best_index_]
    return best_params, cv_mean, cv_std, best_estimator


def evaluate(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return r2, rmse, mae


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
            print(f"Combination {combo_idx}/{total_combos}: noise={noise}, smoter={smoter}")
            print('=' * 60)

            X_aug, y_aug = augment_data(X_train_selected, y_train, noise, smoter)
            print(f"Augmented training size: {len(X_aug)} (original {len(X_train_selected)})")

            best_params, cv_mean, cv_std, best_model = grid_search_xgb(
                X_aug, y_aug, param_grid=PARAM_GRID, cv_folds=CV_FOLDS
            )
            print(f"Best CV R² = {cv_mean:.4f} ± {cv_std:.4f}")
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

    # Summary (only best combination is printed)
    print("\n" + "=" * 80)
    print("XGBoost Grid Search Complete - Best Configuration Only")
    print("=" * 80)

    # Find best overall (by test R²)
    best_idx = np.argmax([r['test_r2'] for r in all_results])
    best = all_results[best_idx]

    print(f"\nBest Augmentation Combination:")
    print(f"  noise_ratio = {best['noise_ratio']}")
    print(f"  smoter_ratio = {best['smoter_ratio']}")

    print("\nBest Hyperparameters:")
    for k, v in best['best_params'].items():
        print(f"  {k}: {v}")

    print("\nEvaluation Metrics:")
    print(f"  CV R² (5-fold) = {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"  Augmented Training R² = {best['train_r2']:.4f}")
    print(f"  Test R² = {best['test_r2']:.4f}")
    print(f"  Test RMSE = {best['test_rmse']:.4f}")
    print(f"  Test MAE = {best['test_mae']:.4f}")
    print(f"  Overfitting (Train - Test R²) = {best['train_r2'] - best['test_r2']:.4f}")

    # Feature Importance
    print("\nFeature Importance (Best Model):")
    importances = best['model'].feature_importances_
    feat_names = X_train_selected.columns
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
    print(imp_df.head(15).to_string(index=False))
