"""
Supplementary Code - LightGBM Greedy Grid Search with Gaussian Noise + SMOTER Augmentation (C14 Alloy Screening Dataset)

This script performs greedy grid search for LightGBM with combined Gaussian noise
and SMOTER augmentation on the C14 Alloy Screening phase dataset.
"""

import pandas as pd
import numpy as np
import random
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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


SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = "BCC_data.csv"
TARGET_COL = "Max_H2_Uptake_wt_pct"
TEST_SIZE = 0.2
CV_FOLDS = 5

# 基于最佳结果微调
NOISE_RATIOS = [0.01]
SMOTER_RATIOS = [0.08]

# 在最佳参数附近微调
PARAM_GROUP1 = {
    'num_leaves': [7],
    'max_depth': [6],
    'min_child_samples': [25],
    'reg_lambda': [0.5]
}
PARAM_GROUP2 = {
    'learning_rate': [0.03],
    'n_estimators': [430],
    'subsample': [0.6],
    'colsample_bytree': [0.7]
}

FEATURE_IMPORTANCE_THRESHOLD = 0.95
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
    lgb_temp = lgb.LGBMRegressor(n_estimators=100, random_state=SEED, verbose=-1)
    lgb_temp.fit(X_train, y_train)
    importances = lgb_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio * np.sum(importances)) + 1
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


# ====================== Greedy Grid Search ======================
def greedy_grid_search_lgbm(X_train, y_train):
    base_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=SEED,
        verbose=-1
    )
    gs1 = GridSearchCV(base_lgb, PARAM_GROUP1, cv=CV_FOLDS, scoring='r2', n_jobs=1, verbose=0)
    gs1.fit(X_train, y_train)
    best_group1 = gs1.best_params_

    base_lgb2 = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **best_group1)
    gs2 = GridSearchCV(base_lgb2, PARAM_GROUP2, cv=CV_FOLDS, scoring='r2', n_jobs=1, verbose=0)
    gs2.fit(X_train, y_train)
    best_params = {**best_group1, **gs2.best_params_}
    best_estimator = gs2.best_estimator_

    # Recalculate CV R² mean and std
    cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=CV_FOLDS, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

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
    print(f"\nPerforming feature selection ({FEATURE_IMPORTANCE_THRESHOLD*100}% cumulative importance)...")
    selected_features = select_features_by_importance(X_train_scaled, y_train, target_ratio=FEATURE_IMPORTANCE_THRESHOLD)
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

            best_params, cv_mean, cv_std, best_model = greedy_grid_search_lgbm(X_aug, y_aug)
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
    print("LightGBM Results Summary (Gaussian Noise + SMOTER, 3 copies):")
    summary = []
    for res in all_results:
        params = res['best_params']
        summary.append({
            'noise': res['noise_ratio'],
            'smoter': res['smoter_ratio'],
            'num_leaves': params.get('num_leaves'),
            'max_depth': params.get('max_depth'),
            'min_child_samples': params.get('min_child_samples'),
            'reg_lambda': params.get('reg_lambda'),
            'learning_rate': params.get('learning_rate'),
            'n_estimators': params.get('n_estimators'),
            'subsample': params.get('subsample'),
            'colsample_bytree': params.get('colsample_bytree'),
            'CV_R2': f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
            'Train_R2': f"{res['train_r2']:.4f}",
            'Test_R2': f"{res['test_r2']:.4f}",
            'Overfit': f"{res['train_r2'] - res['test_r2']:.4f}"
        })
