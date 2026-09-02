"""
Supplementary Code - SVR Regularized Tuning with SMOGN Augmentation (TiFe Dataset)

This script performs regularized tuning for SVR with SMOGN augmentation
on the TiFe phase dataset.
"""

import pandas as pd
import numpy as np
import random
import os
import re
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ====================== Fixed Random Seed ======================
SEED = 49


def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(SEED)


# ====================== Column Name Cleaning ======================
def clean_column_names(df):
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        cleaned = re.sub(r'[\[\]\(\),;:\s]+', '_', str(col))
        cleaned = cleaned.strip('_')
        if not cleaned:
            cleaned = col
        new_cols[col] = cleaned
    df.rename(columns=new_cols, inplace=True)
    return df


# ====================== SMOGN Implementation ======================
def smogn_augmentation(X, y, smoter_ratio=1.0, noise_ratio=0.03, k=7,
                       bins=10, extreme_factor=1.8, dist_threshold_factor=0.5,
                       random_state=SEED):
    np.random.seed(random_state)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X = X.copy()
    y = y.copy()
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)
    if n_generate <= 0:
        return X, y

    y_percentile = np.percentile(y, np.linspace(0, 100, bins + 1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices - 1] if bins > 1 else np.ones(n_original)
    sample_weights = sample_weights / sample_weights.sum()

    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)

    all_distances = []
    for i in range(n_original):
        dist, _ = nn.kneighbors(X.iloc[i].values.reshape(1, -1), n_neighbors=k + 1)
        all_distances.append(dist[0][1:].mean())
    global_avg_dist = np.mean(all_distances)
    dist_threshold = global_avg_dist * dist_threshold_factor

    X_list = [X]
    y_list = [y]

    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y.iloc[idx]
        distances, indices = nn.kneighbors(x_seed.reshape(1, -1), n_neighbors=k + 1)
        neighbor_indices = indices[0][1:]
        avg_dist_to_neighbors = distances[0][1:].mean()

        if avg_dist_to_neighbors > dist_threshold:
            x_new = x_seed.copy()
            for i_col, col in enumerate(X.columns):
                std_col = X[col].std()
                if std_col > 0:
                    x_new[i_col] += np.random.normal(0, noise_ratio * std_col)
            y_new = y_seed
        else:
            neighbor_idx = np.random.choice(neighbor_indices)
            x_neighbor = X.iloc[neighbor_idx].values
            y_neighbor = y.iloc[neighbor_idx]
            lam = np.random.uniform()
            x_new = x_seed + lam * (x_neighbor - x_seed)
            y_new = y_seed + lam * (y_neighbor - y_seed)

        X_list.append(pd.DataFrame([x_new], columns=X.columns))
        y_list.append([y_new])

    X_aug = pd.concat(X_list, ignore_index=True)
    y_aug = np.concatenate(y_list)
    return X_aug, y_aug


# ====================== Data Preprocessing ======================
def preprocess_dataframe(df, target_col):
    df = clean_column_names(df)
    if target_col not in df.columns:
        cleaned_target = re.sub(r'[\[\]\(\),;:\s]+', '_', target_col).strip('_')
        if cleaned_target in df.columns:
            target_col = cleaned_target
            print(f"Target column auto-mapped to: {target_col}")
        else:
            raise ValueError(f"Target column '{target_col}' not found")
    df = df.copy()
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"Target column {target_col} is not numeric")
    df = df[numeric_cols]
    for col in df.columns:
        if df[col].nunique() <= 1:
            print(f"Removing constant column: {col}")
            df.drop(columns=[col], inplace=True)
    return df, target_col


# ====================== Feature Selection ======================
def select_features_by_count(X_train, y_train, n_features=18):
    temp_model = XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0)
    temp_model.fit(X_train, y_train)
    importances = temp_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    n_select = min(n_features, len(indices))
    selected_features = X_train.columns[indices[:n_select]]
    print(f"Feature importance ranking (XGBoost), selected top {n_select} features")
    print("Selected features:", list(selected_features))
    return selected_features


# ====================== SVR Grid Search ======================
def manual_grid_search_svr(X_train, y_train, X_test, y_test, param_grid, cv_folds=5):
    best_score = -np.inf
    best_params = None
    best_model = None

    total = len(ParameterGrid(param_grid))
    print(f"Total parameter combinations: {total}, starting search...")

    for i, params in enumerate(ParameterGrid(param_grid)):
        model = SVR(**params)
        # Cross-validation evaluation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_cv_tr, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_tr, y_cv_val = y_train[train_idx], y_train[val_idx]
            model_cv = SVR(**params)
            model_cv.fit(X_cv_tr, y_cv_tr)
            y_cv_pred = model_cv.predict(X_cv_val)
            cv_scores.append(r2_score(y_cv_val, y_cv_pred))
        mean_score = np.mean(cv_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_model = SVR(**params)
            best_model.fit(X_train, y_train)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total}, Best CV R2 so far: {best_score:.4f}")

    print(f"Search completed. Best CV R2: {best_score:.4f}")

    # Cross-validation evaluation for best model
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    cv_r2_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_cv_tr, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_tr, y_cv_val = y_train[train_idx], y_train[val_idx]
        model_cv = SVR(**best_params)
        model_cv.fit(X_cv_tr, y_cv_tr)
        y_cv_pred = model_cv.predict(X_cv_val)
        cv_r2_scores.append(r2_score(y_cv_val, y_cv_pred))
    cv_r2_mean = np.mean(cv_r2_scores)
    cv_r2_std = np.std(cv_r2_scores)

    # Final evaluation
    y_train_pred = best_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    y_test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return best_model, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae, best_params, cv_r2_mean, cv_r2_std


# ====================== Main Program ======================
if __name__ == "__main__":
    FEATURE_COUNT = 18
    SMOTER_RATIO = 4

    # SVR parameter grid (C, gamma, epsilon for RBF kernel)
    PARAM_GRID_SVR = {
        'C': [0.312],
        'gamma': [0.105],
        'epsilon': [0.09],
        'kernel': ['rbf']
    }

    # Dataset path (CSV with English column names)
    file_path = "TiFe_data.csv"
    target_col = "Max_H2_Uptake_wt_pct"

    print("=" * 70)
    print("SVR Regularized Tuning (Augmentation Ratio=4, RBF Kernel)")
    print("=" * 70)

    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"Original data shape: {df.shape}")
    df, target_col = preprocess_dataframe(df, target_col)
    print(f"After preprocessing: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_orig = X_train_orig.reset_index(drop=True)
    y_train_orig = y_train_orig.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Original training: {X_train_orig.shape}, Test: {X_test.shape}")

    selected_features = select_features_by_count(X_train_orig, y_train_orig, n_features=FEATURE_COUNT)
    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]
    print(f"After feature selection: {X_train_orig.shape}")

    print(f"\nApplying SMOGN augmentation: smoter_ratio = {SMOTER_RATIO}")
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=SMOTER_RATIO,
        noise_ratio=0.08,
        k=7,
        bins=10,
        extreme_factor=1.8,
        dist_threshold_factor=0.5
    )
    print(f"Augmented training size: {X_train_aug.shape[0]}")

    scaler = StandardScaler()
    X_train_aug_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    print("\nStarting SVR grid search...")
    best_model, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae, best_params, cv_r2_mean, cv_r2_std = manual_grid_search_svr(
        X_train_aug_scaled, y_train_aug, X_test_scaled, y_test, PARAM_GRID_SVR, cv_folds=5
    )

    print("\n" + "=" * 70)
    print("SVR Tuning Results")
    print("=" * 70)
    print(f"Best parameters: {best_params}")
    print(f"Augmented Training: R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
    print(f"Cross-Validation (5-fold): R2={cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"Test (Original Distribution): R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

