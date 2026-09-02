"""
Supplementary Code - GBDT Fine-Tuning with SMOGN Augmentation (C14 Dataset)

This script performs fine-tuning for GBDT with SMOGN augmentation
on the C14 Laves phase dataset.
"""

import pandas as pd
import numpy as np
import random
import os
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

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
def smogn_augmentation(X, y, smoter_ratio=1.0, noise_ratio=0.05, k=7,
                       bins=10, extreme_factor=2.0, dist_threshold_factor=0.8,
                       random_state=SEED):
    np.random.seed(random_state)
    X = X.reset_index(drop=True).copy()
    y = y.reset_index(drop=True).copy()
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
def preprocess_dataframe(df):
    df = clean_column_names(df)
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        print(f"Removing constant columns: {const_cols}")
        df.drop(columns=const_cols, inplace=True)
    target_col = df.columns[-1]
    print(f"Target column: {target_col}")
    return df, target_col

# ====================== Feature Selection ======================
def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    gbdt_temp = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    gbdt_temp.fit(X_train, y_train)
    importances = gbdt_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    print(f"Selected {n_selected} features (95% importance): {list(selected_features)}")
    return selected_features

# ====================== Main Program ======================
if __name__ == "__main__":
    # Dataset path (English column names)
    file_path = "C14_data.csv"
    target_col = "Max_H2_Uptake_wt_pct"

    # SMOGN augmentation parameters
    SMOTER_RATIO = 3
    NOISE_RATIO = 0.05

    # 1. Load and preprocess
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"Original data shape: {df.shape}")
    df, target_col = preprocess_dataframe(df)
    print(f"After preprocessing: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train/test split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_orig = X_train_orig.reset_index(drop=True)
    y_train_orig = y_train_orig.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Original training: {X_train_orig.shape}, Test: {X_test.shape}")

    # 3. Feature selection
    selected_features = select_features_by_importance(X_train_orig, y_train_orig, target_ratio=0.95)
    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]
    print(f"After feature selection: {X_train_orig.shape}")

    # 4. SMOGN augmentation
    print(f"\n{'='*70}")
    print(f"SMOGN Augmentation: smoter_ratio = {SMOTER_RATIO} (total = {1+SMOTER_RATIO}x), noise_ratio = {NOISE_RATIO}")
    print('='*70)
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=SMOTER_RATIO,
        noise_ratio=NOISE_RATIO,
        k=7, bins=10, extreme_factor=2.0, dist_threshold_factor=0.8
    )
    print(f"Augmented training size: {X_train_aug.shape[0]}")

    # 5. Randomized search
    param_dist = {
        'n_estimators': [182],
        'max_depth': [7],
        'min_samples_split': [9],
        'min_samples_leaf': [8],
        'learning_rate': [0.08],
        'subsample': [0.92],
        'max_features': [0.8],
        'ccp_alpha': [4e-6],
    }

    gbdt = GradientBoostingRegressor(
        random_state=SEED,
        n_iter_no_change=10,
        tol=1e-4,
        validation_fraction=0.1,
        verbose=0
    )

    random_search = RandomizedSearchCV(
        gbdt, param_distributions=param_dist,
        n_iter=150,
        cv=5, scoring='r2',
        n_jobs=1, verbose=1, random_state=SEED
    )
    random_search.fit(X_train_aug, y_train_aug)

    best_gbdt = random_search.best_estimator_
    best_params = random_search.best_params_

    cv_results = random_search.cv_results_
    best_index = random_search.best_index_
    cv_mean = cv_results['mean_test_score'][best_index]
    cv_std = cv_results['std_test_score'][best_index]

    print(f"\nBest parameters: {best_params}")
    print(f"CV R² (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    def metrics(model, X, y):
        pred = model.predict(X)
        return (
            r2_score(y, pred),
            np.sqrt(mean_squared_error(y, pred)),
            mean_absolute_error(y, pred)
        )

    train_r2, train_rmse, train_mae = metrics(best_gbdt, X_train_aug, y_train_aug)
    test_r2, test_rmse, test_mae = metrics(best_gbdt, X_test, y_test)
    overfit_gap = train_r2 - test_r2

    print(f"\nTraining: R²={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
    print(f"Test: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
    print(f"Overfitting gap: {overfit_gap:.4f}")

    # Save results
    result_dict = {
        'smoter_ratio': SMOTER_RATIO,
        'total_multiple': 1 + SMOTER_RATIO,
        'noise_ratio': NOISE_RATIO,
        'train_size_aug': X_train_aug.shape[0],
        'best_params': str(best_params),
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'train_r2': train_r2, 'train_rmse': train_rmse, 'train_mae': train_mae,
        'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mae': test_mae,
        'overfit_gap': overfit_gap
    }
