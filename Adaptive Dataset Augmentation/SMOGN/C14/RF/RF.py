"""
Supplementary Code - Random Forest Fine-Tuning with SMOGN Augmentation (C14 Dataset)

This script performs fine-tuning for Random Forest with SMOGN augmentation
on the C14 Laves phase dataset.
"""

import pandas as pd
import numpy as np
import pickle
import re
import random
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

# ==================== Fixed Random Seed ====================
SEED = 49
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_global_seed(SEED)

# ==================== Column Name Cleaning ====================
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

# ==================== SMOGN Implementation ====================
def smogn_augmentation(X, y, smoter_ratio=1.0, noise_ratio=0.05, k=7,
                       bins=10, extreme_factor=2.0, dist_threshold_factor=0.8,
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

# ==================== Data Preprocessing ====================
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
    if df.isnull().any().any():
        print("Missing values found, filling with 0")
        df = df.fillna(0)
    for col in df.columns:
        if df[col].nunique() <= 1:
            print(f"Removing constant column: {col}")
            df.drop(columns=[col], inplace=True)
    return df, target_col

# ==================== Main Program ====================
if __name__ == "__main__":
    # 1. Read data
    file_path = "C14_data.csv"
    target_col = "Max_H2_Uptake_wt_pct"
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df, target_col = preprocess_dataframe(df, target_col)

    # Print column names for debugging
    print("Available columns:", df.columns.tolist())

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train/test split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_orig = X_train_orig.reset_index(drop=True)
    y_train_orig = y_train_orig.reset_index(drop=True)

    # 3. Selected features (English column names - check available columns)
    selected_features = [
        'Element_at_pct_Ti',
        'Element_at_pct_Mn',
        'Element_at_pct_V',
        'Lattice_Parameter_c_Å',
        'Element_at_pct_Al',
        'Element_at_pct_Cu',
        'Hydrogen_Absorption_Cycles',
        'Element_at_pct_Nb',
        'Initial_Hydrogen_Pressure_MPa',
        'Element_at_pct_Cr',
        'Additive_Zr'
    ]
    missing = [f for f in selected_features if f not in X_train_orig.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
        print(f"Available columns: {X_train_orig.columns.tolist()}")
        # Use only available features
        selected_features = [f for f in selected_features if f in X_train_orig.columns]
        print(f"Using available features: {selected_features}")

    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]

    print(f"Original training size: {X_train_orig.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    print(f"Selected features: {selected_features}\n")

    # 4. SMOGN augmentation (3x)
    print("Performing 3x SMOGN augmentation...")
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=2.0,
        noise_ratio=0.08,
        k=7,
        bins=10,
        extreme_factor=2.0,
        dist_threshold_factor=0.8
    )
    print(f"Augmented training size: {X_train_aug.shape[0]}\n")

    # 5. Fine grid search
    param_grid_narrow = {
        'n_estimators': [148],
        'max_depth': [11],
        'min_samples_split': [6],
        'min_samples_leaf': [5],
        'max_features': [0.9],
        'bootstrap': [True]
    }

    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=1)
    random_search = RandomizedSearchCV(
        rf_base, param_grid_narrow, n_iter=100, scoring='r2', cv=5,
        random_state=SEED, n_jobs=1, verbose=1
    )
    print("Starting fine randomized search (100 iterations, 5-fold CV)...")
    random_search.fit(X_train_aug, y_train_aug)

    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_

    cv_results = random_search.cv_results_
    best_index = random_search.best_index_
    cv_mean = cv_results['mean_test_score'][best_index]
    cv_std = cv_results['std_test_score'][best_index]

    print(f"\nBest parameters: {best_params}")
    print(f"CV R² (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    # 6. Evaluate final model
    y_train_pred = best_rf.predict(X_train_aug)
    train_r2 = r2_score(y_train_aug, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_train_pred))
    train_mae = mean_absolute_error(y_train_aug, y_train_pred)

    y_test_pred = best_rf.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    overfit = train_r2 - test_r2

    print("\n" + "=" * 70)
    print("Random Forest Tuned Performance (3x Augmented Training Set)")
    print("=" * 70)
    print(f"Augmented Training: R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}, MAE = {train_mae:.4f}")
    print(f"Test (Original Distribution): R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}")
    print(f"Overfitting (Train R² - Test R²) = {overfit:.4f}")

    # 7. Feature importance
    importance = best_rf.feature_importances_
    imp_df = pd.DataFrame({'feature': selected_features, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print("\nFeature Importance (Tuned Random Forest):")
    print(imp_df.to_string(index=False))

