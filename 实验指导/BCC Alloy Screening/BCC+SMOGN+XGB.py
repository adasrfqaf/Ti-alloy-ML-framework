"""
Supplementary Code - XGBoost Fine-Tuning with SMOGN Augmentation (BCC Dataset)

This script performs fine-tuning for XGBoost with SMOGN augmentation
on the BCC phase dataset.
"""

import pandas as pd
import numpy as np
import random
import os
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings
import joblib

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
def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    xgb_temp = xgb.XGBRegressor(
        n_estimators=100,
        random_state=SEED,
        verbosity=0,
        n_jobs=1
    )
    xgb_temp.fit(X_train, y_train)
    importances = xgb_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    print(f"Cumulative importance {target_ratio * 100}% requires {n_selected} features")
    print("Selected features:", list(selected_features))
    return selected_features


# ====================== Fine Grid Search ======================
def fine_grid_search_xgb(X_train, y_train, X_test, y_test, cv=5):
    param_grid = {
        'n_estimators': [273],
        'max_depth': [4],
        'learning_rate': [0.29],
        'subsample': [0.84],
        'colsample_bytree': [0.8],
        'min_child_weight': [7],
        'reg_alpha': [1.56],
        'reg_lambda': [0.921]
    }
    xgb_model = xgb.XGBRegressor(
        random_state=SEED,
        verbosity=0,
        n_jobs=1
    )
    grid = GridSearchCV(
        xgb_model, param_grid, cv=cv, scoring='r2',
        n_jobs=1, verbose=1
    )
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_

    cv_results = grid.cv_results_
    best_index = grid.best_index_
    cv_mean = cv_results['mean_test_score'][best_index]
    cv_std = cv_results['std_test_score'][best_index]

    y_train_pred = best_xgb.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    y_test_pred = best_xgb.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    best_params = grid.best_params_
    return best_xgb, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae, best_params, cv_mean, cv_std, y_test_pred


# ====================== Main Program ======================
if __name__ == "__main__":
    # Dataset path
    file_path = "BCC_data.csv"
    target_col = "Max_H2_Uptake_wt_pct"

    # SMOGN augmentation parameters
    SMOTER_RATIO = 3
    NOISE_RATIO = 0.05

    # 1. Load and preprocess
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"Original data shape: {df.shape}")
    df, target_col = preprocess_dataframe(df, target_col)
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
    # Step 3a: Auto-select features by cumulative importance (95%)
    selected_features = select_features_by_importance(X_train_orig, y_train_orig, target_ratio=0.95)

    # ================================================================
    # ===== 方案A修改：强制加入 Fe 和 Mn 作为特征 =====
    # ================================================================
    selected_features = list(selected_features)

    # 检查 Fe 和 Mn 列是否存在于训练数据中
    force_features = []
    for feat in ['Element_at_pct_Fe', 'Element_at_pct_Mn']:
        if feat in X_train_orig.columns:
            if feat not in selected_features:
                selected_features.append(feat)
                force_features.append(feat)
        else:
            print(f"Warning: '{feat}' not found in training data columns. Skipping.")

    if force_features:
        print(f"Force added features: {force_features}")
    else:
        print("No features force added (Fe and Mn either already present or not in data).")

    print(f"Final selected features ({len(selected_features)}): {selected_features}")
    # ================================================================

    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]
    print(f"After feature selection: {X_train_orig.shape}")

    # 4. SMOGN augmentation
    print(f"\n{'=' * 70}")
    print(f"SMOGN Augmentation: smoter_ratio = {SMOTER_RATIO} (total = {1 + SMOTER_RATIO}x), noise_ratio = {NOISE_RATIO}")
    print('=' * 70)
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=SMOTER_RATIO,
        noise_ratio=NOISE_RATIO,
        k=7,
        bins=10,
        extreme_factor=2.0,
        dist_threshold_factor=0.8
    )
    print(f"Augmented training size: {X_train_aug.shape[0]}")

    # 5. Fine grid search
    print("Executing fine grid search (5-fold CV)...")
    best_model, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae, best_params, cv_mean, cv_std, y_test_pred = fine_grid_search_xgb(
        X_train_aug, y_train_aug, X_test, y_test, cv=5
    )

    # 6. Overfitting
    overfit = train_r2 - test_r2

    # 7. Output all metrics
    print("\n" + "=" * 70)
    print("Final Model Performance Evaluation")
    print("=" * 70)
    print(f"Best parameters: {best_params}")
    print(f"\nCV R² (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"\nTraining Metrics:")
    print(f"  R²   = {train_r2:.4f}")
    print(f"  RMSE = {train_rmse:.4f}")
    print(f"  MAE  = {train_mae:.4f}")
    print(f"\nTest Metrics:")
    print(f"  R²   = {test_r2:.4f}")
    print(f"  RMSE = {test_rmse:.4f}")
    print(f"  MAE  = {test_mae:.4f}")
    print(f"\nOverfitting (Train R² - Test R²) = {overfit:.4f}")

    # 8. Save test set predictions
    test_results = pd.DataFrame({
        'true_capacity': y_test.values,
        'pred_capacity': y_test_pred
    })
    output_path = "test_predictions_BCC.csv"
    test_results.to_csv(output_path, index=False)
    print(f"\n✅ Test predictions saved to: {output_path}")

    # ========== 9. Save trained model and artifacts ==========
    print("\n" + "=" * 70)
    print("Saving trained model and artifacts...")
    print("=" * 70)

    # 9a. Save the best model
    model_save_path = "bcc_xgboost_smogn_best.pkl"
    joblib.dump(best_model, model_save_path)
    print(f"✅ Best model saved to: {model_save_path}")

    # 9b. Save selected feature names (now includes Fe and Mn if forced)
    feature_save_path = "bcc_selected_features.pkl"
    joblib.dump(selected_features, feature_save_path)
    print(f"✅ Selected features saved to: {feature_save_path}")

    # 9c. Save training statistics for candidate screening
    train_stats = {
        'X_train_stats': X_train_orig.describe(),
        'y_train_stats': y_train_orig.describe(),
        'selected_features': selected_features,
        'seed': SEED,
        'feature_names': X_train_orig.columns.tolist(),
    }
    stats_save_path = "bcc_train_stats.pkl"
    joblib.dump(train_stats, stats_save_path)
    print(f"✅ Training stats saved to: {stats_save_path}")

    print("\n✅ All artifacts saved successfully! Ready for candidate screening.")