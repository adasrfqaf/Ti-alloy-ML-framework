"""
Supplementary Code - Model Screening with SMOGN Augmentation

This script evaluates multiple models with SMOGN (SMOTE for Regression with Gaussian Noise)
augmentation for C14, C14 Alloy Screening, and TiFe phase datasets.
"""

import pandas as pd
import numpy as np
import random
import os
import re
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ====================== Global Configuration ======================
SEED = 49
N_JOBS = 1


# ====================== Set Random Seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(SEED)


# ====================== Column Name Cleaning ======================
def clean_column_names(df):
    """Clean column names: remove special characters, replace with underscores."""
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
def smogn_augmentation(X, y, smoter_ratio=0.3, noise_ratio=0.05, k=7,
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

    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean', n_jobs=N_JOBS)
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
            raise ValueError(f"Target column '{target_col}' or cleaned version '{cleaned_target}' not found")
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


def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=N_JOBS)
    rf_temp.fit(X_train, y_train)
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    print(f"Cumulative importance {target_ratio * 100}% requires {n_selected} features")
    print(f"Selected features: {list(selected_features)}")
    return selected_features


# ====================== Multi-Model Evaluation ======================
def evaluate_models_on_augmented_data(X_train, y_train, X_test, y_test,
                                      cv_folds=5, models=None):
    if models is None:
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=N_JOBS),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=SEED,
                                          verbosity=-1, force_col_wise=True, n_jobs=N_JOBS),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=SEED,
                                        verbosity=0, n_jobs=N_JOBS),
            'GBT': GradientBoostingRegressor(n_estimators=100, random_state=SEED),
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED,
                                early_stopping=True),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    for name, model in models.items():
        try:
            if name in ['GBT', 'MLP', 'SVR']:
                X_tr = X_train_scaled
                X_te = X_test_scaled
            else:
                X_tr = X_train
                X_te = X_test

            cv_r2_scores = cross_val_score(model, X_tr, y_train, cv=cv,
                                           scoring='r2', n_jobs=N_JOBS)
            cv_r2_mean = cv_r2_scores.mean()
            cv_r2_std = cv_r2_scores.std()

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)

            y_sorted = np.sort(y_test)
            high_th = y_sorted[int(0.9 * len(y_sorted))]
            low_th = y_sorted[int(0.1 * len(y_sorted))]
            high_mask = y_test >= high_th
            low_mask = y_test <= low_th
            high_mae = mean_absolute_error(y_test[high_mask], y_pred[high_mask]) if high_mask.sum() > 0 else np.nan
            low_mae = mean_absolute_error(y_test[low_mask], y_pred[low_mask]) if low_mask.sum() > 0 else np.nan

            results.append({
                'Model': name,
                'CV_R2_mean': cv_r2_mean,
                'CV_R2_std': cv_r2_std,
                'Test_R2': test_r2,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'High_MAE': high_mae,
                'Low_MAE': low_mae
            })

            print(f"  {name}: CV R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}, Test R² = {test_r2:.4f}")

        except Exception as e:
            print(f"Model {name} failed: {e}")
            continue

    return pd.DataFrame(results)


# ====================== Single Dataset Pipeline ======================
def process_one_dataset(file_path, dataset_name, target_col,
                        smoter_ratio=1.0, test_size=0.2,
                        noise_ratio=0.05, k=7, dist_threshold_factor=0.8,
                        extreme_factor=2.0, bins=10):
    print(f"\n{'=' * 80}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        df = pd.read_excel(file_path, engine='openpyxl')
    print(f"Original data shape: {df.shape}")

    df, target_col = preprocess_dataframe(df, target_col)
    print(f"After preprocessing: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Original training: {X_train.shape}, Test: {X_test.shape}")

    selected = select_features_by_importance(X_train, y_train, target_ratio=0.95)
    X_train = X_train[selected]
    X_test = X_test[selected]
    print(f"After feature selection: {X_train.shape}")

    print(f"Applying SMOGN: smoter_ratio={smoter_ratio}, noise_ratio={noise_ratio}, k={k}")
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train, y_train,
        smoter_ratio=smoter_ratio,
        noise_ratio=noise_ratio,
        k=k,
        bins=bins,
        extreme_factor=extreme_factor,
        dist_threshold_factor=dist_threshold_factor
    )
    print(f"Augmented training: {X_train_aug.shape} (original {len(X_train)} → {len(X_train_aug)})")

    print("\nMulti-model evaluation (5-fold CV + Test)...")
    print("-" * 60)
    results_df = evaluate_models_on_augmented_data(X_train_aug, y_train_aug, X_test, y_test)

    results_df_sorted = results_df.sort_values('CV_R2_mean', ascending=False).reset_index(drop=True)
    top4 = results_df_sorted.head(4)

    print(f"\n{dataset_name} Best Models (by CV R²):")
    print("=" * 80)
    print(top4[['Model', 'CV_R2_mean', 'CV_R2_std', 'Test_R2', 'Test_RMSE', 'Test_MAE']].to_string(index=False))

    results_df_sorted.to_csv(f"{dataset_name}_model_results_by_CV_R2.csv", index=False)
    top4.to_csv(f"{dataset_name}_top4_models_by_CV_R2.csv", index=False)

    return top4, results_df_sorted


# ====================== Main Program ======================
if __name__ == "__main__":
    datasets = {
        "C14": {
            "file": "C14_data.csv",
            "target": "Max_H2_Uptake_wt_pct"
        },
        "C14 Alloy Screening": {
            "file": "BCC_data.csv",
            "target": "Max_H2_Uptake_wt_pct"
        },
        "TiFe": {
            "file": "TiFe_data.csv",
            "target": "Max_H2_Uptake_wt_pct"
        }
    }

    SMOTER_RATIO = 1.0
    NOISE_RATIO = 0.05
    K_NEIGHBORS = 7
    DIST_THRESH_FACTOR = 0.8
    EXTREME_FACTOR = 2.0
    BINS = 10

    all_top_models = {}
    all_summary = []

    for name, cfg in datasets.items():
        try:
            top, full = process_one_dataset(
                file_path=cfg["file"],
                dataset_name=name,
                target_col=cfg["target"],
                smoter_ratio=SMOTER_RATIO,
                noise_ratio=NOISE_RATIO,
                k=K_NEIGHBORS,
                dist_threshold_factor=DIST_THRESH_FACTOR,
                extreme_factor=EXTREME_FACTOR,
                bins=BINS
            )
            all_top_models[name] = top

            best = top.iloc[0]
            all_summary.append({
                'Dataset': name,
                'Best_Model': best['Model'],
                'CV_R2': best['CV_R2_mean'],
                'CV_R2_std': best['CV_R2_std'],
                'Test_R2': best['Test_R2'],
                'Test_RMSE': best['Test_RMSE'],
                'Test_MAE': best['Test_MAE']
            })

        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 100)
    print("Summary: Best Model per Dataset (by CV R²)")
    print("=" * 100)

    summary_df = pd.DataFrame(all_summary)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    summary_df.to_csv("all_datasets_best_models_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("Top 4 Models per Dataset:")
    print("=" * 100)
    for name, top in all_top_models.items():
        print(f"\n[{name}] Top 4 Models (by CV R²):")
        print(top[['Model', 'CV_R2_mean', 'CV_R2_std', 'Test_R2', 'Test_RMSE', 'Test_MAE']].to_string(index=False))

    print("\n" + "=" * 100)
    print("All results saved:")
    print("  - {dataset_name}_model_results_by_CV_R2.csv (all models)")
    print("  - {dataset_name}_top4_models_by_CV_R2.csv (Top 4 models)")
    print("  - all_datasets_best_models_summary.csv (summary)")
    print("=" * 100)