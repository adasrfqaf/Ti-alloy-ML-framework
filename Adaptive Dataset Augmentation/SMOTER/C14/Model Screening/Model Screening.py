"""
Supplementary Code for C14 Laves Phase - SMOTER Augmentation Model Screening
This script evaluates multiple models with SMOTER augmentation for the C14 Laves phase dataset.
Outputs results sorted by 5-fold CV R² (mean±std) in a concise list format.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.base import clone
import warnings
import time

warnings.filterwarnings('ignore')
start_time = time.time()

# ==================== Load Data ====================
df = pd.read_csv('C14_data.csv', encoding='utf-8-sig')
print("=" * 80)
print("C14 Laves Phase Dataset - SMOTER Augmentation Model Screening")
print("=" * 80)
print(f"Original data shape: {df.shape}")

# Target column (English name)
target_col = 'Max_H2_Uptake_wt_pct'
feature_cols = [col for col in df.columns if col != target_col]

# Convert boolean columns to int, encode other non-numeric columns
for col in feature_cols:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
    elif df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols].fillna(0)
y = df[target_col].fillna(0)

print(f"Number of features: {X.shape[1]}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Valid samples: {len(X)}")


# ==================== SMOTER Extreme Oversampling ====================
def smoter_extreme_oversample(X_train, y_train, extreme_percentile, n_copies=5, k_neighbors=5, noise_std=0.05):
    """
    Oversample extreme low and high value samples using SMOTER-style interpolation.
    """
    X = np.array(X_train)
    y = np.array(y_train).flatten()

    low_thresh = np.percentile(y, extreme_percentile)
    high_thresh = np.percentile(y, 100 - extreme_percentile)

    low_idx = np.where(y <= low_thresh)[0]
    high_idx = np.where(y >= high_thresh)[0]

    print(f"    Low threshold: {low_thresh:.4f} (bottom {extreme_percentile}%), samples: {len(low_idx)}")
    print(f"    High threshold: {high_thresh:.4f} (top {extreme_percentile}%), samples: {len(high_idx)}")

    X_aug = []
    y_aug = []

    def augment_group(indices, label):
        if len(indices) == 0:
            return
        group_X = X[indices]
        if len(group_X) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(group_X)))
            nbrs.fit(group_X)
        for i, idx in enumerate(indices):
            for _ in range(n_copies):
                if len(group_X) > 1:
                    pos = np.where(indices == idx)[0][0]
                    distances, neigh_positions = nbrs.kneighbors([group_X[pos]])
                    candidates = [p for p in neigh_positions[0] if p != pos]
                    if len(candidates) == 0:
                        neigh_pos = pos
                    else:
                        neigh_pos = np.random.choice(candidates)
                    neighbor_idx = indices[neigh_pos]
                    gap = np.random.random()
                    new_X = X[idx] + gap * (X[neighbor_idx] - X[idx])
                else:
                    new_X = X[idx].copy()
                noise = np.random.normal(0, noise_std, len(new_X))
                new_X = new_X + noise
                if len(group_X) > 1:
                    new_y = y[idx] + gap * (y[neighbor_idx] - y[idx])
                else:
                    new_y = y[idx]
                new_y = new_y + np.random.normal(0, noise_std * y.std())
                X_aug.append(new_X)
                y_aug.append(new_y)

    augment_group(low_idx, 'low')
    augment_group(high_idx, 'high')

    if len(X_aug) > 0:
        X_aug = np.vstack([X, np.array(X_aug)])
        y_aug = np.concatenate([y, np.array(y_aug)])
    else:
        X_aug, y_aug = X, y

    return X_aug, y_aug


# ==================== Data Standardization ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

np.random.seed(42)
X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nOriginal training set: {len(X_train_base)}")
print(f"Test set: {len(X_test)}")
print(f"Training target range: [{y_train_base.min():.2f}, {y_train_base.max():.2f}]")

# ==================== SMOTER Configuration ====================
EXTREME_PERCENTILES = [5, 10, 20]
N_COPIES = 5
FEATURE_COUNTS = [15, 18, 20]

# ==================== Define Models (corrected) ====================
models = {
    'RF': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    'XGB': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=1),
    'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=1),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
                        early_stopping=True, verbose=False),
    'SVR': SVR(kernel='rbf', C=5, epsilon=0.1)
}

# Store all results
all_results = []

# 1. Baseline: Original data (no augmentation)
print("\n" + "=" * 80)
print("Baseline: Original Data (No Augmentation)")
print("=" * 80)
X_train_curr = X_train_base.values
y_train_curr = y_train_base.values
print(f"Training set size: {len(X_train_curr)}")

for n_feat in FEATURE_COUNTS:
    if n_feat > X_train_curr.shape[1]:
        continue
    selector = SelectKBest(mutual_info_regression, k=n_feat)
    X_train_sel = selector.fit_transform(X_train_curr, y_train_curr)
    X_test_sel = selector.transform(X_test.values)
    print(f"\nFeatures: {n_feat}")
    for name, model in models.items():
        try:
            model_clone = clone(model)
            model_clone.fit(X_train_sel, y_train_curr)
            y_pred_train = model_clone.predict(X_train_sel)
            y_pred_test = model_clone.predict(X_test_sel)
            train_r2 = r2_score(y_train_curr, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train_curr, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train_curr, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            overfit = train_r2 - test_r2

            # Compute CV R² (5-fold)
            cv_scores = cross_val_score(model_clone, X_train_sel, y_train_curr, cv=5, scoring='r2')
            cv_r2_mean = cv_scores.mean()
            cv_r2_std = cv_scores.std()

            all_results.append({
                'SMOTER_Config': 'Original (No Aug)',
                'Extreme_Percentile': '-',
                'Aug_Multiplier': 0,
                'Features': n_feat,
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Overfit': overfit,
                'CV_R2_mean': cv_r2_mean,
                'CV_R2_std': cv_r2_std,
                'Combined_Score': test_r2 - abs(overfit) * 0.3
            })
            print(f"  {name}: CV R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}, Test R²={test_r2:.4f}")
        except Exception as e:
            print(f"  {name}: Failed - {str(e)[:50]}")

# 2. SMOTER augmentation with different extreme percentiles
print("\n" + "=" * 80)
print("SMOTER Augmentation (Fixed 5x Multiplier)")
print("=" * 80)

for pct in EXTREME_PERCENTILES:
    print(f"\n{'=' * 60}")
    print(f"Extreme Percentile: {pct}% (both tails), Multiplier: {N_COPIES}x")
    print(f"{'=' * 60}")
    X_train_aug, y_train_aug = smoter_extreme_oversample(
        X_train_base.values, y_train_base.values,
        extreme_percentile=pct, n_copies=N_COPIES,
        k_neighbors=5, noise_std=0.05
    )
    print(f"Augmented training set size: {len(X_train_aug)} (Added: {len(X_train_aug) - len(X_train_base)})")

    for n_feat in FEATURE_COUNTS:
        if n_feat > X_train_aug.shape[1]:
            continue
        selector = SelectKBest(mutual_info_regression, k=n_feat)
        X_train_sel = selector.fit_transform(X_train_aug, y_train_aug)
        X_test_sel = selector.transform(X_test.values)
        print(f"\nFeatures: {n_feat}")
        for name, model in models.items():
            try:
                model_clone = clone(model)
                model_clone.fit(X_train_sel, y_train_aug)
                y_pred_train = model_clone.predict(X_train_sel)
                y_pred_test = model_clone.predict(X_test_sel)
                train_r2 = r2_score(y_train_aug, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train_aug, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                overfit = train_r2 - test_r2

                # Compute CV R² (5-fold)
                cv_scores = cross_val_score(model_clone, X_train_sel, y_train_aug, cv=5, scoring='r2')
                cv_r2_mean = cv_scores.mean()
                cv_r2_std = cv_scores.std()

                all_results.append({
                    'SMOTER_Config': f'{pct}% both tails',
                    'Extreme_Percentile': pct,
                    'Aug_Multiplier': N_COPIES,
                    'Features': n_feat,
                    'Model': name,
                    'Train_R2': train_r2,
                    'Test_R2': test_r2,
                    'Train_MAE': train_mae,
                    'Test_MAE': test_mae,
                    'Train_RMSE': train_rmse,
                    'Test_RMSE': test_rmse,
                    'Overfit': overfit,
                    'CV_R2_mean': cv_r2_mean,
                    'CV_R2_std': cv_r2_std,
                    'Combined_Score': test_r2 - abs(overfit) * 0.3
                })
                print(f"  {name}: CV R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}, Test R²={test_r2:.4f}")
            except Exception as e:
                print(f"  {name}: Failed - {str(e)[:50]}")

# ==================== Results Summary (Sorted by CV R²) ====================
results_df = pd.DataFrame(all_results)
results_df_sorted = results_df.sort_values('CV_R2_mean', ascending=False)

print("\n" + "=" * 80)
print("All Configurations Sorted by 5-fold CV R² (mean ± std)")
print("=" * 80)

# Print concise list
for idx, row in results_df_sorted.iterrows():
    print(f"CV R²: {row['CV_R2_mean']:.4f} ± {row['CV_R2_std']:.4f} | "
          f"Test R²: {row['Test_R2']:.4f} | "
          f"Overfit: {row['Overfit']:.4f} | "
          f"Model: {row['Model']:<6} | "
          f"SMOTER: {row['SMOTER_Config']:<18} | "
          f"Features: {int(row['Features'])}")

# Global best (by CV R²)
global_best = results_df_sorted.iloc[0]
print("\n" + "=" * 80)
print("Global Best Model (by CV R²)")
print("=" * 80)
print(f"SMOTER Config: {global_best['SMOTER_Config']}")
print(f"Extreme Percentile: {global_best['Extreme_Percentile']}%")
print(f"Aug Multiplier: {int(global_best['Aug_Multiplier'])}x")
print(f"Features: {int(global_best['Features'])}")
print(f"Model: {global_best['Model']}")
print(f"CV R²: {global_best['CV_R2_mean']:.4f} ± {global_best['CV_R2_std']:.4f}")
print(f"Test R²: {global_best['Test_R2']:.4f}")
print(f"Test MAE: {global_best['Test_MAE']:.4f}")
print(f"Test RMSE: {global_best['Test_RMSE']:.4f}")
print(f"Overfit: {global_best['Overfit']:.4f}")

# Optional: Best per model (also by CV R²)
print("\n" + "=" * 80)
print("Best Config per Model (by CV R²)")
print("=" * 80)
best_per_model = results_df.loc[results_df.groupby('Model')['CV_R2_mean'].idxmax()]
best_per_model = best_per_model.sort_values('CV_R2_mean', ascending=False)
for _, row in best_per_model.iterrows():
    print(f"{row['Model']:<6} | {row['SMOTER_Config']:<18} | Features={int(row['Features'])} | CV R²={row['CV_R2_mean']:.4f}±{row['CV_R2_std']:.4f} | Test R²={row['Test_R2']:.4f} | Overfit={row['Overfit']:.4f}")

print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
print("\nResults saved to: C14_SMOTER_results.csv (and others)")
print("\n" + "=" * 80)
print("Final Recommended Configuration")
print("=" * 80)
print(f"""
SMOTER Config:   {global_best['SMOTER_Config']}
Extreme Percentile: {global_best['Extreme_Percentile']}% (both tails)
Aug Multiplier:  {int(global_best['Aug_Multiplier'])}x
Features:        {int(global_best['Features'])}
Model:           {global_best['Model']}

Performance:
  CV R²    = {global_best['CV_R2_mean']:.4f} ± {global_best['CV_R2_std']:.4f}
  Test R²  = {global_best['Test_R2']:.4f}
  Test MAE = {global_best['Test_MAE']:.4f}
  Test RMSE= {global_best['Test_RMSE']:.4f}
  Overfit  = {global_best['Overfit']:.4f}
""")

# Save results to CSV (optional, keep for record)
results_df.to_csv('C14_SMOTER_results.csv', index=False)
best_per_model.to_csv('C14_SMOTER_best_per_model.csv', index=False)