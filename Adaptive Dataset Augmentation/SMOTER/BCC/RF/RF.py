"""
Supplementary Code for C14 Alloy Screening Phase - Random Forest Fine-Tuning with SMOTER Augmentation

This script performs comprehensive hyperparameter search for Random Forest
under different SMOTER augmentation ratios and feature counts on the C14 Alloy Screening dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
SEED = 42
np.random.seed(SEED)

# ====================== User Parameters ======================
# Data augmentation modes (add/remove as needed)
DATA_MODES = [
    (0.20, 0.20, 5, "SMOTER both tails 20%")
]
FEATURE_COUNTS = [19]

# Expanded hyperparameter grid for Random Forest
PARAM_GRID = {
    'n_estimators': [100],
    'max_depth': [9],
    'min_samples_split': [3],
    'min_samples_leaf': [1],
    'max_features': [0.9],
    'ccp_alpha': [0.0002],
    'max_samples': [0.9]
}
CV_FOLDS = 3


# ====================== Load C14 Alloy Screening Data ======================
def load_bcc_data():
    df = pd.read_csv('BCC_data.csv', encoding='utf-8-sig')
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    # Separate features and target
    X = df.select_dtypes(include=[np.number]).drop(columns=['Max_H2_Uptake_wt_pct'], errors='ignore')
    y = df['Max_H2_Uptake_wt_pct'].values
    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]
    # Fill missing values with mean
    X = X.fillna(X.mean())
    return X, y


# ====================== SMOTER Bilateral Interpolation ======================
def bilateral_smoter_interpolate(X_train, y_train, low_ratio, high_ratio, k=5):
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    low_th = np.percentile(y, 100 * low_ratio)
    high_th = np.percentile(y, 100 * (1 - high_ratio))
    minority_idx = np.where((y <= low_th) | (y >= high_th))[0]

    print(f"  Low threshold: ≤ {low_th:.3f} (bottom {low_ratio*100:.0f}%)")
    print(f"  High threshold: ≥ {high_th:.3f} (top {high_ratio*100:.0f}%)")
    print(f"  Selected samples: {len(minority_idx)}")

    if len(minority_idx) < 2:
        print("  Warning: Insufficient samples, returning original data")
        return X_train, y_train

    nbrs = NearestNeighbors(n_neighbors=min(k, len(minority_idx)-1), metric='euclidean').fit(X[minority_idx])
    synthetic_X, synthetic_y = [], []
    for idx in minority_idx:
        distances, indices = nbrs.kneighbors(X[idx].reshape(1, -1))
        neighbor_local_idx = np.random.choice(indices[0][1:], 1)[0]
        neighbor_global_idx = minority_idx[neighbor_local_idx]
        gap = np.random.uniform(0, 1)
        synthetic_X.append(X[idx] + gap * (X[neighbor_global_idx] - X[idx]))
        synthetic_y.append(y[idx] + gap * (y[neighbor_global_idx] - y[idx]))
    X_aug = np.vstack([X, synthetic_X])
    y_aug = np.concatenate([y, synthetic_y])
    if isinstance(X_train, pd.DataFrame):
        X_aug = pd.DataFrame(X_aug, columns=X_train.columns)
    return X_aug, y_aug


# ====================== Evaluation Function ======================
def evaluate_config(X_train, X_test, y_train, y_test, feature_names, top_k, param_grid, cv):
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature importance
    temp_rf = RandomForestRegressor(random_state=SEED, n_jobs=1)
    temp_rf.fit(X_train_scaled, y_train)
    importances = temp_rf.feature_importances_

    if top_k is not None and top_k < len(feature_names):
        top_indices = np.argsort(importances)[::-1][:top_k]
        important_mask = np.zeros(len(feature_names), dtype=bool)
        important_mask[top_indices] = True
        selected_features = [feature_names[i] for i in top_indices]
    else:
        important_mask = np.ones(len(feature_names), dtype=bool)
        selected_features = feature_names

    X_train_filt = X_train_scaled[:, important_mask]
    X_test_filt = X_test_scaled[:, important_mask]

    # Grid search
    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=1)
    gs = GridSearchCV(rf_base, param_grid, cv=cv, scoring='r2', n_jobs=1, verbose=0)
    gs.fit(X_train_filt, y_train)

    best_model = gs.best_estimator_
    best_params = gs.best_params_

    # CV on best model
    cv_scores = cross_val_score(best_model, X_train_filt, y_train, cv=cv, scoring='r2')
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    y_train_pred = best_model.predict(X_train_filt)
    y_test_pred = best_model.predict(X_test_filt)

    return {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'best_params': best_params,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'selected_features': selected_features,
        'n_features': len(selected_features)
    }


# ====================== Main Program ======================
X, y = load_bcc_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
feature_names = X.columns.tolist()
print(f"Original training: {len(X_train)}, Test: {len(X_test)}")
print(f"Total features: {len(feature_names)}\n")

all_results = []

for mode in DATA_MODES:
    if mode[0] == 'original':
        data_name = mode[3]
        print(f"\nData Mode: {data_name}")
        X_curr, y_curr = X_train, y_train
        aug_ratio = 1.0
        print(f"Training samples: {len(X_curr)} (no augmentation)")
    else:
        low, high, k, data_name = mode
        print(f"\nData Mode: {data_name}")
        X_curr, y_curr = bilateral_smoter_interpolate(X_train, y_train, low, high, k)
        aug_ratio = len(X_curr) / len(X_train)
        print(f"Augmented training: {len(X_curr)} ({aug_ratio:.2f}x)")

    for n_feat in FEATURE_COUNTS:
        metrics = evaluate_config(
            X_curr, X_test, y_curr, y_test,
            feature_names, top_k=n_feat,
            param_grid=PARAM_GRID, cv=CV_FOLDS
        )
        all_results.append({
            'Data_Mode': data_name,
            'n_features': n_feat,
            'aug_ratio': f"{aug_ratio:.2f}",
            'best_params': str(metrics['best_params']),
            'cv_mean': metrics['cv_mean'],
            'cv_std': metrics['cv_std'],
            'train_r2': metrics['train_r2'],
            'test_r2': metrics['test_r2'],
            'train_mae': metrics['train_mae'],
            'test_mae': metrics['test_mae'],
            'train_rmse': metrics['train_rmse'],
            'test_rmse': metrics['test_rmse'],
            'selected_features': metrics['selected_features']
        })

# ====================== Global Summary ======================
print("\n" + "=" * 80)
print("All Results Summary (sorted by Test R²):")
df_all = pd.DataFrame(all_results)
df_sorted = df_all.sort_values('test_r2', ascending=False)
df_display = df_sorted.copy()
df_display['CV R² (mean±std)'] = df_display.apply(
    lambda row: f"{row['cv_mean']:.4f} ± {row['cv_std']:.4f}", axis=1
)
print(df_display[['Data_Mode', 'n_features', 'CV R² (mean±std)',
                  'test_r2', 'test_mae', 'test_rmse']].to_string(index=False))

best = df_sorted.iloc[0]
print(f"\n{'='*80}")
print("Global Best Model:")
print(f"  Data Mode: {best['Data_Mode']}")
print(f"  Features: {best['n_features']}")
print(f"  Augmentation ratio: {best['aug_ratio']}x")
print(f"  Selected features: {best['selected_features']}")
print(f"\n  Best Hyperparameters:")
for k, v in eval(best['best_params']).items():
    print(f"    {k}: {v}")
print(f"\n  Train R² : {best['train_r2']:.4f}")
print(f"  Test R²  : {best['test_r2']:.4f}")
print(f"  Train MAE: {best['train_mae']:.4f}")
print(f"  Test MAE : {best['test_mae']:.4f}")
print(f"  Train RMSE: {best['train_rmse']:.4f}")
print(f"  Test RMSE : {best['test_rmse']:.4f}")
print(f"  Overfit gap: {best['train_r2'] - best['test_r2']:.4f}")
print(f"  CV R²: {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print("="*80)
