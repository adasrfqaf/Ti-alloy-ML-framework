"""
Supplementary Code for C14 Alloy Screening Phase - Gradient Boosting Model Fine-Tuning with SMOTER Augmentation

This script evaluates Gradient Boosting performance with SMOTER augmentation for the C14 Alloy Screening phase dataset.
Only SMOTER augmentation modes (5%, 10%, 20%) are used; original data is not included.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
SEED = 49
np.random.seed(SEED)

# ====================== User Parameters ======================
DATA_MODES = [
    (0.05, 0.05, 5, "SMOTER both tails 5%"),
    (0.10, 0.10, 5, "SMOTER both tails 10%"),
    (0.20, 0.20, 5, "SMOTER both tails 20%")
]
FEATURE_COUNTS = [15]

PARAM_GRID = {
    'n_estimators': [180],
    'max_depth': [5],
    'learning_rate': [0.052],
    'min_samples_split': [5],
    'min_samples_leaf': [4],
    'subsample': [0.75]
}
CV_FOLDS = 3


# ====================== Load C14 Alloy Screening Data ======================
def load_bcc_data():
    df = pd.read_csv('BCC_data.csv', encoding='utf-8-sig')
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    X = df.select_dtypes(include=[np.number]).drop(columns=['Max_H2_Uptake_wt_pct'], errors='ignore')
    y = df['Max_H2_Uptake_wt_pct'].values
    X = X.loc[:, X.nunique() > 1]
    X = X.fillna(X.mean())
    return X, y


# ====================== SMOTER Interpolation ======================
def bilateral_smoter_interpolate(X_train, y_train, low_ratio, high_ratio, k=5):
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    low_th = np.percentile(y, 100 * low_ratio)
    high_th = np.percentile(y, 100 * (1 - high_ratio))
    minority_idx = np.where((y <= low_th) | (y >= high_th))[0]

    print(f"  Low threshold: ≤ {low_th:.3f} (bottom {low_ratio * 100:.0f}%)")
    print(f"  High threshold: ≥ {high_th:.3f} (top {high_ratio * 100:.0f}%)")
    print(f"  Selected samples for augmentation: {len(minority_idx)}")

    if len(minority_idx) < 2:
        print("  Warning: Insufficient samples for interpolation, returning original data")
        return X_train, y_train

    nbrs = NearestNeighbors(n_neighbors=min(k, len(minority_idx) - 1), metric='euclidean').fit(X[minority_idx])
    synthetic_X = []
    synthetic_y = []
    for idx in minority_idx:
        distances, indices = nbrs.kneighbors(X[idx].reshape(1, -1))
        neighbor_local_idx = np.random.choice(indices[0][1:], 1)[0]
        neighbor_global_idx = minority_idx[neighbor_local_idx]
        gap = np.random.uniform(0, 1)
        synthetic_sample = X[idx] + gap * (X[neighbor_global_idx] - X[idx])
        synthetic_target = y[idx] + gap * (y[neighbor_global_idx] - y[idx])
        synthetic_X.append(synthetic_sample)
        synthetic_y.append(synthetic_target)

    X_aug = np.vstack([X, synthetic_X])
    y_aug = np.concatenate([y, synthetic_y])
    if isinstance(X_train, pd.DataFrame):
        X_aug = pd.DataFrame(X_aug, columns=X_train.columns)
    return X_aug, y_aug


# ====================== Feature Selection + Grid Search ======================
def evaluate_feature_and_gridsearch(X_train, X_test, y_train, y_test,
                                    feature_names, top_k, param_grid, cv):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    temp_gb = GradientBoostingRegressor(random_state=SEED)
    temp_gb.fit(X_train_scaled, y_train)
    importances = temp_gb.feature_importances_

    if top_k is not None and top_k < len(feature_names):
        top_indices = np.argsort(importances)[::-1][:top_k]
        important_mask = np.zeros(len(feature_names), dtype=bool)
        important_mask[top_indices] = True
    else:
        important_mask = np.ones(len(feature_names), dtype=bool)
        top_indices = np.arange(len(feature_names))

    X_train_filt = X_train_scaled[:, important_mask]
    X_test_filt = X_test_scaled[:, important_mask]

    gb_base = GradientBoostingRegressor(random_state=SEED)
    gs = GridSearchCV(gb_base, param_grid, cv=cv, scoring='r2', n_jobs=1, verbose=0)
    gs.fit(X_train_filt, y_train)

    best_model = gs.best_estimator_
    best_params = gs.best_params_

    cv_r2_scores = cross_val_score(best_model, X_train_filt, y_train, cv=cv, scoring='r2')
    cv_r2_mean = cv_r2_scores.mean()
    cv_r2_std = cv_r2_scores.std()

    y_train_pred = best_model.predict(X_train_filt)
    y_test_pred = best_model.predict(X_test_filt)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'best_params': best_params,
        'n_features': len(top_indices),
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'selected_features': [feature_names[i] for i in top_indices]
    }


# ====================== Main ======================
print("=" * 80)
print("C14 Alloy Screening Dataset - Gradient Boosting Model Fine-Tuning")
print("Comparing SMOTER augmentation modes (5%, 10%, 20%)")
print("Feature count: 20 | Expanded grid search")
print("Final model CV evaluation (3-fold)")
print("=" * 80)

X, y = load_bcc_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
feature_names = X.columns.tolist()
print(f"Original training set: {len(X_train)}, Test set: {len(X_test)}")
print(f"Original feature count: {len(feature_names)}\n")

all_results = []

for mode in DATA_MODES:
    low_ratio, high_ratio, k, data_name = mode
    print(f"\nData Mode: {data_name}")
    X_curr, y_curr = bilateral_smoter_interpolate(X_train, y_train, low_ratio, high_ratio, k=k)
    aug_ratio = len(X_curr) / len(X_train)
    print(f"Augmented training set: {len(X_curr)} ({aug_ratio:.2f}x)")

    for n_feat in FEATURE_COUNTS:
        metrics = evaluate_feature_and_gridsearch(
            X_curr, X_test, y_curr, y_test, feature_names,
            top_k=n_feat, param_grid=PARAM_GRID, cv=CV_FOLDS
        )
        all_results.append({
            'Data_Mode': data_name,
            'n_features': n_feat,
            'aug_ratio': f"{aug_ratio:.2f}",
            'best_params': str(metrics['best_params']),
            'train_R2': metrics['train_r2'],
            'test_R2': metrics['test_r2'],
            'train_MAE': metrics['train_mae'],
            'test_MAE': metrics['test_mae'],
            'train_RMSE': metrics['train_rmse'],
            'test_RMSE': metrics['test_rmse'],
            'cv_R2_mean': metrics['cv_r2_mean'],
            'cv_R2_std': metrics['cv_r2_std'],
            'selected_features': metrics['selected_features']
        })

# Summary
print("\n" + "=" * 80)
print("All Combinations Summary (sorted by Test R²):")
df_all = pd.DataFrame(all_results)
df_sorted = df_all.sort_values('test_R2', ascending=False)
df_display = df_sorted.copy()
df_display['CV R² (mean±std)'] = df_display.apply(
    lambda row: f"{row['cv_R2_mean']:.4f} ± {row['cv_R2_std']:.4f}", axis=1
)
print(df_display[['Data_Mode', 'n_features', 'CV R² (mean±std)', 'test_R2', 'train_R2']].to_string(index=False))

best_overall = df_sorted.iloc[0]
print(f"\n{'='*80}")
print("Global Best Model:")
print(f"  Data Mode: {best_overall['Data_Mode']}")
print(f"  Features: {best_overall['n_features']}")
print(f"  Augmentation ratio: {best_overall['aug_ratio']}x")
print(f"  Selected features: {best_overall['selected_features']}")
print(f"\n  Best Hyperparameters: {best_overall['best_params']}")
print(f"\n  Train R²: {best_overall['train_R2']:.4f}")
print(f"  Test R² : {best_overall['test_R2']:.4f}")
print(f"  Train MAE: {best_overall['train_MAE']:.4f}")
print(f"  Test MAE : {best_overall['test_MAE']:.4f}")
print(f"  Train RMSE: {best_overall['train_RMSE']:.4f}")
print(f"  Test RMSE : {best_overall['test_RMSE']:.4f}")
print(f"  Overfit gap (Train-Test R²): {best_overall['train_R2'] - best_overall['test_R2']:.4f}")
print(f"  CV R²: {best_overall['cv_R2_mean']:.4f} ± {best_overall['cv_R2_std']:.4f}")
print("="*80)
