"""
GBDT Fine-Tuning for C14 Alloy Screening Dataset (with Gaussian Noise Augmentation)
No files are saved; results are printed to console only.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')


# ====================== Fixed Random Seeds ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = r'BCC_data.csv'   # Update if needed
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 20

# GBDT hyperparameter search space (single values – no actual search)
PARAM_DIST = {
    'n_estimators': [190],
    'learning_rate': [0.072],
    'max_depth': [5],
    'min_samples_split': [20],
    'min_samples_leaf': [6],
    'max_features': [0.3],
    'subsample': [0.85],
    'loss': ['huber'],
    'validation_fraction': [0.2],
    'n_iter_no_change': [15]
}
N_ITER = 100

# Noise and expansion parameters
SIGMA_RATIOS = [0.01, 0.02]
EXPANSION_FACTORS = [5]


# ====================== Data Loading and Preprocessing ======================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    print(f"Number of numeric features retained: {X.shape[1]}")
    return X, y


def select_top_features_rf(X_train, y_train, top_n=15):
    """Feature selection using Random Forest importance"""
    temp_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_model.fit(X_scaled, y_train)
    importance = temp_model.feature_importances_
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    print(f"\nTop {top_n} features selected (based on RF importance):")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (importance: {importance[sorted_idx[i - 1]]:.4f})")
    return top_features


def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=1):
    """Add relative noise: noise std = sigma_ratio * feature std"""
    if n_copies == 0 or sigma_ratio == 0.0:
        return X, y

    X_noisy_list = [X]
    y_noisy_list = [y]
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            noise = np.random.normal(0, sigma_ratio * stds[col], len(X))
            X_copy[col] = X_copy[col] + noise
        X_noisy_list.append(X_copy)
        y_noisy_list.append(y)
    X_aug = pd.concat(X_noisy_list, ignore_index=True)
    y_aug = np.concatenate(y_noisy_list)
    return X_aug, y_aug


def evaluate_model(model, X, y):
    """Evaluate model and return R², MAE, RMSE"""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse


def calculate_overfitting(cv_r2_mean, train_r2, test_r2):
    """Calculate overfitting metrics"""
    train_test_gap = train_r2 - test_r2
    cv_test_gap = cv_r2_mean - test_r2
    overfitting_score = train_test_gap + max(0, cv_test_gap)
    return {
        'train_test_gap': train_test_gap,
        'cv_test_gap': cv_test_gap,
        'overfitting_score': overfitting_score
    }


# ====================== Main Pipeline ======================
print("=" * 100)
print("GBDT Fine-Tuning with Full Train/Test Metrics including CV R² (mean ± std)")
print("=" * 100)

print("\nLoading C14 Alloy Screening dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")

# Fixed split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection (based on original training set)
top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"Training set after feature selection: {X_train_sel.shape}")

# Compute feature standard deviations (for relative noise)
feature_stds = X_train_sel.std(axis=0)

# Store all results
all_results = []

total_combinations = len(SIGMA_RATIOS) * len(EXPANSION_FACTORS)
current_combo = 0

for sigma_ratio in SIGMA_RATIOS:
    for n_copies in EXPANSION_FACTORS:
        current_combo += 1
        print(f"\n{'=' * 80}")
        print(f"Experiment [{current_combo}/{total_combinations}]")
        print(f"Noise level sigma_ratio = {sigma_ratio}, Expansion factor = {n_copies}x")
        print('=' * 80)

        # Data augmentation
        if sigma_ratio == 0.0 or n_copies == 0:
            X_train_aug = X_train_sel
            y_train_aug = y_train
            print(f"Using original training set (no noise), samples: {len(X_train_aug)}")
        else:
            X_train_aug, y_train_aug = add_relative_gaussian_noise(
                X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=n_copies
            )
            print(f"Augmented training set size: {len(X_train_aug)} (original {len(X_train_sel)} + {n_copies} noise copies)")

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_sel)

        # GBDT model
        gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE)
        random_search = RandomizedSearchCV(
            gbdt, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
            scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=0
        )
        random_search.fit(X_train_scaled, y_train_aug)

        best_params = random_search.best_params_
        best_cv_r2_mean = random_search.best_score_
        best_model = random_search.best_estimator_

        # Recompute CV scores for exact mean and std
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train_aug,
                                    cv=CV_FOLDS, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()

        # Evaluate on train and test
        r2_train, mae_train, rmse_train = evaluate_model(best_model, X_train_scaled, y_train_aug)
        r2_test, mae_test, rmse_test = evaluate_model(best_model, X_test_scaled, y_test)
        overfitting_metrics = calculate_overfitting(cv_r2_mean, r2_train, r2_test)

        print(f"\nBest parameters: {best_params}")
        print(f"Cross-validation R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        print(f"Train R²: {r2_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
        print(f"Test  R²: {r2_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")
        print(f"Train-Test gap (R²): {overfitting_metrics['train_test_gap']:.4f}")
        print(f"Overfitting score: {overfitting_metrics['overfitting_score']:.4f}")

        all_results.append({
            'sigma_ratio': sigma_ratio,
            'n_copies': n_copies,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'train_r2': r2_train,
            'test_r2': r2_test,
            'train_mae': mae_train,
            'test_mae': mae_test,
            'train_rmse': rmse_train,
            'test_rmse': rmse_test,
            'train_test_gap': overfitting_metrics['train_test_gap'],
            'cv_test_gap': overfitting_metrics['cv_test_gap'],
            'overfitting_score': overfitting_metrics['overfitting_score'],
            'best_params': str(best_params)
        })

# ====================== Summary Output ======================
print("\n\n" + "=" * 120)
print("GBDT Fine-Tuning Results (full train/test metrics with CV R² mean ± std)")
print("=" * 120)

df_all_results = pd.DataFrame(all_results)
df_all_results = df_all_results.sort_values(['sigma_ratio', 'n_copies'])

print("\nAll Experiment Results:")
display_cols = ['sigma_ratio', 'n_copies',
                'cv_r2_mean', 'cv_r2_std',
                'train_r2', 'test_r2',
                'train_mae', 'test_mae',
                'train_rmse', 'test_rmse',
                'train_test_gap', 'overfitting_score']
print(df_all_results[display_cols].to_string(index=False, float_format="%.4f"))

# Global best by test R²
global_best_idx = df_all_results['test_r2'].idxmax()
global_best = df_all_results.loc[global_best_idx]

print("\n" + "=" * 120)
print("🏆 Global Best Model (by test R²)")
print("=" * 120)
print(f"Noise level: σ = {global_best['sigma_ratio']}")
print(f"Expansion factor: {int(global_best['n_copies'])}x")
print(f"\nCross-validation:")
print(f"  R² = {global_best['cv_r2_mean']:.4f} ± {global_best['cv_r2_std']:.4f}")
print(f"\nTraining Metrics:")
print(f"  R²  = {global_best['train_r2']:.4f}")
print(f"  MAE = {global_best['train_mae']:.4f}")
print(f"  RMSE= {global_best['train_rmse']:.4f}")
print(f"\nTest Metrics:")
print(f"  R²  = {global_best['test_r2']:.4f}")
print(f"  MAE = {global_best['test_mae']:.4f}")
print(f"  RMSE= {global_best['test_rmse']:.4f}")
print(f"\nOthers:")
print(f"  Train-Test gap (R²) = {global_best['train_test_gap']:.4f}")
print(f"  Overfitting score = {global_best['overfitting_score']:.4f}")

# Best by overfitting score (generalization)
best_gen_idx = df_all_results['overfitting_score'].idxmin()
best_gen = df_all_results.loc[best_gen_idx]

print("\n" + "=" * 120)
print("🌟 Best Generalization Model (min overfitting score)")
print("=" * 120)
print(f"Noise level: σ = {best_gen['sigma_ratio']}")
print(f"Expansion factor: {int(best_gen['n_copies'])}x")
print(f"Overfitting score = {best_gen['overfitting_score']:.4f}")
print(f"\nCross-validation:")
print(f"  R² = {best_gen['cv_r2_mean']:.4f} ± {best_gen['cv_r2_std']:.4f}")
print(f"\nTraining Metrics:")
print(f"  R²  = {best_gen['train_r2']:.4f}")
print(f"  MAE = {best_gen['train_mae']:.4f}")
print(f"  RMSE= {best_gen['train_rmse']:.4f}")
print(f"\nTest Metrics:")
print(f"  R²  = {best_gen['test_r2']:.4f}")
print(f"  MAE = {best_gen['test_mae']:.4f}")
print(f"  RMSE= {best_gen['test_rmse']:.4f}")

# Group by noise level
print("\n" + "=" * 120)
print("Best Expansion Factor per Noise Level")
print("=" * 120)

best_per_sigma = []
for sigma in SIGMA_RATIOS:
    subset = df_all_results[df_all_results['sigma_ratio'] == sigma]
    if len(subset) > 0:
        best_idx = subset['test_r2'].idxmax()
        best_row = subset.loc[best_idx]
        best_per_sigma.append(best_row)
        print(f"σ={sigma:.3f}: best expansion factor = {int(best_row['n_copies'])}x")
        print(f"  CV R² = {best_row['cv_r2_mean']:.4f} ± {best_row['cv_r2_std']:.4f}")
        print(f"  Train R²={best_row['train_r2']:.4f}, Test R²={best_row['test_r2']:.4f}")
        print(f"  Test MAE={best_row['test_mae']:.4f}, Test RMSE={best_row['test_rmse']:.4f}")
        print(f"  Overfitting score={best_row['overfitting_score']:.4f}")

# Group by expansion factor
print("\n" + "=" * 120)
print("Best Noise Level per Expansion Factor")
print("=" * 120)

best_per_copies = []
for n_copies in EXPANSION_FACTORS:
    subset = df_all_results[df_all_results['n_copies'] == n_copies]
    if len(subset) > 0:
        best_idx = subset['test_r2'].idxmax()
        best_row = subset.loc[best_idx]
        best_per_copies.append(best_row)
        print(f"Expansion={n_copies}x: best noise σ={best_row['sigma_ratio']:.3f}")
        print(f"  CV R² = {best_row['cv_r2_mean']:.4f} ± {best_row['cv_r2_std']:.4f}")
        print(f"  Train R²={best_row['train_r2']:.4f}, Test R²={best_row['test_r2']:.4f}")
        print(f"  Test MAE={best_row['test_mae']:.4f}, Test RMSE={best_row['test_rmse']:.4f}")
        print(f"  Overfitting score={best_row['overfitting_score']:.4f}")

# Print full hyperparameters of global best
print("\n" + "=" * 120)
print("Complete Hyperparameters of Global Best Model")
print("=" * 120)
print(f"Best configuration:")
print(f"  - Noise level (sigma_ratio): {global_best['sigma_ratio']}")
print(f"  - Expansion factor (n_copies): {int(global_best['n_copies'])}x")
print(f"\nBest hyperparameters:")
best_params_dict = eval(global_best['best_params'])
for key, value in best_params_dict.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 120)
print("GBDT fine-tuning complete! No files were saved.")
print("=" * 120)