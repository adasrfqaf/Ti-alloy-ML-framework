"""
Supplementary Code - SVR Fine-Tuning with Gaussian Noise Augmentation (TiFe Dataset)

This script performs fine-tuning for SVR with Gaussian noise augmentation
on the TiFe phase dataset. No files are saved; all results are printed to console.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import warnings

warnings.filterwarnings('ignore')


# ====================== Set Global Random Seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = 'TiFe_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 18

# SVR hyperparameter search space (using RBF kernel)
PARAM_DIST = {
    'C': [3],
    'epsilon': [0.05],
    'gamma': ['scale'],
    'kernel': ['rbf']
}
N_ITER = 50

# Noise levels and expansion factors (same as GBDT)
SIGMA_RATIOS = [0.1]
EXPANSION_FACTORS = [ 4]


# ====================== Data Loading & Preprocessing ======================
def load_data(path):
    """Load TiFe dataset with English column names."""
    df = pd.read_csv(path, encoding='utf-8-sig')

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    # Handle missing values
    X = X.fillna(0)
    y = np.nan_to_num(y)

    print(f"Number of numeric features retained: {X.shape[1]}")
    return X, y


def select_top_features_rf(X_train, y_train, top_n=18):
    """Select top features using Random Forest importance."""
    from sklearn.ensemble import RandomForestRegressor
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
    print(f"\nSelected top {top_n} features (by Random Forest importance):")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (importance: {importance[sorted_idx[i - 1]]:.4f})")
    return top_features


def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=2):
    """Add relative Gaussian noise: noise_std = sigma_ratio * feature_std."""
    if n_copies == 0 or sigma_ratio == 0.0:
        return X, y

    X_noisy_list = [X]
    y_noisy_list = [y]
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            col_std = stds[col]
            if col_std > 0:
                noise = np.random.normal(0, sigma_ratio * col_std, len(X))
                X_copy[col] = X_copy[col] + noise
        X_noisy_list.append(X_copy)
        y_noisy_list.append(y)
    X_aug = pd.concat(X_noisy_list, ignore_index=True)
    y_aug = np.concatenate(y_noisy_list)
    return X_aug, y_aug


def evaluate_model(model, X, y):
    """Evaluate model and return R², MAE, RMSE."""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse


def calculate_overfitting(cv_r2_mean, train_r2, test_r2):
    """Calculate overfitting metrics using CV mean."""
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
print("SVR Fine-Tuning - TiFe Dataset (Gaussian Noise Augmentation)")
print("=" * 100)

print("\nLoading TiFe dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Fixed train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection (based on original training set)
top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"After feature selection - Training set shape: {X_train_sel.shape}")

# Calculate feature standard deviations for relative noise
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
            print(f"Augmented training set: {len(X_train_aug)} (original {len(X_train_sel)} + {n_copies} noise copies)")

        # Standardization (SVR is scale-sensitive)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_sel)

        # SVR model
        svr = SVR()
        random_search = RandomizedSearchCV(
            svr, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
            scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=0
        )
        random_search.fit(X_train_scaled, y_train_aug)

        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        # Recalculate CV R² mean and std for the best model
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train_aug,
                                    cv=CV_FOLDS, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()

        # Evaluate on training and test sets
        r2_train, mae_train, rmse_train = evaluate_model(best_model, X_train_scaled, y_train_aug)
        r2_test, mae_test, rmse_test = evaluate_model(best_model, X_test_scaled, y_test)
        overfitting_metrics = calculate_overfitting(cv_r2_mean, r2_train, r2_test)

        print(f"\nBest params: {best_params}")
        print(f"CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        print(f"Train R²: {r2_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
        print(f"Test  R²: {r2_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")
        print(f"Train-Test Gap: {overfitting_metrics['train_test_gap']:.4f}")
        print(f"Overfitting Score: {overfitting_metrics['overfitting_score']:.4f}")

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

# ====================== Results Summary ======================
print("\n\n" + "=" * 120)
print("SVR Performance Comparison (Different Noise Levels × Expansion Factors)")
print("=" * 120)

df_all_results = pd.DataFrame(all_results)
df_all_results = df_all_results.sort_values(['sigma_ratio', 'n_copies'])

print("\n[All Experimental Results]")
display_cols = ['sigma_ratio', 'n_copies',
                'cv_r2_mean', 'cv_r2_std',
                'train_r2', 'test_r2',
                'train_mae', 'test_mae',
                'train_rmse', 'test_rmse',
                'train_test_gap', 'overfitting_score']
print(df_all_results[display_cols].to_string(index=False, float_format="%.4f"))

# Global best (by Test R²)
global_best_idx = df_all_results['test_r2'].idxmax()
global_best = df_all_results.loc[global_best_idx]

print("\n" + "=" * 120)
print("Global Best Model (by Test R²)")
print("=" * 120)
print(f"Noise level: σ = {global_best['sigma_ratio']}")
print(f"Expansion factor: {int(global_best['n_copies'])}x")
print(f"\n[Cross-Validation]")
print(f"  R² = {global_best['cv_r2_mean']:.4f} ± {global_best['cv_r2_std']:.4f}")
print(f"\n[Training Metrics]")
print(f"  R²  = {global_best['train_r2']:.4f}")
print(f"  MAE = {global_best['train_mae']:.4f}")
print(f"  RMSE= {global_best['train_rmse']:.4f}")
print(f"\n[Test Metrics]")
print(f"  R²  = {global_best['test_r2']:.4f}")
print(f"  MAE = {global_best['test_mae']:.4f}")
print(f"  RMSE= {global_best['test_rmse']:.4f}")
print(f"\n[Other]")
print(f"  Train-Test Gap = {global_best['train_test_gap']:.4f}")
print(f"  Overfitting Score = {global_best['overfitting_score']:.4f}")

# Best generalization (minimum overfitting score)
best_gen_idx = df_all_results['overfitting_score'].idxmin()
best_gen = df_all_results.loc[best_gen_idx]

print("\n" + "=" * 120)
print("Best Generalization Model (Minimum Overfitting Score)")
print("=" * 120)
print(f"Noise level: σ = {best_gen['sigma_ratio']}")
print(f"Expansion factor: {int(best_gen['n_copies'])}x")
print(f"Overfitting Score = {best_gen['overfitting_score']:.4f}")
print(f"Test R² = {best_gen['test_r2']:.4f}")

# Global best model complete parameters
print("\n" + "=" * 120)
print("[Global Best Model Complete Hyperparameters]")
print("=" * 120)
print(f"Best configuration:")
print(f"  - Noise level (sigma_ratio): {global_best['sigma_ratio']}")
print(f"  - Expansion factor (n_copies): {int(global_best['n_copies'])}x")
print(f"\nBest hyperparameters:")
best_params_dict = eval(global_best['best_params'])
for key, value in best_params_dict.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 120)
print("SVR Optimization Complete! No files were saved.")
print("=" * 120)