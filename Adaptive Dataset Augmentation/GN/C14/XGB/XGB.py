"""
XGBoost Fine-Tuning with Gaussian Noise Augmentation (C14 Dataset)
This script loads C14 data, selects top features, augments with Gaussian noise,
and tunes XGBoost hyperparameters using RandomizedSearchCV.
Includes both train and test metrics to evaluate overfitting.
No files are saved; all results are printed to console.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


# ====================== Set Global Random Seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


SEED = 42
set_global_seed(SEED)


# ====================== Configuration ======================
FILE_PATH = r'D:\python_file\Titanium-based hydrogen storage alloy\数据集扩充--高斯噪声\数据集\C14_processed_final.xlsx'
TARGET_COL = '最大吸氢量 (wt.%)'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 17

# Hyperparameter search space
PARAM_DIST = {
    'n_estimators': [260],
    'max_depth': [7],
    'learning_rate': [0.01],
    'subsample': [0.95],
    'colsample_bytree': [0.8],
    'reg_lambda': [0.15],
    'reg_alpha': [0.001]
}
N_ITER = 40

# Noise levels to evaluate
SIGMA_RATIOS = [0.01,0.05]


# ====================== Data Loading & Preprocessing ======================
def load_data(path):
    """Load data from Excel, drop additive columns and non-numeric columns."""
    df = pd.read_excel(path, sheet_name=0)
    # Drop columns containing '含量(at.%)' (additive content columns)
    drop_cols = [col for col in df.columns if '含量(at.%)' in col]
    if drop_cols:
        print(f"Dropping additive content columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Drop non-numeric columns (except target)
    for col in df.columns:
        if col != TARGET_COL and not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Dropping non-numeric column: {col}")
            df = df.drop(columns=[col])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    X = X.select_dtypes(include=[np.number])
    return X, y


def select_top_features(X_train, y_train, top_n=15):
    """Select top features using XGBoost importance."""
    temp_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=1
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_model.fit(X_scaled, y_train)

    importance = temp_model.feature_importances_
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]

    print(f"\nSelected top {top_n} features (by importance):")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (importance: {importance[sorted_idx[i - 1]]:.4f})")
    return top_features


def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=3):
    """Add Gaussian noise with standard deviation proportional to feature std."""
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
    """Evaluate model on given dataset and return R², MAE, RMSE."""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse


# ====================== Main Pipeline ======================
print("Loading C14 dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")

# Fixed train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection (based on original training set, no noise)
top_features = select_top_features(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"Training set after feature selection: {X_train_sel.shape}")

# Compute feature standard deviations for relative noise
feature_stds = X_train_sel.std(axis=0)

# Store results
results = []

# Tune for each noise level
for sigma_ratio in SIGMA_RATIOS:
    print(f"\n{'=' * 50}")
    print(f"Noise level sigma_ratio = {sigma_ratio}")
    print('=' * 50)

    if sigma_ratio == 0.0:
        X_train_aug = X_train_sel
        y_train_aug = y_train
        print("Using original training set (no noise)")
    else:
        X_train_aug, y_train_aug = add_relative_gaussian_noise(
            X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=3
        )
        print(f"Augmented training set: {len(X_train_aug)} (original {len(X_train_sel)} + 3 noise copies)")

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_sel)

    # Hyperparameter tuning with fixed random state and single-thread
    xgb = XGBRegressor(
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=1
    )
    random_search = RandomizedSearchCV(
        xgb, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
        scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=1
    )
    random_search.fit(X_train_scaled, y_train_aug)

    best_params = random_search.best_params_
    best_cv_r2 = random_search.best_score_
    best_model = random_search.best_estimator_

    # Evaluate on training set (augmented)
    train_r2, train_mae, train_rmse = evaluate_model(best_model, X_train_scaled, y_train_aug)

    # Evaluate on test set
    test_r2, test_mae, test_rmse = evaluate_model(best_model, X_test_scaled, y_test)

    # Calculate overfitting gap
    overfit_gap = train_r2 - test_r2

    print(f"Best params: {best_params}")
    print(f"Best CV R²: {best_cv_r2:.4f}")
    print(f"Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Test  R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    print(f"Overfitting gap (Train - Test R²): {overfit_gap:.4f}")

    results.append({
        'sigma_ratio': sigma_ratio,
        'cv_r2': best_cv_r2,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'overfit_gap': overfit_gap,
        'best_params': best_params
    })

# ====================== Results Summary ======================
print("\n\n" + "=" * 85)
print("XGBoost Performance Comparison at Different Noise Levels (C14 Dataset, Top 15 Features)")
print("=" * 85)

df_results = pd.DataFrame([
    {
        'σ_ratio': r['sigma_ratio'],
        'CV R²': r['cv_r2'],
        'Train R²': r['train_r2'],
        'Test R²': r['test_r2'],
        'Overfit Gap': r['overfit_gap'],
        'Train MAE': r['train_mae'],
        'Test MAE': r['test_mae'],
        'Train RMSE': r['train_rmse'],
        'Test RMSE': r['test_rmse']
    }
    for r in results
])
print(df_results.to_string(index=False))

# Best by Test R²
best_idx = df_results['Test R²'].idxmax()
best_sigma = df_results.loc[best_idx, 'σ_ratio']
print(f"\n🏆 Best noise level (by Test R²): σ_ratio = {best_sigma}")
print(
    f"   Test R² = {df_results.loc[best_idx, 'Test R²']:.4f}, "
    f"Train R² = {df_results.loc[best_idx, 'Train R²']:.4f}, "
    f"Overfit Gap = {df_results.loc[best_idx, 'Overfit Gap']:.4f}"
)

# Best by overfitting gap (minimum gap)
best_gen_idx = df_results['Overfit Gap'].idxmin()
best_gen_sigma = df_results.loc[best_gen_idx, 'σ_ratio']
print(f"\n🌟 Best generalization (minimum overfit gap): σ_ratio = {best_gen_sigma}")
print(
    f"   Overfit Gap = {df_results.loc[best_gen_idx, 'Overfit Gap']:.4f}, "
    f"Test R² = {df_results.loc[best_gen_idx, 'Test R²']:.4f}"
)

print("\n✅ Evaluation complete. No files were saved.")