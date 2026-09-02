import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# ====================== Fixed random seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = 'TiFe_data.csv'                     # Use new CSV file
TARGET_COL = 'Max_H2_Uptake_wt_pct'             # English target column
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 18

# Random forest hyperparameter search space
PARAM_DIST = {
    'n_estimators': [220, 280, 250],
    'max_depth': [7],
    'min_samples_split': [6,7],
    'min_samples_leaf': [3,4],
    'max_features': [0.75],
    'bootstrap': [False]
}
N_ITER = 80

# Noise levels
SIGMA_RATIOS = [0.01, 0.02, 0.05, 0.08, 0.1]

# ====================== Data loading and preprocessing ======================
def load_data(path):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' does not exist")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    X = X.loc[:, X.var() != 0]                  # Remove constant columns
    print(f"Number of retained numerical features: {X.shape[1]}")
    return X, y

def select_top_features_rf(X_train, y_train, top_n=18):
    """Select top features using a Random Forest model"""
    temp_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_depth=20
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_model.fit(X_scaled, y_train)
    importance = temp_model.feature_importances_
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    print(f"\nTop {top_n} selected features (based on Random Forest importance):")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (importance: {importance[sorted_idx[i-1]]:.4f})")
    return top_features

def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=3):
    """Add relative Gaussian noise: noise std = sigma_ratio * feature std"""
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
    """Return R², MAE, RMSE"""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse

# ====================== Main workflow ======================
print("Loading TiFe dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")

# Fixed train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection (fixed to 18 features)
top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"Training set dimension after feature selection: {X_train_sel.shape}")

# Compute feature standard deviations on training set (for relative noise)
feature_stds = X_train_sel.std(axis=0)

# Store results (no file output)
results = []

# Tune for each noise level
for sigma_ratio in SIGMA_RATIOS:
    print(f"\n{'='*50}")
    print(f"Noise level sigma_ratio = {sigma_ratio}")
    print('='*50)

    if sigma_ratio == 0.0:
        X_train_aug = X_train_sel
        y_train_aug = y_train
        print("Using original training set (no noise)")
    else:
        X_train_aug, y_train_aug = add_relative_gaussian_noise(
            X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=3
        )
        print(f"Augmented training set size: {len(X_train_aug)} (original {len(X_train_sel)} + 3 noisy copies)")

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_sel)

    # Random Forest model
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    random_search = RandomizedSearchCV(
        rf, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
        scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=1
    )
    random_search.fit(X_train_scaled, y_train_aug)

    best_params = random_search.best_params_
    best_cv_r2 = random_search.best_score_  # mean CV R²
    best_model = random_search.best_estimator_

    # --- Extract all split scores for the best parameter set ---
    best_index = random_search.best_index_
    # Retrieve the test scores for each fold
    split_scores = [random_search.cv_results_[f'split{i}_test_score'][best_index] for i in range(CV_FOLDS)]
    cv_mean = np.mean(split_scores)
    cv_std = np.std(split_scores, ddof=1)  # sample standard deviation

    # ---- Evaluate on original (noise-free) training set ----
    X_train_original_scaled = scaler.transform(X_train_sel)   # use same scaler
    train_r2, train_mae, train_rmse = evaluate_model(best_model, X_train_original_scaled, y_train)

    # ---- Evaluate on test set ----
    test_r2, test_mae, test_rmse = evaluate_model(best_model, X_test_scaled, y_test)

    # ---- Compute overfitting gap ----
    gap_r2 = train_r2 - test_r2
    gap_rmse = train_rmse - test_rmse

    print(f"Best parameters: {best_params}")
    # 修改此处：显示 CV R² 均值 ± 标准差
    print(f"Best cross-validation R²: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Training set (original) R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Test set R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    print(f"Overfitting gap (Train R² - Test R²): {gap_r2:.4f}, (Train RMSE - Test RMSE): {gap_rmse:.4f}")

    results.append({
        'sigma_ratio': sigma_ratio,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'gap_r2': gap_r2,
        'gap_rmse': gap_rmse,
        'best_params': best_params
    })

# ====================== Summary output ======================
print("\n\n" + "="*70)
print("Performance comparison of RandomForest under different noise levels (training & test)")
print("="*70)
df_results = pd.DataFrame([
    {
        'σ_ratio': r['sigma_ratio'],
        # 修改此处：CV R² 显示为 均值 ± 标准差
        'CV R² (mean±std)': f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        'Train R²': r['train_r2'],
        'Test R²': r['test_r2'],
        'Gap R²': r['gap_r2'],
        'Train RMSE': r['train_rmse'],
        'Test RMSE': r['test_rmse'],
        'Gap RMSE': r['gap_rmse']
    }
    for r in results
])
print(df_results.to_string(index=False))

# Output the best noise level (based on Test R²)
best_idx = max(range(len(results)), key=lambda i: results[i]['test_r2'])
best_sigma = results[best_idx]['sigma_ratio']
print(f"\n🏆 Best noise level (by Test R²): σ_ratio = {best_sigma}")
print(f"   Corresponding Test R² = {results[best_idx]['test_r2']:.4f}, "
      f"Train R² = {results[best_idx]['train_r2']:.4f}, "
      f"Gap R² = {results[best_idx]['gap_r2']:.4f}")
print(f"   RMSE: Test {results[best_idx]['test_rmse']:.4f}, "
      f"Train {results[best_idx]['train_rmse']:.4f}, "
      f"Gap RMSE = {results[best_idx]['gap_rmse']:.4f}")

print("\nNote: Training set metrics are evaluated on the original (noise‑free) training data to measure overfitting.")