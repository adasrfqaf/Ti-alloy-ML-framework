import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
SEED = 42
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = r"D:\python_file\Titanium-based hydrogen storage alloy\模型优化\高斯噪声+SMOTER\数据集\cleaned_BCC_data.csv"
TARGET_COL = 'max_h2_uptake'
TEST_SIZE = 0.2
CV_FOLDS = 5

# Best augmentation parameters identified previously
NOISE_RATIO = 0.05
SMOTER_RATIO = 0.05
NOISE_COPIES = 3
EXTREME_FACTOR = 1.2

# Hyperparameter groups for two-step greedy search
PARAM_GROUP1 = {
    'max_depth': [5],
    'min_samples_split': [15],
    'min_samples_leaf': [5],
    'learning_rate': [0.1],
    'subsample': [0.8]
}
PARAM_GROUP2 = {
    'max_features': [0.6]
}

# ====================== Helper Functions ======================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    return X, y

def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    from sklearn.ensemble import RandomForestRegressor
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    rf_temp.fit(X_train, y_train)
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    return selected_features

def add_gaussian_noise(X, y, sigma_ratio, n_copies):
    X_noisy_list, y_noisy_list = [], []
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            std_col = X[col].std()
            if std_col > 0:
                noise = np.random.normal(0, sigma_ratio * std_col, len(X))
                X_copy[col] += noise
        X_noisy_list.append(X_copy)
        y_noisy_list.append(y)
    X_noisy = pd.concat(X_noisy_list, ignore_index=True)
    y_noisy = np.concatenate(y_noisy_list)
    return X_noisy, y_noisy

def smoter_manual(X, y, smoter_ratio, extreme_factor=EXTREME_FACTOR, k=5, bins=10):
    np.random.seed(SEED)
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)
    if n_generate <= 0:
        return pd.DataFrame(columns=X.columns), np.array([])
    y_percentile = np.percentile(y, np.linspace(0, 100, bins+1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices-1]
    sample_weights /= sample_weights.sum()
    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)
    X_smoter_list, y_smoter_list = [], []
    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y[idx]
        distances, indices = nn.kneighbors(x_seed.reshape(1,-1), n_neighbors=k+1)
        neighbor_idx = np.random.choice(indices[0][1:])
        x_neighbor = X.iloc[neighbor_idx].values
        y_neighbor = y[neighbor_idx]
        lam = np.random.uniform()
        x_new = x_seed + lam*(x_neighbor - x_seed)
        y_new = y_seed + lam*(y_neighbor - y_seed)
        X_smoter_list.append(x_new)
        y_smoter_list.append(y_new)
    X_smoter = pd.DataFrame(X_smoter_list, columns=X.columns)
    y_smoter = np.array(y_smoter_list)
    return X_smoter, y_smoter

def augment_data(X, y, noise_ratio, smoter_ratio):
    if smoter_ratio > 0:
        X_smoter, y_smoter = smoter_manual(X, y, smoter_ratio)
    else:
        X_smoter, y_smoter = pd.DataFrame(columns=X.columns), np.array([])
    X_combined = pd.concat([X, X_smoter], ignore_index=True)
    y_combined = np.concatenate([y, y_smoter])
    if noise_ratio > 0:
        X_noise, y_noise = add_gaussian_noise(X_combined, y_combined, noise_ratio, NOISE_COPIES)
        X_final = pd.concat([X_combined, X_noise], ignore_index=True)
        y_final = np.concatenate([y_combined, y_noise])
    else:
        X_final, y_final = X_combined, y_combined
    return X_final, y_final

def greedy_grid_search_gbdt_early_stop(X_train, y_train):
    """
    Two‑step greedy grid search for GBDT hyperparameters.
    Returns: (best_params, final_model)
    """
    base_gbdt = GradientBoostingRegressor(n_estimators=200, random_state=SEED, max_features='sqrt')
    gs1 = GridSearchCV(base_gbdt, PARAM_GROUP1, cv=CV_FOLDS, scoring='r2', n_jobs=1, verbose=0)
    gs1.fit(X_train, y_train)
    best_params1 = gs1.best_params_

    base_gbdt2 = GradientBoostingRegressor(random_state=SEED, **best_params1)
    gs2 = GridSearchCV(base_gbdt2, PARAM_GROUP2, cv=CV_FOLDS, scoring='r2', n_jobs=1, verbose=0)
    gs2.fit(X_train, y_train)
    best_params2 = gs2.best_params_

    base_params = {**best_params1, **best_params2}

    # Early stopping with internal validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)
    final_model = GradientBoostingRegressor(
        n_estimators=500, validation_fraction=0.1, n_iter_no_change=20,
        tol=1e-4, random_state=SEED, **base_params
    )
    final_model.fit(X_tr, y_tr)
    return base_params, final_model

def evaluate(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return r2, rmse, mae

# ====================== Main Program ======================
if __name__ == '__main__':
    # 1. Load data
    X, y = load_data(FILE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # 2. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 3. Feature selection (95% cumulative importance)
    selected_features = select_features_by_importance(X_train_scaled, y_train, target_ratio=0.95)
    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]

    # 4. Data augmentation
    X_aug, y_aug = augment_data(X_train_selected, y_train, NOISE_RATIO, SMOTER_RATIO)

    # 5. Train model with greedy grid search + early stopping
    best_params, best_model = greedy_grid_search_gbdt_early_stop(X_aug, y_aug)

    # 6. Cross-validation on augmented training set
    cv_scores = cross_val_score(best_model, X_aug, y_aug, cv=CV_FOLDS, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # 7. Evaluate on training (augmented) and test sets
    train_r2, train_rmse, train_mae = evaluate(best_model, X_aug, y_aug)
    test_r2, test_rmse, test_mae = evaluate(best_model, X_test_selected, y_test)
    overfit = train_r2 - test_r2

    # 8. Print only the required metrics and best parameters
    print(f"Train R2: {train_r2:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Overfitting (Train R2 - Test R2): {overfit:.4f}")
    print(f"CV R2 (5-fold, mean±std): {cv_mean:.4f} ± {cv_std:.4f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")