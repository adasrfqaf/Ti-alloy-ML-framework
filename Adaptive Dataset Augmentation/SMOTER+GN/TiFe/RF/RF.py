import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = "TiFe_data.csv"          # Update if needed
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
CV_FOLDS = 5

# Fixed augmentation parameters (best found)
FIXED_NOISE = 0.01
FIXED_SMOTER = 0.08
N_COPIES = 3

# Fixed number of features (based on prior best)
N_FEATURES = 20

# ====================== Fine-Grained Hyperparameter Grid ======================
# Narrow ranges around the best known parameters
RF_PARAM_GRID_FINE = {
    'n_estimators': [178],
    'max_depth': [8],
    'min_samples_split': [10],
    'min_samples_leaf': [3],
    'max_features': [10],
    'bootstrap': [False]   # fixed to False based on best result
}

# ====================== Data Loading ======================
def load_and_preprocess(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {list(df.columns)}")
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    bool_cols = X.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    return X, y

# ====================== Feature Selection ======================
def select_top_n_features(X_train, y_train, n_features):
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    rf_temp.fit(X_train, y_train)
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    selected_features = X_train.columns[indices]
    return selected_features

# ====================== Data Augmentation ======================
def add_gaussian_noise(X, y, sigma_ratio, n_copies=1):
    if sigma_ratio == 0 or n_copies == 0:
        return pd.DataFrame(columns=X.columns), np.array([])
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

def smoter_manual(X, y, smoter_ratio, extreme_factor=2.0, k=5, bins=10, random_state=SEED):
    if smoter_ratio == 0:
        return pd.DataFrame(columns=X.columns), np.array([])
    np.random.seed(random_state)
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)
    if n_generate <= 0:
        return pd.DataFrame(columns=X.columns), np.array([])

    y_percentile = np.percentile(y, np.linspace(0, 100, bins + 1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices - 1]
    sample_weights = sample_weights / sample_weights.sum()

    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)

    X_smoter_list, y_smoter_list = [], []
    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y[idx]
        distances, indices = nn.kneighbors(x_seed.reshape(1, -1), n_neighbors=k + 1)
        neighbor_idx = np.random.choice(indices[0][1:])
        x_neighbor = X.iloc[neighbor_idx].values
        y_neighbor = y[neighbor_idx]
        lam = np.random.uniform()
        x_new = x_seed + lam * (x_neighbor - x_seed)
        y_new = y_seed + lam * (y_neighbor - y_seed)
        X_smoter_list.append(x_new)
        y_smoter_list.append(y_new)

    X_smoter = pd.DataFrame(X_smoter_list, columns=X.columns)
    y_smoter = np.array(y_smoter_list)
    return X_smoter, y_smoter

def augment_data(X, y, noise_ratio, smoter_ratio, n_copies=1, random_state=SEED):
    np.random.seed(random_state)
    X_smoter, y_smoter = smoter_manual(X, y, smoter_ratio)
    X_combined = pd.concat([X, X_smoter], ignore_index=True)
    y_combined = np.concatenate([y, y_smoter])
    if noise_ratio > 0 and n_copies > 0:
        X_noise, y_noise = add_gaussian_noise(X_combined, y_combined, noise_ratio, n_copies)
        if len(X_noise) > 0:
            X_final = pd.concat([X_combined, X_noise], ignore_index=True)
            y_final = np.concatenate([y_combined, y_noise])
        else:
            X_final, y_final = X_combined, y_combined
    else:
        X_final, y_final = X_combined, y_combined
    return X_final, y_final

# ====================== Evaluation ======================
def evaluate(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return r2, rmse, mae

# ====================== Main ======================
if __name__ == '__main__':
    # Load and preprocess data
    X, y = load_and_preprocess(FILE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Feature selection
    selected_features = select_top_n_features(X_train_scaled, y_train, N_FEATURES)
    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]

    # Data augmentation
    X_aug, y_aug = augment_data(X_train_selected, y_train,
                                noise_ratio=FIXED_NOISE,
                                smoter_ratio=FIXED_SMOTER,
                                n_copies=N_COPIES)

    # Grid search for best hyperparameters
    rf = RandomForestRegressor(random_state=SEED, n_jobs=1)
    gs = GridSearchCV(rf, RF_PARAM_GRID_FINE, cv=CV_FOLDS, scoring='r2', n_jobs=1, verbose=0)
    gs.fit(X_aug, y_aug)

    best_params = gs.best_params_
    best_model = gs.best_estimator_

    # Evaluate on training and test sets
    train_r2, train_rmse, train_mae = evaluate(best_model, X_aug, y_aug)
    test_r2, test_rmse, test_mae = evaluate(best_model, X_test_selected, y_test)
    overfit = train_r2 - test_r2

    # Cross-validation on augmented training set (mean ± std)
    cv_scores = cross_val_score(best_model, X_aug, y_aug, cv=CV_FOLDS, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Print only the required metrics and best parameters
    print("===== Best Random Forest Model Results =====")
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