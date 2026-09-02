import pandas as pd
import numpy as np
import random
import os
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
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
# Use the CSV file with English column names
FILE_PATH = "TiFe_data.csv"   # Change to full path if needed
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
CV_FOLDS = 5

# Data augmentation parameters
NOISE_RATIOS = [0.05]
SMOTER_RATIOS = [0.08]
N_COPIES = 3

# 20 selected features (English names, exactly as in CSV header)
SELECTED_FEATURES = [
    'Test_Temperature_K',
    'Element_at_pct_Fe',
    'Unit_Cell_Volume_Å3',
    'Element_at_pct_Zr',
    'Second_Phase_FeCr',
    'Initial_Hydrogen_Pressure_MPa',
    'Second_Phase_Ti2Fe',
    'Hydrogen_Absorption_Cycles',
    'Element_at_pct_Ce',
    'Second_Phase_TiFe2',
    'Element_at_pct_Cr',
    'Element_at_pct_Al',
    'Additive_Al',
    'Element_at_pct_V',
    'Element_at_pct_Mn',
    'Second_Phase_Ti4Fe2O',
    'Element_at_pct_Ti',
    'Element_at_pct_Mo',
    'Additive_Mo',
    'Second_Phase_Ce_Phase'
]

# Fixed hyperparameters
FIXED_PARAMS = {
    'num_leaves': 15,
    'min_child_samples': 12,
    'min_split_gain': 0,
    'learning_rate': 0.07,
    'n_estimators': 220,
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.78,
    'reg_lambda': 1,
    'random_state': SEED,
    'n_jobs': 1,
    'verbosity': -1
}


# ====================== Helper: Clean column names ======================
def clean_feature_names(df):
    """Replace non-alphanumeric/underscore chars with underscores, ensure uniqueness."""
    original_names = df.columns.tolist()
    cleaned_names = []
    for name in original_names:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        if not clean:
            clean = 'feature'
        cleaned_names.append(clean)
    for i, name in enumerate(cleaned_names):
        if cleaned_names.count(name) > 1:
            cleaned_names[i] = f"{name}_{i}"
    df_cleaned = df.copy()
    df_cleaned.columns = cleaned_names
    return df_cleaned, dict(zip(cleaned_names, original_names))


# ====================== Data Loading (CSV) ======================
def load_and_preprocess(path):
    # Read CSV with UTF-8 BOM handling
    df = pd.read_csv(path, encoding='utf-8-sig')
    print(f"Raw data shape: {df.shape}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {list(df.columns)}")
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    bool_cols = X.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    print(f"After processing: {X.shape[1]} features, {len(X)} samples")
    return X, y


# ====================== Data Augmentation ======================
def add_gaussian_noise(X, y, sigma_ratio, n_copies=3):
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


def augment_data(X, y, noise_ratio, smoter_ratio, n_copies=3, random_state=SEED):
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


# ====================== Cross-validation and Evaluation ======================
def cross_val_r2(X, y, params, cv=CV_FOLDS):
    model = lgb.LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores.mean(), scores.std()


def evaluate(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return r2, rmse, mae


# ====================== Main Pipeline ======================
if __name__ == '__main__':
    print("Loading data from CSV...")
    X, y = load_and_preprocess(FILE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"Original training set: {X_train.shape}, test set: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    available_features = [f for f in SELECTED_FEATURES if f in X_train_scaled.columns]
    if len(available_features) < len(SELECTED_FEATURES):
        missing = set(SELECTED_FEATURES) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    X_train_selected = X_train_scaled[available_features]
    X_test_selected = X_test_scaled[available_features]
    print(f"After feature selection: training set shape {X_train_selected.shape}")

    X_train_clean, name_map = clean_feature_names(X_train_selected)
    X_test_clean, _ = clean_feature_names(X_test_selected)
    print("Feature names cleaned; mapping saved.")

    all_results = []
    total = len(NOISE_RATIOS) * len(SMOTER_RATIOS)
    idx = 0
    for noise in NOISE_RATIOS:
        for smoter in SMOTER_RATIOS:
            idx += 1
            print(f"\n{'='*60}")
            print(f"Combination {idx}/{total}: noise={noise}, smoter={smoter}")
            print('='*60)

            X_aug, y_aug = augment_data(X_train_selected, y_train, noise, smoter, n_copies=N_COPIES)
            X_aug_clean, _ = clean_feature_names(X_aug)
            print(f"Augmented training set size: {len(X_aug_clean)} (original {len(X_train_selected)})")

            cv_r2_mean, cv_r2_std = cross_val_r2(X_aug_clean, y_aug, FIXED_PARAMS, cv=CV_FOLDS)
            print(f"CV R² (mean ± std): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

            model = lgb.LGBMRegressor(**FIXED_PARAMS)
            model.fit(X_aug_clean, y_aug)

            train_r2, train_rmse, train_mae = evaluate(model, X_aug_clean, y_aug)
            test_r2, test_rmse, test_mae = evaluate(model, X_test_clean, y_test)

            print(f"Train R²={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
            print(f"Test  R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

            all_results.append({
                'noise_ratio': noise,
                'smoter_ratio': smoter,
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'model': model,
                'name_map': name_map
            })

    print("\n" + "="*80)
    print("LightGBM Performance Summary (TiFe dataset, fixed 20 features, fixed hyperparameters)")
    summary = []
    for res in all_results:
        summary.append({
            'noise': res['noise_ratio'],
            'smoter': res['smoter_ratio'],
            'CV_R2_mean': f"{res['cv_r2_mean']:.4f}",
            'CV_R2_std': f"{res['cv_r2_std']:.4f}",
            'Train_R2': f"{res['train_r2']:.4f}",
            'Train_RMSE': f"{res['train_rmse']:.4f}",
            'Train_MAE': f"{res['train_mae']:.4f}",
            'Test_R2': f"{res['test_r2']:.4f}",
            'Test_RMSE': f"{res['test_rmse']:.4f}",
            'Test_MAE': f"{res['test_mae']:.4f}",
            'Overfit': f"{res['train_r2'] - res['test_r2']:.4f}"
        })
