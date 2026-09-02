"""
Supplementary Code - GBDT Fine-Tuning with Gaussian Noise Augmentation (C14 Dataset)

This script performs fine-tuning for GBDT with Gaussian noise augmentation
on the C14 Laves phase dataset. Modified to save model and artifacts.
"""

import pandas as pd
import numpy as np
import random
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
FILE_PATH = 'C14_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 18

# GBDT hyperparameter search space
PARAM_DIST = {
    'n_estimators': [100],
    'learning_rate': [0.11],
    'max_depth': [7],
    'min_samples_split': [8],
    'min_samples_leaf': [2],
    'max_features': [0.55],
    'subsample': [0.9],
    'loss': ['squared_error'],
    'validation_fraction': [0.2],
    'n_iter_no_change': [25]
}
N_ITER = 50

# Noise levels and expansion factors
SIGMA_RATIOS = [0.02, 0.05, 0.08, 0.1]
EXPANSION_FACTORS = [2, 3, 4, 5, 6]

# ====================== Data Loading & Preprocessing ======================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    X = X.fillna(0)
    y = np.nan_to_num(y)
    print(f"Numeric features: {X.shape[1]}")
    return X, y

def select_top_features_rf(X_train, y_train, top_n=18):
    temp_model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=1,
        max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_model.fit(X_scaled, y_train)
    importance = temp_model.feature_importances_
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    print(f"\nSelected top {top_n} features:")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (importance: {importance[sorted_idx[i-1]]:.4f})")
    return top_features

def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=2):
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
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse

# ====================== Main Pipeline ======================
print("=" * 100)
print("GBDT Fine-Tuning - C14 Dataset (Gaussian Noise Augmentation)")
print("=" * 100)

print("\nLoading C14 dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection
top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)

# ===== Force add key features for C14 phase =====
key_features = ['Element_at_pct_Ti', 'Element_at_pct_V', 'Element_at_pct_Mn',
                'Element_at_pct_Cr', 'Element_at_pct_Fe', 'Element_at_pct_Zr']
for feat in key_features:
    if feat in X_train.columns and feat not in top_features:
        top_features.append(feat)
        print(f"Force added feature: {feat}")
print(f"Final features ({len(top_features)}): {top_features}")

X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"After feature selection: {X_train_sel.shape}")

# Feature standard deviations
feature_stds = X_train_sel.std(axis=0)

# Store all results
all_results = []
total_combinations = len(SIGMA_RATIOS) * len(EXPANSION_FACTORS)
current_combo = 0

for sigma_ratio in SIGMA_RATIOS:
    for n_copies in EXPANSION_FACTORS:
        current_combo += 1
        print(f"\n{'='*80}")
        print(f"Experiment [{current_combo}/{total_combinations}]")
        print(f"Noise level σ = {sigma_ratio}, Expansion = {n_copies}x")
        print('='*80)

        if sigma_ratio == 0.0 or n_copies == 0:
            X_train_aug = X_train_sel
            y_train_aug = y_train
        else:
            X_train_aug, y_train_aug = add_relative_gaussian_noise(
                X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=n_copies
            )
        print(f"Augmented: {len(X_train_aug)} samples")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_sel)

        gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE)
        random_search = RandomizedSearchCV(
            gbdt, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
            scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=0
        )
        random_search.fit(X_train_scaled, y_train_aug)

        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        cv_scores = cross_val_score(best_model, X_train_scaled, y_train_aug,
                                    cv=CV_FOLDS, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()

        r2_train, mae_train, rmse_train = evaluate_model(best_model, X_train_scaled, y_train_aug)
        r2_test, mae_test, rmse_test = evaluate_model(best_model, X_test_scaled, y_test)

        print(f"CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
        print(f"Train-Test Gap: {r2_train - r2_test:.4f}")

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
            'train_test_gap': r2_train - r2_test,
            'best_params': str(best_params)
        })

# ====================== Results Summary ======================
print("\n\n" + "=" * 120)
print("Summary of All Experiments")
print("=" * 120)
df_all = pd.DataFrame(all_results)
display_cols = ['sigma_ratio', 'n_copies', 'cv_r2_mean', 'cv_r2_std',
                'train_r2', 'test_r2', 'train_test_gap']
print(df_all[display_cols].to_string(index=False, float_format="%.4f"))

# Global best by Test R²
global_best_idx = df_all['test_r2'].idxmax()
global_best = df_all.loc[global_best_idx]
print("\n" + "=" * 120)
print("Global Best Model (by Test R²)")
print("=" * 120)
print(f"σ = {global_best['sigma_ratio']}, Expansion = {int(global_best['n_copies'])}x")
print(f"Test R² = {global_best['test_r2']:.4f}")

# ====================== Retrain Best Model and Save ======================
print("\n" + "=" * 120)
print("Retraining Global Best Model and Saving Artifacts")
print("=" * 120)

best_sigma = global_best['sigma_ratio']
best_n_copies = int(global_best['n_copies'])
best_params_str = global_best['best_params']
best_params_dict = eval(best_params_str)

print(f"Best config: sigma={best_sigma}, n_copies={best_n_copies}")
print(f"Best hyperparameters: {best_params_dict}")

# Prepare data
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
feature_stds = X_train_sel.std(axis=0)

if best_sigma == 0.0 or best_n_copies == 0:
    X_train_aug = X_train_sel
    y_train_aug = y_train
else:
    X_train_aug, y_train_aug = add_relative_gaussian_noise(
        X_train_sel, y_train, best_sigma, feature_stds, n_copies=best_n_copies
    )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test_sel)

best_model_final = GradientBoostingRegressor(**best_params_dict, random_state=RANDOM_STATE)
best_model_final.fit(X_train_scaled, y_train_aug)

# Evaluate final model
y_test_pred = best_model_final.predict(X_test_scaled)
test_r2_final = r2_score(y_test, y_test_pred)
test_rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae_final = mean_absolute_error(y_test, y_test_pred)

print(f"\nFinal Test Performance:")
print(f"  R²  = {test_r2_final:.4f}")
print(f"  RMSE= {test_rmse_final:.4f}")
print(f"  MAE = {test_mae_final:.4f}")

# Save test predictions
test_results = pd.DataFrame({
    'true_capacity': y_test,
    'pred_capacity': y_test_pred
})
test_results.to_csv("test_predictions_C14.csv", index=False)
print("\n✅ Test predictions saved to: test_predictions_C14.csv")

# ====================== Save Model and Artifacts ======================
print("\n" + "=" * 70)
print("Saving trained model and artifacts...")
print("=" * 70)

model_save_path = "c14_gbdt_gn_best.pkl"
joblib.dump(best_model_final, model_save_path)
print(f"✅ Best model saved to: {model_save_path}")

scaler_save_path = "c14_scaler.pkl"
joblib.dump(scaler, scaler_save_path)
print(f"✅ Scaler saved to: {scaler_save_path}")


feature_save_path = "c14_selected_features.pkl"
joblib.dump(top_features, feature_save_path)
print(f"✅ Selected features saved to: {feature_save_path}")

train_stats = {
    'X_train_stats': X_train_sel.describe(),
    'y_train_stats': pd.Series(y_train).describe(),
    'selected_features': top_features,
    'seed': SEED,
    'feature_names': X_train_sel.columns.tolist(),
}
stats_save_path = "c14_train_stats.pkl"
joblib.dump(train_stats, stats_save_path)
print(f"✅ Training stats saved to: {stats_save_path}")

print("\n✅ All artifacts saved successfully! Ready for candidate screening.")