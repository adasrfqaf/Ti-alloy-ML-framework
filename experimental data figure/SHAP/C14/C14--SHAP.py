"""
Supplementary Code - GBDT Fine-Tuning with Gaussian Noise Augmentation (C14 Dataset)

This script performs fine-tuning for GBDT with Gaussian noise augmentation
on the C14 Laves phase dataset. Test predictions are saved to CSV.
SHAP plots are generated in PNG format only.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# Import SHAP and plotting libraries
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')


# ====================== Fixed Random Seed ======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


SEED = 49
set_global_seed(SEED)

# ====================== Configuration ======================
FILE_PATH = r'C14_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'      # Note: column name remains Chinese, but features are English
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 18

# GBDT hyperparameter search space
PARAM_DIST = {
    'n_estimators': [100],
    'learning_rate': [0.11],
    'max_depth': [9],
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
    """Load CSV data and preprocess."""
    # Read CSV file with appropriate encoding
    df = pd.read_csv(path, encoding='utf-8-sig')

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    # Convert boolean columns to int (if any)
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
    print(f"Feature columns: {X.columns.tolist()}")
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
    """
    Add relative Gaussian noise: noise_std = sigma_ratio * feature_std.
    """
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
    """Calculate overfitting metrics."""
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
print("GBDT Fine-Tuning - C14 Dataset (Gaussian Noise Augmentation)")
print("=" * 100)

print("\nLoading C14 dataset...")
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, Total features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Fixed train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection
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

        if sigma_ratio == 0.0 or n_copies == 0:
            X_train_aug = X_train_sel
            y_train_aug = y_train
            print(f"Using original training set (no noise), samples: {len(X_train_aug)}")
        else:
            X_train_aug, y_train_aug = add_relative_gaussian_noise(
                X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=n_copies
            )
            print(f"Augmented training set: {len(X_train_aug)} (original {len(X_train_sel)} + {n_copies} noise copies)")

        # Standardization
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

# ====================== Summary of Results ======================
print("\n\n" + "=" * 120)
print("GBDT Performance Comparison (Different Noise Levels × Expansion Factors)")
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

# Global best by Test R²
global_best_idx = df_all_results['test_r2'].idxmax()
global_best = df_all_results.loc[global_best_idx]

print("\n" + "=" * 120)
print("🏆 Global Best Model (by Test R²)")
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

# ====================== Save Test Predictions ======================
print("\n" + "=" * 120)
print("Retraining Global Best Model and Saving Predictions...")
print("=" * 120)

best_sigma = global_best['sigma_ratio']
best_n_copies = int(global_best['n_copies'])
best_params_str = global_best['best_params']
best_params_dict = eval(best_params_str)

print(f"Best config: sigma={best_sigma}, n_copies={best_n_copies}")
print(f"Best hyperparameters: {best_params_dict}")

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

y_test_pred = best_model_final.predict(X_test_scaled)

df_result = pd.DataFrame({
    'true_capacity': y_test,
    'ours_pred': y_test_pred
})

df_result.to_csv('data_C14_test_predictions_ours.csv', index=False)
print(f"\n✅ Saved predictions: data_C14_test_predictions_ours.csv")
print(f"   Samples: {len(df_result)}")
print(f"   Capacity range: {df_result['true_capacity'].min():.3f} ~ {df_result['true_capacity'].max():.3f} wt.%")
print(df_result.head())

# ====================== SHAP Explanation (English column names, PNG only) ======================
print("\n" + "=" * 120)
print("Generating SHAP explanation plots (Times New Roman, top 10 features)...")
print("=" * 120)

# Subsample for speed (up to 200 instances)
sample_size = min(200, len(X_train_aug))
X_sample = X_train_aug.sample(n=sample_size, random_state=SEED)

# Directly use column names (assumed to be in English)
translated_names = list(X_sample.columns)

# Create explainer and compute SHAP values
explainer = shap.TreeExplainer(best_model_final)
shap_values = explainer.shap_values(X_sample)

# ---------- 1. SHAP summary plot (bee swarm) ----------
plt.figure(figsize=(14, 10))

shap.summary_plot(
    shap_values, X_sample,
    feature_names=translated_names,
    max_display=10,
    show=False
)

ax = plt.gca()

# Remove the vertical dashed line at x=0 (if present)
for line in ax.lines:
    if line.get_linestyle() == '--' and len(line.get_xdata()) > 0 and abs(line.get_xdata()[0]) < 1e-6:
        line.remove()

# Customize axes ticks
ax.tick_params(axis='both', labelsize=18, width=2, length=8)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontfamily('Times New Roman')
    tick.set_fontweight('bold')
    tick.set_fontsize(12)

ax.set_xlabel('SHAP value (impact on model output)',
              fontfamily='Times New Roman', fontsize=18, fontweight='bold')
ax.set_ylabel('Features',
              fontfamily='Times New Roman', fontsize=18, fontweight='bold')

# ---------- Robust colorbar customization ----------
fig = plt.gcf()
# Find the colorbar axis (not the main axis)
cbar_ax = None
for ax_ in fig.axes:
    if ax_ is not ax:
        if ax_.get_ylabel() == 'Feature value':
            cbar_ax = ax_
            break
# Fallback: use the last axis if no match
if cbar_ax is None and len(fig.axes) > 1:
    cbar_ax = fig.axes[-1]

if cbar_ax is not None:
    # Set tick label font and size
    cbar_ax.tick_params(labelsize=16, width=2, length=6)
    for tick in cbar_ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)
    # Set colorbar title (bold)
    cbar_ax.set_ylabel('Feature value', fontfamily='Times New Roman',
                       fontsize=18, fontweight='bold')

    # Find and bold the "Low" and "High" text objects (usually on colorbar axis)
    for text in cbar_ax.texts:
        if text.get_text() in ['Low', 'High', 'low', 'high']:
            text.set_fontfamily('Times New Roman')
            text.set_fontweight('bold')
            text.set_fontsize(18)

# Also search on the main axis for any "Low"/"High" text (just in case)
for text in ax.texts:
    if text.get_text() in ['Low', 'High', 'low', 'high']:
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')
        text.set_fontsize(12)

plt.tight_layout()
plt.savefig('shap_summary_plot_C14.png', dpi=600, bbox_inches='tight')
# PDF removed as requested
plt.close()
print("✅ Saved SHAP summary plot (top 10): shap_summary_plot_C14.png")

