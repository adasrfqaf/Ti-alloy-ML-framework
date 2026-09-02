"""
Supplementary Code for TiFe Phase - XGBoost Fine-Tuning with SMOTER Augmentation

This script performs grid search for XGBoost hyperparameter optimization
with SMOTER augmentation on the TiFe phase dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neighbors import NearestNeighbors
import joblib
import warnings

warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
SEED = 49
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

# ====================== Optimal Configuration ======================
EXTREME_PERCENTILE = 20
N_COPIES = 1
K_NEIGHBORS = 5
NOISE_STD = 0.05
N_FEATURES = 15


# ====================== SMOTER Bilateral Interpolation ======================
def smoter_extreme_oversample(X_train, y_train, extreme_percentile, n_copies=1,
                              k_neighbors=5, noise_std=0.05, random_state=None):
    """
    SMOTER-style oversampling for extreme low and high target values.

    Parameters:
    - extreme_percentile: percentile threshold for extremes
    - n_copies: number of synthetic samples per extreme sample
    - k_neighbors: number of KNN neighbors for interpolation
    - noise_std: standard deviation of Gaussian noise
    """
    if random_state is None:
        random_state = np.random.RandomState(SEED)
    X = np.array(X_train)
    y = np.array(y_train).flatten()

    low_thresh = np.percentile(y, extreme_percentile)
    high_thresh = np.percentile(y, 100 - extreme_percentile)

    low_idx = np.where(y <= low_thresh)[0]
    high_idx = np.where(y >= high_thresh)[0]

    print(f"  Low threshold: {low_thresh:.4f} (bottom {extreme_percentile}%), samples: {len(low_idx)}")
    print(f"  High threshold: {high_thresh:.4f} (top {extreme_percentile}%), samples: {len(high_idx)}")

    X_aug = []
    y_aug = []

    def augment_group(indices):
        if len(indices) == 0:
            return
        group_X = X[indices]
        if len(group_X) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(group_X)))
            nbrs.fit(group_X)
        for i, idx in enumerate(indices):
            for _ in range(n_copies):
                if len(group_X) > 1:
                    pos = np.where(indices == idx)[0][0]
                    distances, neigh_positions = nbrs.kneighbors([group_X[pos]])
                    candidates = [p for p in neigh_positions[0] if p != pos]
                    if len(candidates) == 0:
                        neigh_pos = pos
                    else:
                        neigh_pos = random_state.choice(candidates)
                    neighbor_idx = indices[neigh_pos]
                    gap = random_state.random()
                    new_X = X[idx] + gap * (X[neighbor_idx] - X[idx])
                    new_y = y[idx] + gap * (y[neighbor_idx] - y[idx])
                else:
                    new_X = X[idx].copy()
                    new_y = y[idx]
                new_X += random_state.normal(0, noise_std, len(new_X))
                new_y += random_state.normal(0, noise_std * y.std())
                X_aug.append(new_X)
                y_aug.append(new_y)

    augment_group(low_idx)
    augment_group(high_idx)

    if len(X_aug) > 0:
        X_aug = np.vstack([X, np.array(X_aug)])
        y_aug = np.concatenate([y, np.array(y_aug)])
    else:
        X_aug, y_aug = X, y
    return X_aug, y_aug


# ====================== Load Data ====================
df = pd.read_csv('TiFe_data.csv', encoding='utf-8-sig')
print("=" * 80)
print("TiFe Phase Dataset - XGBoost Fine-Tuning (SMOTER Augmentation + Grid Search)")
print("=" * 80)
print(f"Original data shape: {df.shape}")

target_col = 'Max_H2_Uptake_wt_pct'
feature_cols = [col for col in df.columns if col != target_col]

# Data type processing
for col in feature_cols:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
    elif df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols].fillna(0)
y = df[target_col].fillna(0)

print(f"Number of features: {X.shape[1]}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Train-test split
X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED
)
print(f"\nOriginal training set: {len(X_train_base)}")
print(f"Test set: {len(X_test)}")

# ==================== SMOTER Augmentation ====================
print("\nApplying SMOTER augmentation...")
X_train_aug, y_train_aug = smoter_extreme_oversample(
    X_train_base.values, y_train_base.values,
    extreme_percentile=EXTREME_PERCENTILE,
    n_copies=N_COPIES,
    k_neighbors=K_NEIGHBORS,
    noise_std=NOISE_STD,
    random_state=rng
)
print(f"Augmented training set size: {len(X_train_aug)} (Added: {len(X_train_aug)-len(X_train_base)})")

# ==================== Feature Selection ====================
print(f"\nFeature selection (Mutual Information, top {N_FEATURES} features)...")
selector = SelectKBest(mutual_info_regression, k=N_FEATURES)
X_train_selected = selector.fit_transform(X_train_aug, y_train_aug)
X_test_selected = selector.transform(X_test.values)

selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]
print("Selected features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# ==================== Grid Search ====================
print("\n" + "=" * 80)
print("XGBoost Grid Search (Fine-Tuning)")
print("=" * 80)

param_grid = {
    'n_estimators': [150],
    'max_depth': [5],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.7],
    'reg_alpha': [0.5],
    'reg_lambda': [1]
}

xgb_base = XGBRegressor(random_state=SEED, verbosity=0)
grid_search = GridSearchCV(
    xgb_base, param_grid,
    cv=3,
    scoring='r2',
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_train_selected, y_train_aug)

best_xgb = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ==================== Evaluate Best Model ====================
y_train_pred = best_xgb.predict(X_train_selected)
y_test_pred = best_xgb.predict(X_test_selected)

train_r2 = r2_score(y_train_aug, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train_aug, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
overfit = train_r2 - test_r2

print("\n" + "=" * 80)
print("Best XGBoost Model Evaluation Results (Fixed Test Set)")
print("=" * 80)
print(f"Training: R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")
print(f"Test:     R²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
print(f"Overfit gap: {overfit:.4f}")

# ==================== Cross-Validation Evaluation ====================
print("\n" + "=" * 80)
print("Cross-Validation Evaluation (5-fold, on augmented training set)")
print("=" * 80)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_r2 = cross_val_score(best_xgb, X_train_selected, y_train_aug, cv=cv, scoring='r2')
cv_mae = -cross_val_score(best_xgb, X_train_selected, y_train_aug, cv=cv, scoring='neg_mean_absolute_error')
cv_rmse = -cross_val_score(best_xgb, X_train_selected, y_train_aug, cv=cv, scoring='neg_root_mean_squared_error')

print(f"R²  (5-fold): Mean={cv_r2.mean():.4f}, Std={cv_r2.std():.4f}")
print(f"MAE (5-fold): Mean={cv_mae.mean():.4f}, Std={cv_mae.std():.4f}")
print(f"RMSE(5-fold): Mean={cv_rmse.mean():.4f}, Std={cv_rmse.std():.4f}")

# ==================== Save Model ====================
joblib.dump(best_xgb, 'TiFe_best_xgb_tuned.pkl')
joblib.dump(scaler, 'TiFe_xgb_scaler.pkl')
joblib.dump(selector, 'TiFe_xgb_selector.pkl')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importance.to_csv('TiFe_xgb_feature_importance_tuned.csv', index=False)
