"""
Supplementary Code for C14 Laves Phase - XGBoost Grid Search Optimization with SMOTER Augmentation

This script performs hyperparameter optimization for XGBoost with SMOTER augmentation
on the C14 Laves phase dataset. Only essential evaluation metrics and best parameters are printed.
No files are saved.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

# ==================== Reproducible Configuration ====================
RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)

EXTREME_PERCENTILE = 10
N_COPIES = 1
N_FEATURES = 15
K_NEIGHBORS = 5
NOISE_STD = 0.05


# ==================== SMOTER Function (Reproducible) ====================
def smoter_extreme_oversample(X_train, y_train, extreme_percentile, n_copies=5,
                              k_neighbors=5, noise_std=0.05, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(RANDOM_STATE)
    X = np.array(X_train)
    y = np.array(y_train).flatten()

    low_thresh = np.percentile(y, extreme_percentile)
    high_thresh = np.percentile(y, 100 - extreme_percentile)

    low_idx = np.where(y <= low_thresh)[0]
    high_idx = np.where(y >= high_thresh)[0]

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


# ==================== Load Data ====================
df = pd.read_csv('C14_data.csv', encoding='utf-8-sig')
print("=" * 60)
print("C14 Laves Phase - XGBoost Grid Search")
print("=" * 60)

target_col = 'Max_H2_Uptake_wt_pct'
feature_cols = [col for col in df.columns if col != target_col]

for col in feature_cols:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
    elif df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols].fillna(0)
y = df[target_col].fillna(0)

# Standardization & Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

# SMOTER Augmentation
X_train_aug, y_train_aug = smoter_extreme_oversample(
    X_train_base.values, y_train_base.values,
    extreme_percentile=EXTREME_PERCENTILE,
    n_copies=N_COPIES,
    k_neighbors=K_NEIGHBORS,
    noise_std=NOISE_STD,
    random_state=rng
)

# Feature Selection
selector = SelectKBest(mutual_info_regression, k=N_FEATURES)
X_train_selected = selector.fit_transform(X_train_aug, y_train_aug)
X_test_selected = selector.transform(X_test.values)

# Grid Search
param_grid = {
    'n_estimators': [155],
    'max_depth': [6],
    'learning_rate': [0.06],
    'subsample': [0.7],
    'colsample_bytree': [0.7]
}

xgb_base = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
grid_search = GridSearchCV(
    xgb_base, param_grid,
    cv=5, scoring='r2',
    n_jobs=1, verbose=0  # 静默模式
)
grid_search.fit(X_train_selected, y_train_aug)

best_xgb = grid_search.best_estimator_
best_params = grid_search.best_params_

# ==================== Evaluation ====================
y_train_pred = best_xgb.predict(X_train_selected)
y_test_pred = best_xgb.predict(X_test_selected)

train_r2 = r2_score(y_train_aug, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train_aug, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
overfit = train_r2 - test_r2

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores_r2 = cross_val_score(best_xgb, X_train_selected, y_train_aug, cv=cv, scoring='r2')
cv_r2_mean, cv_r2_std = cv_scores_r2.mean(), cv_scores_r2.std()

# ==================== Output (only essential) ====================
print("\n" + "=" * 60)
print("Best Parameters")
print("=" * 60)
for param, value in best_params.items():
    print(f"  {param}: {value}")

print("\n" + "=" * 60)
print("Evaluation Metrics")
print("=" * 60)
print(f"Train R²:     {train_r2:.4f}")
print(f"Test R²:      {test_r2:.4f}")
print(f"Train MAE:    {train_mae:.4f}")
print(f"Test MAE:     {test_mae:.4f}")
print(f"Train RMSE:   {train_rmse:.4f}")
print(f"Test RMSE:    {test_rmse:.4f}")
print(f"Overfit gap (Train-Test R²): {overfit:.4f}")
print(f"CV R² (5-fold): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
print("=" * 60)
print("\n✅ Evaluation complete. No files were saved.")