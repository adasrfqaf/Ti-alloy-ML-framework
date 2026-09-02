"""
Supplementary Code for TiFe Phase - SVR Grid Search with SMOTER Augmentation

This script performs grid search for SVR hyperparameter optimization
with SMOTER augmentation on the TiFe phase dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neighbors import NearestNeighbors
import joblib
import warnings

warnings.filterwarnings('ignore')

SEED = 49
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

# ====================== Fixed Configuration ======================
EXTREME_LOW = 0.10
EXTREME_HIGH = 0.10
K_NEIGHBORS = 5
N_COPIES = 1
NOISE_STD = 0.05
N_FEATURES = 15


# ====================== SMOTER Function ======================
def bilateral_smoter_interpolate(X_train, y_train, low_ratio, high_ratio, k=5, n_copies=1, noise_std=0.05,
                                 random_state=None):
    """
    SMOTER interpolation for both low and high target value tails.

    Parameters:
    - low_ratio: fraction of low-value samples to augment
    - high_ratio: fraction of high-value samples to augment
    - k: number of nearest neighbors
    - n_copies: number of synthetic samples per extreme sample
    - noise_std: standard deviation of Gaussian noise
    """
    if random_state is None:
        random_state = np.random.RandomState(SEED)
    X = np.array(X_train)
    y = np.array(y_train).flatten()
    low_th = np.percentile(y, 100 * low_ratio)
    high_th = np.percentile(y, 100 * (1 - high_ratio))
    low_idx = np.where(y <= low_th)[0]
    high_idx = np.where(y >= high_th)[0]

    print(f"  Low threshold: {low_th:.4f} (bottom {low_ratio * 100:.0f}%), samples: {len(low_idx)}")
    print(f"  High threshold: {high_th:.4f} (top {high_ratio * 100:.0f}%), samples: {len(high_idx)}")

    X_aug = []
    y_aug = []

    def augment_group(indices):
        if len(indices) == 0:
            return
        group_X = X[indices]
        if len(group_X) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(k, len(group_X)))
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
print("TiFe Phase Dataset - SVR Grid Search (SMOTER 10% + 15 Features)")
print("=" * 80)

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED
)
print(f"Original training set: {len(X_train_base)}, Test set: {len(X_test)}")

# ==================== SMOTER Augmentation ====================
print("\nApplying SMOTER augmentation...")
X_train_aug, y_train_aug = bilateral_smoter_interpolate(
    X_train_base.values, y_train_base.values,
    low_ratio=EXTREME_LOW, high_ratio=EXTREME_HIGH, k=K_NEIGHBORS,
    n_copies=N_COPIES, noise_std=NOISE_STD, random_state=rng
)
print(f"Augmented training set: {len(X_train_aug)} (Added: {len(X_train_aug) - len(X_train_base)})")

# ==================== Feature Selection ====================
selector = SelectKBest(mutual_info_regression, k=N_FEATURES)
X_train_sel = selector.fit_transform(X_train_aug, y_train_aug)
X_test_sel = selector.transform(X_test.values)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"\nSelected {N_FEATURES} features:")
for i, f in enumerate(selected_features[:10], 1):
    print(f"  {i}. {f}")

# ==================== Grid Search ====================
param_grid = {
    'C': [7, 8, 9],
    'gamma': [0.008, 0.01, 0.0102, 0.012, 0.015],
    'epsilon': [0.04, 0.05, 0.057, 0.06, 0.07],
    'kernel': ['rbf']
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='r2', n_jobs=1, verbose=1)
grid_search.fit(X_train_sel, y_train_aug)

best_svr = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ==================== Evaluation ====================
y_train_pred = best_svr.predict(X_train_sel)
y_test_pred = best_svr.predict(X_test_sel)

train_r2 = r2_score(y_train_aug, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train_aug, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
overfit = train_r2 - test_r2

print("\n" + "=" * 80)
print("SVR Model Performance (After Grid Search)")
print("=" * 80)
print(f"Training: R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")
print(f"Test:     R²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
print(f"Overfit gap: {overfit:.4f}")

# ==================== Cross-Validation ====================
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(best_svr, X_train_sel, y_train_aug, cv=cv, scoring='r2')
print(f"\n5-fold CV R²: Mean={cv_scores.mean():.4f}, Std={cv_scores.std():.4f}")
