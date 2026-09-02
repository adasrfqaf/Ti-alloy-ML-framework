"""
GBDT Fine-Tuning with Refined Grid for Best Configuration (TiFe Dataset)
Fixed: SMOTER 20%, 18 features.
Grid refined around current best hyperparameters.
No files saved; results printed to console.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

SEED = 420
np.random.seed(SEED)

# ====================== Fixed Configuration ======================
SMOTER_RATIO = 0.20
N_FEATURES = 18
K_NEIGHBORS = 5
CV_FOLDS = 5

# ====================== Refined Grid (around best parameters) ======================
PARAM_GRID = {
    'learning_rate': [0.067],
    'max_depth': [6],
    'max_features': [0.75],
    'min_samples_leaf': [6],
    'min_samples_split': [7],
    'n_estimators': [250],
    'subsample': [0.8]
}

# ====================== Utility Functions ======================
def load_tife_data():
    df = pd.read_csv('TiFe_data.csv', encoding='utf-8-sig')
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    target = 'Max_H2_Uptake_wt_pct'
    X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target].values
    X = X.loc[:, X.nunique() > 1]
    X = X.fillna(X.mean())
    return X, y


def bilateral_smoter_interpolate(X_train, y_train, ratio, k=5):
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train
    low_th = np.percentile(y, 100 * ratio)
    high_th = np.percentile(y, 100 * (1 - ratio))
    minority_idx = np.where((y <= low_th) | (y >= high_th))[0]
    if len(minority_idx) < 2:
        return X_train, y_train
    nbrs = NearestNeighbors(n_neighbors=min(k, len(minority_idx)-1)).fit(X[minority_idx])
    syn_X, syn_y = [], []
    for idx in minority_idx:
        _, indices = nbrs.kneighbors(X[idx].reshape(1, -1))
        neighbor_local_idx = np.random.choice(indices[0][1:], 1)[0]
        neighbor_global_idx = minority_idx[neighbor_local_idx]
        gap = np.random.uniform(0, 1)
        syn_X.append(X[idx] + gap * (X[neighbor_global_idx] - X[idx]))
        syn_y.append(y[idx] + gap * (y[neighbor_global_idx] - y[idx]))
    X_aug = np.vstack([X, syn_X])
    y_aug = np.concatenate([y, syn_y])
    if isinstance(X_train, pd.DataFrame):
        X_aug = pd.DataFrame(X_aug, columns=X_train.columns)
    return X_aug, y_aug


def select_top_features(X_train, y_train, n_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_gbdt = GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=SEED)
    temp_gbdt.fit(X_scaled, y_train)
    importances = temp_gbdt.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    return X_train.columns[indices].tolist()


# ====================== Main ======================
print("=" * 60)
print("GBDT Refined Grid Search (Best Configuration)")
print(f"SMOTER Ratio: {SMOTER_RATIO*100:.0f}%, Features: {N_FEATURES}")
print("=" * 60)

X, y = load_tife_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# SMOTER augmentation
X_train_aug, y_train_aug = bilateral_smoter_interpolate(X_train, y_train, SMOTER_RATIO, K_NEIGHBORS)
print(f"Augmented training size: {len(X_train_aug)} ({len(X_train_aug)/len(X_train):.2f}x)")

# Feature selection
selected_features = select_top_features(X_train_aug, y_train_aug, N_FEATURES)
print("Selected features:", selected_features)

X_train_sel = X_train_aug[selected_features]
X_test_sel = X_test[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# Grid search
print(f"\nGrid search over {np.prod([len(v) for v in PARAM_GRID.values()])} combinations...")
gbdt = GradientBoostingRegressor(random_state=SEED)
grid_search = GridSearchCV(
    gbdt, PARAM_GRID, cv=CV_FOLDS, scoring='r2',
    n_jobs=1, verbose=1
)
grid_search.fit(X_train_scaled, y_train_aug)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

# Evaluate
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

train_r2 = r2_score(y_train_aug, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train_aug, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
overfit_gap = train_r2 - test_r2

# CV R²
cv_outer = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train_aug, cv=cv_outer, scoring='r2')
cv_r2_mean = cv_scores.mean()
cv_r2_std = cv_scores.std()

print("\n" + "=" * 60)
print("Refined Grid Search Results")
print("=" * 60)
print("Best Parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

print("\nEvaluation Metrics:")
print(f"Train R²:     {train_r2:.4f}")
print(f"Test R²:      {test_r2:.4f}")
print(f"Train MAE:    {train_mae:.4f}")
print(f"Test MAE:     {test_mae:.4f}")
print(f"Train RMSE:   {train_rmse:.4f}")
print(f"Test RMSE:    {test_rmse:.4f}")
print(f"Overfit gap:  {overfit_gap:.4f}")
print(f"CV R² (5-fold): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
print("=" * 60)
print("\n✅ Refined grid search complete. No files saved.")