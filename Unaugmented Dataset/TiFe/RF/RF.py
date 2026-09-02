# -*- coding: utf-8 -*-
"""
Random Forest Model Training (with temperature column retained, fine-tuned regularization, 15 features)
Goal: Maintain test R2 ≈ 0.68, reduce train/test gap
Results are printed to console only; no files are saved.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
FILE_PATH = r'D:\python_file\PythonProject2\Unaugmented Dataset\TiFe\Model Screening\TiFe_data.csv'
N_FEATURES = 15
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


print("=" * 60)
print(f"Random Forest Fine-Tuning (Features={N_FEATURES})")
print("=" * 60)


# ==================== 1. Load Data ====================
df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')
print("Dataset shape:", df.shape)

target = 'Max_H2_Uptake_wt_pct'
X = df.drop(columns=[target])
y = df[target]

# Remove additive columns (columns starting with 'Additive_')
additive_cols = [col for col in X.columns if col.startswith('Additive_')]
if additive_cols:
    X = X.drop(columns=additive_cols)
    print(f"Removed additive columns: {len(additive_cols)} columns")
print(f"Current total features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")


# ==================== 2. Train-Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# ==================== 3. Feature Selection ====================
print("\nCalculating full-feature Random Forest importance...")
rf_all = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
rf_all.fit(X_train, y_train)

importances = rf_all.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("\nFull feature importance ranking (Top 20):")
for i, idx in enumerate(indices[:20]):
    print(f"{i+1:2d}. {feature_names[idx]:<30} : {importances[idx]:.6f}")

selected_features = [feature_names[idx] for idx in indices[:N_FEATURES]]
print(f"\nSelected top {N_FEATURES} features:")
for f in selected_features:
    print(f"  - {f}")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# ==================== 4. Hyperparameter Tuning ====================
param_grid = {
    'n_estimators': [100],
    'max_depth': [8],
    'min_samples_split': [6],
    'min_samples_leaf': [2],
    'max_features': ['sqrt'],
    'max_samples': [0.95]
}

print("\nStarting Grid Search (fine-tuning regularization)...")
rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_train_sel, y_train)

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")


# ==================== 4b. Compute CV R² ====================
cv_r2_scores = cross_val_score(best_rf, X_train_sel, y_train, cv=CV_FOLDS, scoring='r2')
cv_r2_mean = cv_r2_scores.mean()
cv_r2_std = cv_r2_scores.std()
print(f"\nCV R² Mean: {cv_r2_mean:.4f} (± {cv_r2_std:.4f})")


# ==================== 5. Model Evaluation ====================
y_train_pred = best_rf.predict(X_train_sel)
y_test_pred = best_rf.predict(X_test_sel)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

gap = train_r2 - test_r2

print("\n" + "=" * 60)
print("Model Performance Comparison (Train vs Test)")
print("=" * 60)
print(f"{'Metric':<10} {'Train':<15} {'Test':<15}")
print("-" * 40)
print(f"{'R2':<10} {train_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'RMSE':<10} {train_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<10} {train_mae:<15.4f} {test_mae:<15.4f}")
print("=" * 60)
print(f"Train-Test R2 Gap: {gap:.4f}  {'(Overfitting risk)' if gap > 0.1 else '(Good generalization)'}")


# ==================== 6. Feature Importance ====================
final_importances = best_rf.feature_importances_
final_indices = np.argsort(final_importances)[::-1]

print("\nSelected feature importance ranking (high to low):")
for i, idx in enumerate(final_indices):
    print(f"{i+1:2d}. {selected_features[idx]:<30} : {final_importances[idx]:.6f}")

print("\n✅ Evaluation complete. No files were saved.")