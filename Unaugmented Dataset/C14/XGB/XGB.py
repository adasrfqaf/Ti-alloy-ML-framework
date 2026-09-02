"""
XGBoost Model for C14 Hydrogen Storage Dataset
This script performs feature selection (excluding Initial_Hydrogen_Pressure_MPa)
and trains a model with predefined best hyperparameters.
No files are saved; results are printed to console only.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_STATE = 42
N_FEATURES = 15
TEST_SIZE = 0.2

# ==================== 1. Load Dataset ====================
file_path = 'C14_data.csv'   # Ensure this file is in the same directory
df = pd.read_csv(file_path, encoding='utf-8-sig')
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

target = 'Max_H2_Uptake_wt_pct'
X = df.drop(columns=[target])
y = df[target]

print(f"\nTarget variable: {target}")
print(f"Number of features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ==================== 2. Feature Importance (Excluding Initial_Hydrogen_Pressure_MPa) ====================
print("\n=== Calculating full feature importance (excluding Initial_Hydrogen_Pressure_MPa) ===")
xgb_all = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=0)
xgb_all.fit(X_train, y_train)

importances = xgb_all.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Exclude Initial_Hydrogen_Pressure_MPa
importance_df = importance_df[importance_df['feature'] != 'Initial_Hydrogen_Pressure_MPa']
importance_df = importance_df.sort_values('importance', ascending=False)

print(f"\nFeature importance ranking (Top 20, excluding Initial_Hydrogen_Pressure_MPa):")
for i, row in importance_df.head(20).iterrows():
    print(f"{list(importance_df.index).index(i)+1}. {row['feature']} : {row['importance']:.4f}")

# ==================== 3. Select Top N Features ====================
selected_features = importance_df.head(N_FEATURES)['feature'].tolist()
print(f"\nSelected top {N_FEATURES} features (excluding Initial_Hydrogen_Pressure_MPa):")
print(selected_features)

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# ==================== 4. Train Final Model with Best Parameters ====================
best_params = {
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.7,
    'reg_lambda': 2.5,
    'min_child_weight': 6,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'verbosity': 0
}

print("\n=== Best Parameters ===")
for k, v in best_params.items():
    print(f"  {k}: {v}")

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train_sel, y_train)

# ==================== 5. Model Evaluation ====================
y_train_pred = final_model.predict(X_train_sel)
y_test_pred = final_model.predict(X_test_sel)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
overfit_gap = train_r2 - test_r2

print("\n" + "="*60)
print(f"Model Evaluation Results (Features={N_FEATURES})")
print("="*60)
print(f"Training RMSE:  {train_rmse:.4f}")
print(f"Testing RMSE:   {test_rmse:.4f}")
print(f"Training MAE:   {train_mae:.4f}")
print(f"Testing MAE:    {test_mae:.4f}")
print(f"Training R2:    {train_r2:.4f}")
print(f"Testing R2:     {test_r2:.4f}")
print(f"Overfitting:    {overfit_gap:.4f}")

if overfit_gap > 0.05:
    print("  → Slight overfitting detected.")
elif overfit_gap < -0.05:
    print("  → Test set outperforms training set.")
else:
    print("  → Good generalization.")
print("="*60)

print("\n✅ Evaluation complete. No files were saved.")