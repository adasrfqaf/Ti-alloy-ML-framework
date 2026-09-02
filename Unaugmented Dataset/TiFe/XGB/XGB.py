"""
XGBoost Model for TiFe Dataset (with predefined best hyperparameters)
Results are printed to console only; no files are saved.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FEATURES = 20


# ==================== 1. Load Data ====================
file_path = r'D:\python_file\PythonProject2\Unaugmented Dataset\TiFe\Model Screening\TiFe_data.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
print("Dataset shape:", df.shape)

target = 'Max_H2_Uptake_wt_pct'
df = df.dropna(subset=[target])

# Remove redundant additive columns (where additive column equals element column)
additive_cols = [col for col in df.columns if col.startswith('Additive_')]
cols_to_drop = []
for add_col in additive_cols:
    elem_col = add_col.replace('Additive_', 'Element_at_pct_')
    if elem_col in df.columns and df[add_col].equals(df[elem_col]):
        cols_to_drop.append(add_col)
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Removed redundant additive columns: {cols_to_drop}")

features = df.columns.drop(target)
X = df[features]
y = df[target]

print(f"Features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ==================== 2. Feature Importance Screening ====================
print("\nCalculating feature importance, selecting top 20...")
xgb_init = xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
xgb_init.fit(X_train, y_train)

importances = xgb_init.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
selected_features = importance_df.head(N_FEATURES)['feature'].tolist()
print(f"Selected {N_FEATURES} features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# ==================== 3. Train Model with Best Parameters ====================
params = {
    'n_estimators': 250,
    'learning_rate': 0.015,
    'max_depth': 5,
    'min_child_weight': 5,
    'subsample': 0.78,
    'colsample_bytree': 0.73,
    'reg_alpha': 0.12,
    'reg_lambda': 1.0,
    'gamma': 0.08,
    'random_state': RANDOM_STATE,
    'verbosity': 0,
    'n_jobs': 1,
    'early_stopping_rounds': 10,
    'eval_metric': 'rmse'
}

print("\nBest Parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

model = xgb.XGBRegressor(**params)
model.fit(X_train_sel, y_train, eval_set=[(X_test_sel, y_test)], verbose=False)


# ==================== 4. Model Evaluation ====================
y_train_pred = model.predict(X_train_sel)
y_test_pred = model.predict(X_test_sel)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
gap = train_r2 - test_r2

print("\n" + "=" * 60)
print(f"Model Evaluation Results (Features={N_FEATURES})")
print("=" * 60)
print(f"Training RMSE:  {train_rmse:.4f}")
print(f"Testing RMSE:   {test_rmse:.4f}")
print(f"Training MAE:   {train_mae:.4f}")
print(f"Testing MAE:    {test_mae:.4f}")
print(f"Training R2:    {train_r2:.4f}")
print(f"Testing R2:     {test_r2:.4f}")
print(f"Overfitting:    {gap:.4f}")

if gap > 0.05:
    print("  → Slight overfitting detected.")
elif gap < -0.05:
    print("  → Test set outperforms training set.")
else:
    print("  → Good generalization.")

# Cross-validation (remove early_stopping params for CV)
cv_params = {k: v for k, v in params.items() if k not in ['early_stopping_rounds', 'eval_metric']}
cv_model = xgb.XGBRegressor(**cv_params)
cv_scores = cross_val_score(cv_model, X_train_sel, y_train, cv=5, scoring='r2')
cv_r2_mean = cv_scores.mean()
cv_r2_std = cv_scores.std()
print(f"CV R2 Mean:     {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
print("=" * 60)


# ==================== 5. Feature Importance ====================
final_importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': selected_features,
    'importance': final_importances
})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)
print("\nFeature importance (selected):")
print(feat_imp_df.to_string(index=False))

print("\n✅ Evaluation complete. No files were saved.")