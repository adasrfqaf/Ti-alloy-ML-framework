"""
GBDT Model for TiFe Dataset with Local Grid Search
No files are saved; results are printed to console only.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FEATURES = 20
CV_FOLDS = 5


# ==================== 1. Load Data ====================
file_path = r'TiFe_data.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
print("Dataset shape:", df.shape)

target = 'Max_H2_Uptake_wt_pct'
df = df.dropna(subset=[target])

# Remove redundant additive columns
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
gbdt_init = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
gbdt_init.fit(X_train, y_train)

importances = gbdt_init.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
selected_features = importance_df.head(N_FEATURES)['feature'].tolist()
print(f"Selected {N_FEATURES} features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# ==================== 3. Grid Search (local fine-tuning) ====================
# Local grid around the base values
param_grid = {
    'n_estimators': [150],
    'learning_rate': [0.05],
    'max_depth': [3],
    'min_samples_split': [5],
    'min_samples_leaf': [4],
    'max_features': ['sqrt'],
    'subsample': [0.8],
    # loss kept fixed to 'squared_error' (can be added if needed)
}

print("\nStarting Grid Search (local fine-tuning)...")
gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    estimator=gbdt,
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_train_sel, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nBest Parameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")


# ==================== 4. Model Evaluation ====================
y_train_pred = best_model.predict(X_train_sel)
y_test_pred = best_model.predict(X_test_sel)

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

# Compute CV R² on the best model (with the same folds)
cv_r2_scores = cross_val_score(best_model, X_train_sel, y_train, cv=CV_FOLDS, scoring='r2')
cv_r2_mean = cv_r2_scores.mean()
cv_r2_std = cv_r2_scores.std()
print(f"CV R2 Mean:     {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
print("=" * 60)


# ==================== 5. Feature Importance of Best Model ====================
final_importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': selected_features,
    'importance': final_importances
})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)
# ... 之前的特征重要性输出 ...

print("\nFeature importance (selected):")
print(feat_imp_df.to_string(index=False))

# ==================== 6. Save Predictions ====================
# Combine true and predicted values for test set into a DataFrame
# Output format: true_capacity, ours_pred (matches the uploaded file)
output_df = pd.DataFrame({
    'true_capacity': y_test.values,
    'ours_pred': y_test_pred
})

# Save to CSV (filename can be changed as needed)
output_file = 'test_predictions_ours_TiFe.csv'
output_df.to_csv(output_file, index=False)

print(f"\n✅ Predictions saved to {output_file}")
print("   Format: true_capacity, ours_pred")
print("   (Test set predictions only)")