import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
RANDOM_STATE = 42
N_FEATURES = 18
TEST_SIZE = 0.2


# ==================== 1. Load Dataset ====================
file_path = r'C14_data.csv'
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
gbdt_all = GradientBoostingRegressor(random_state=RANDOM_STATE)
gbdt_all.fit(X_train, y_train)

importances = gbdt_all.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Exclude Initial_Hydrogen_Pressure_MPa
importance_df = importance_df[importance_df['feature'] != 'Initial_Hydrogen_Pressure_MPa']
importance_df = importance_df.sort_values('importance', ascending=False)

print(f"\nFeature importance ranking (Top 20, excluding Initial_Hydrogen_Pressure_MPa):")
for i, (idx, row) in enumerate(importance_df.head(20).iterrows(), 1):
    print(f"{i}. {row['feature']} : {row['importance']:.4f}")


# ==================== 3. Select Top N Features ====================
selected_features = importance_df.head(N_FEATURES)['feature'].tolist()
print(f"\nSelected top {N_FEATURES} features (excluding Initial_Hydrogen_Pressure_MPa):")
print(selected_features)

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# ==================== 4. Grid Search for Best GBDT Parameters ====================
param_grid = {
    'n_estimators': [150],
    'max_depth': [3],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'max_features': ['sqrt'],
    'min_samples_split': [5],
    'min_samples_leaf': [4]
}

base_gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE)

print("\n=== Starting Grid Search ===")
grid_search = GridSearchCV(
    estimator=base_gbdt,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_train_sel, y_train)

print("\n=== Best Parameters ===")
best_params = grid_search.best_params_
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Best CV MSE: {-grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_


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