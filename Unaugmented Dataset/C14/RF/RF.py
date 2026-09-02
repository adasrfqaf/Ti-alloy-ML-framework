import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_STATE = 42
N_FEATURES = 18  # Can be adjusted to 15, 18, 20, etc.


# ==================== 1. Load Dataset (Using English CSV) ====================
file_path = r'D:\python_file\PythonProject2\Unaugmented Dataset\C14\Model Screening\C14_data.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

target = 'Max_H2_Uptake_wt_pct'
X = df.drop(columns=[target])
y = df[target]

print(f"\nTarget variable: {target}")
print(f"Number of features: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Convert any boolean columns to int (if any remain as bool)
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


# ==================== 2. Train Full-Feature Model for Feature Importance ====================
print("\n=== Training full-feature model for feature importance ===")
rf_all = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
rf_all.fit(X_train, y_train)

importances = rf_all.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("\nFeature importance ranking (Top 20):")
for i, idx in enumerate(indices[:20]):
    print(f"{i+1}. {feature_names[idx]} : {importances[idx]:.4f}")


# ==================== 3. Select Top N Features ====================
selected_features = [feature_names[idx] for idx in indices[:N_FEATURES]]
print(f"\nSelected top {N_FEATURES} features:")
print(selected_features)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# ==================== 4. Hyperparameter Tuning with Regularization ====================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='neg_mean_squared_error',
    verbose=1, n_jobs=1
)
grid_search.fit(X_train_selected, y_train)

best_rf = grid_search.best_estimator_
print("\n=== Best Parameters ===")
print(grid_search.best_params_)


# ==================== 5. Model Evaluation ====================
y_train_pred = best_rf.predict(X_train_selected)
y_test_pred = best_rf.predict(X_test_selected)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
overfit_gap = train_r2 - test_r2

cv_scores = cross_val_score(best_rf, X_train_selected, y_train, cv=5, scoring='r2')
cv_r2_mean = cv_scores.mean()
cv_r2_std = cv_scores.std()

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

print(f"CV R2 Mean:     {cv_r2_mean:.4f} (± {cv_r2_std:.4f})")
print("="*60)

print("\n✅ Evaluation complete. No files were saved.")