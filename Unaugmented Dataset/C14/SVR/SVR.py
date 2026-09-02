import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_STATE = 42
N_FEATURES = 30  # Can be set to 38, 30, 25, 20, 18, 15, etc.
USE_RBF = False  # False = linear kernel (performs better based on previous results)


# ==================== 1. Load Dataset ====================
file_path = r'D:\python_file\PythonProject2\Unaugmented Dataset\C14\Model Screening\C14_data.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
print("Dataset shape:", df.shape)

target = 'Max_H2_Uptake_wt_pct'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


# ==================== 2. Feature Selection ====================
if N_FEATURES < len(X.columns):
    print(f"\nTraining linear SVR to obtain feature importance...")
    linear_base = SVR(kernel='linear')
    scaler_temp = StandardScaler()
    X_train_temp = scaler_temp.fit_transform(X_train)
    linear_base.fit(X_train_temp, y_train)
    coef = linear_base.coef_.flatten()
    coef_df = pd.DataFrame({'feature': X.columns, 'coef_abs': np.abs(coef)})
    coef_df = coef_df.sort_values('coef_abs', ascending=False)
    selected_features = coef_df.head(N_FEATURES)['feature'].tolist()
    print(f"Selected top {N_FEATURES} features: {selected_features}")
else:
    selected_features = list(X.columns)
    print("Using all features.")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# ==================== 3. Standardization ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)


# ==================== 4. Hyperparameter Tuning ====================
kernel = 'rbf' if USE_RBF else 'linear'

if kernel == 'rbf':
    param_grid = {
        'C': [0.5, 1, 2, 5, 10, 20, 50],
        'gamma': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
        'epsilon': [0.03, 0.05, 0.07, 0.1]
    }
else:
    param_grid = {
        'C': [0.5, 1, 2, 5, 10, 20],
        'epsilon': [0.03, 0.05, 0.07, 0.1]
    }

svr = SVR(kernel=kernel)
grid = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_
print(f"\n=== Best Parameters ({kernel} kernel) ===")
print(best_params)


# ==================== 5. Cross-Validation Details ====================
print("\n" + "="*60)
print("5-Fold Cross-Validation Details (Best Model)")
print("="*60)

cv_neg_mse_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error")
cv_mse_scores = -cv_neg_mse_scores
cv_rmse_scores = np.sqrt(cv_mse_scores)

for fold_idx in range(len(cv_mse_scores)):
    print(f"Fold {fold_idx+1} | MSE: {cv_mse_scores[fold_idx]:.4f} | RMSE: {cv_rmse_scores[fold_idx]:.4f}")

cv_mean_mse = np.mean(cv_mse_scores)
cv_std_mse = np.std(cv_mse_scores)
cv_mean_rmse = np.mean(cv_rmse_scores)
cv_std_rmse = np.std(cv_rmse_scores)

print(f"\nCV MSE Mean: {cv_mean_mse:.4f} (± {cv_std_mse:.4f})")
print(f"CV RMSE Mean: {cv_mean_rmse:.4f} (± {cv_std_rmse:.4f})")

cv_r2_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="r2")
for fold_idx in range(len(cv_r2_scores)):
    print(f"Fold {fold_idx+1} R²: {cv_r2_scores[fold_idx]:.4f}")

cv_mean_r2 = np.mean(cv_r2_scores)
cv_std_r2 = np.std(cv_r2_scores)
print(f"CV R² Mean: {cv_mean_r2:.4f} (± {cv_std_r2:.4f})")
print("="*60)


# ==================== 6. Model Evaluation ====================
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
overfit_gap = train_r2 - test_r2

print("\n" + "="*60)
print(f"Model Evaluation Results (Features={N_FEATURES}, {kernel} kernel)")
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

print(f"CV R2 Mean:     {cv_mean_r2:.4f} (± {cv_std_r2:.4f})")
print("="*60)

# ==================== 7. Save Predictions in Uploaded Format ====================
# Combine true and predicted values for test set into a DataFrame
# The output format matches the uploaded file: true_capacity, ours_pred
output_df = pd.DataFrame({
    'true_capacity': y_test.values,
    'ours_pred': y_test_pred
})

# Save as CSV (you can change the filename as needed)
output_file = 'test_predictions_ours_C14.csv'
output_df.to_csv(output_file, index=False)

print(f"\n✅ Predictions saved to {output_file}")
print("   Format: true_capacity, ours_pred")
print("   (Contains test set predictions only)")