"""
SVR Model Training for TiFe Dataset with Feature Count Comparison (15, 18, 20)
No files are saved; results are printed to console.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
DATA_FILE = r'D:\python_file\PythonProject2\Unaugmented Dataset\TiFe\Model Screening\TiFe_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 8
KERNEL = 'linear'
N_JOBS = 1

# Feature numbers to compare
FEATURE_NUMS = [18]

# Hyperparameter grid
param_grid = {
    'C': [1.2],
    'epsilon': [0.20]
}


# ==================== Load and Preprocess Data ====================
print(f"Loading data: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
print("Dataset shape:", df.shape)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Remove zero-variance columns (all zeros)
zero_cols = X.columns[(X == 0).all()]
if len(zero_cols) > 0:
    print(f"Removed zero-only columns: {list(zero_cols)}")
    X = X.drop(columns=zero_cols)

# Convert any object columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
X = X.fillna(0)

print(f"Original features: {X.shape[1]}, samples: {X.shape[0]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Fixed split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ==================== Results Storage ====================
results = []


# ==================== Modeling for Each Feature Count ====================
for n_feat in FEATURE_NUMS:
    print(f"\n{'='*60}")
    print(f"Features = {n_feat}")
    print(f"{'='*60}")

    # Feature selection based on full-feature linear SVR coefficients
    scaler_temp = StandardScaler()
    X_train_temp = scaler_temp.fit_transform(X_train)
    selector = SVR(kernel='linear')
    selector.fit(X_train_temp, y_train)
    coef_abs = np.abs(selector.coef_.flatten())
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': coef_abs})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    selected_features = feat_imp.head(n_feat)['feature'].tolist()
    print(f"Selected {n_feat} features (first 5): {selected_features[:5]}")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # Grid search
    print(f"Grid: C={param_grid['C']}, epsilon={param_grid['epsilon']}")
    svr = SVR(kernel=KERNEL)
    grid = GridSearchCV(
        svr, param_grid, cv=CV_FOLDS, scoring='neg_mean_squared_error',
        verbose=1, n_jobs=N_JOBS
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(f"Best parameters: {best_params}")

    # Compute CV R² on the best model using the same folds
    cv_r2_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=CV_FOLDS, scoring='r2')
    cv_r2_mean = cv_r2_scores.mean()
    cv_r2_std = cv_r2_scores.std()
    print(f"CV R²: {cv_r2_mean:.4f} (± {cv_r2_std:.4f})")

    # Evaluation on train and test
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    overfit_gap = train_r2 - test_r2

    print(f"Training: R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
    print(f"Testing:  R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
    print(f"Overfitting gap: {overfit_gap:.4f}")

    # Store results
    results.append({
        'n_features': n_feat,
        'best_C': best_params['C'],
        'best_epsilon': best_params['epsilon'],
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'train_R2': train_r2,
        'test_R2': test_r2,
        'train_RMSE': train_rmse,
        'test_RMSE': test_rmse,
        'train_MAE': train_mae,
        'test_MAE': test_mae,
        'overfit_gap': overfit_gap,
        'selected_features': selected_features
    })


# ==================== Results Summary ====================
results_df = pd.DataFrame(results)
print("\n\n" + "="*60)
print("Final Comparison Results (with CV R²)")
print("="*60)
print(results_df[['n_features', 'best_C', 'best_epsilon', 'cv_r2_mean', 'cv_r2_std',
                  'train_R2', 'test_R2', 'train_RMSE', 'test_RMSE',
                  'train_MAE', 'test_MAE', 'overfit_gap']].to_string(index=False))

# Find best feature count based on test R2
best_row = results_df.loc[results_df['test_R2'].idxmax()]
print(f"\nRecommended feature count: {best_row['n_features']}, Test R2={best_row['test_R2']:.4f}, CV R2={best_row['cv_r2_mean']:.4f} (± {best_row['cv_r2_std']:.4f}), Overfitting gap={best_row['overfit_gap']:.4f}")

print("\n✅ Evaluation complete. No files were saved.")