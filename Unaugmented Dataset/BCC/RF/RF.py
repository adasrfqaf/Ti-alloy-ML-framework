"""
Random Forest Model for C14 Alloy Screening Hydrogen Storage Dataset
This script performs feature selection and trains a Random Forest regressor
using predefined best hyperparameters obtained from previous grid search.
No files are saved; results are printed to console only.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_STATE = 42
TOP_N_FEATURES = 18
TEST_SIZE = 0.2

# Best hyperparameters from previous grid search
BEST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 11,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 0.7,
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

# ==================== 1. Load Dataset ====================
df = pd.read_csv('BCC_data.csv', encoding='utf-8')
print("Dataset shape:", df.shape)

target = 'Max_H2_Uptake_wt_pct'
features = df.columns.drop(target)

# ==================== 2. Data Preprocessing ====================
# Drop rows with missing target
df = df.dropna(subset=[target])

# Handle missing values in features
for col in features:
    if df[col].isnull().any():
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical features
categorical_cols = df[features].select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== 3. Feature Importance Screening ====================
print("\nCalculating initial feature importance using Random Forest...")
rf_importance = RandomForestRegressor(
    n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
)
rf_importance.fit(X_train, y_train)

importances = rf_importance.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nTop 20 feature importance:")
print(importance_df.head(20))

selected_features = importance_df.head(TOP_N_FEATURES)['feature'].tolist()
print(f"\nSelected top {TOP_N_FEATURES} features:")
print(selected_features)

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# ==================== 4. Train Final Model with Best Parameters ====================
print("\n=== Training Final Model with Best Parameters ===")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")

best_rf = RandomForestRegressor(**BEST_PARAMS)
best_rf.fit(X_train_sel, y_train)

# ==================== 5. Model Evaluation ====================
# Predictions
y_train_pred = best_rf.predict(X_train_sel)
y_test_pred = best_rf.predict(X_test_sel)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
overfit_gap = train_r2 - test_r2

# Cross-validation R² on training set
cv_scores = cross_val_score(best_rf, X_train_sel, y_train, cv=5, scoring='r2')
cv_r2_mean = cv_scores.mean()
cv_r2_std = cv_scores.std()

# Print results
print("\n" + "="*60)
print("Model Evaluation Results")
print("="*60)
print(f"Training RMSE:  {train_rmse:.4f}")
print(f"Testing RMSE:   {test_rmse:.4f}")
print(f"Training MAE:   {train_mae:.4f}")
print(f"Testing MAE:    {test_mae:.4f}")
print(f"Training R2:    {train_r2:.4f}")
print(f"Testing R2:     {test_r2:.4f}")
print(f"Overfitting (Train-Test R2 gap): {overfit_gap:.4f}")

if overfit_gap > 0.05:
    print("  → Slight overfitting detected.")
elif overfit_gap < -0.05:
    print("  → Test set outperforms training set.")
else:
    print("  → Good generalization.")

print(f"CV R2 Mean (5-fold): {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
print("="*60)

print("\n✅ Evaluation complete. No files were saved.")