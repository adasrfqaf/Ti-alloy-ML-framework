"""
XGBoost Model for C14 Alloy Screening Hydrogen Storage Dataset
This script performs feature selection, hyperparameter tuning via RandomizedSearchCV,
and evaluation using XGBoost. No files are saved; all results are printed to console.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
N_TOP_FEATURES = 18
RANDOM_STATE = 42
N_JOBS = 1
N_ITER_RANDOM = 50
TEST_SIZE = 0.2


# ==================== 1. Load Dataset ====================
file_path = 'BCC_data.csv'   # Ensure this file is in the same directory as the script
df = pd.read_csv(file_path, encoding='utf-8')
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

target = 'Max_H2_Uptake_wt_pct'
features = df.columns.drop(target)


# ==================== 2. Data Preprocessing ====================
df = df.dropna(subset=[target])

for col in features:
    if df[col].isnull().any():
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

categorical_cols = df[features].select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ==================== 3. Feature Importance Screening ====================
print("\nCalculating initial feature importance...")
xgb_init = xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0, n_jobs=N_JOBS)
xgb_init.fit(X_train, y_train)

importances = xgb_init.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
print("\nTop 20 initial feature importance:")
print(importance_df.head(20))

selected_features = importance_df.head(N_TOP_FEATURES)['feature'].tolist()
print(f"\nSelected top {len(selected_features)} features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# ==================== 4. Hyperparameter Tuning with Randomized Search ====================
print("\n=== Starting Randomized Search ===")

param_dist = {
    'gamma': [0, 0.1, 0.5, 1],
    'reg_alpha': [1, 2, 3, 5],
    'reg_lambda': [2, 3, 5],
    'max_depth': [2, 3],
    'min_child_weight': [2, 3, 5],
    'subsample': [0.5, 0.6, 0.7],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [200, 300, 400]
}

base_model = xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=N_JOBS)
cv_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=N_ITER_RANDOM,
    cv=cv_inner,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE
)
random_search.fit(X_train_sel, y_train)

best_params = random_search.best_params_
print("\n=== Best Parameters ===")
for k, v in best_params.items():
    print(f"  {k}: {v}")


# ==================== 5. Train Final Model with Early Stopping ====================
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_sel, y_train, test_size=0.2, random_state=RANDOM_STATE
)

final_model = xgb.XGBRegressor(
    **best_params,
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=N_JOBS,
    early_stopping_rounds=20,
    eval_metric='rmse'
)

final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_train_pred = final_model.predict(X_train_sel)
y_test_pred = final_model.predict(X_test_sel)


# ==================== 6. Model Evaluation ====================
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

cv_model = deepcopy(final_model)
cv_model.set_params(early_stopping_rounds=None, eval_metric=None)
cv_scores = cross_val_score(cv_model, X_train_sel, y_train, cv=5, scoring='r2')
cv_r2_mean = cv_scores.mean()
cv_r2_std = cv_scores.std()

print("\n" + "="*60)
print("Model Evaluation Results")
print("="*60)
print(f"Training RMSE:  {train_rmse:.4f}")
print(f"Testing RMSE:   {test_rmse:.4f}")
print(f"Training MAE:   {train_mae:.4f}")
print(f"Testing MAE:    {test_mae:.4f}")
print(f"Training R2:    {train_r2:.4f}")
print(f"Testing R2:     {test_r2:.4f}")
print(f"Overfitting:    {train_r2 - test_r2:.4f}")
print(f"CV R2 Mean:     {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
print("="*60)

print("\n✅ Evaluation complete. No files were saved.")