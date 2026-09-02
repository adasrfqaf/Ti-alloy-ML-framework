import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# 1. Load data from the new English CSV file
file_path = "mixed_alloy_data_english.csv"   # adjust path if needed
df = pd.read_csv(file_path)

target = "Max_H2_Uptake_wt_pct"
exclude = [target, "Main_Phase"]   # Exclude target and main phase column
features = [c for c in df.columns if c not in exclude]

# Convert boolean columns to integers (if any)
for c in features:
    if df[c].dtype == bool:
        df[c] = df[c].astype(int)

# No main-phase encoding is added – the model receives no phase information
X = df[features].copy()
y = df[target].copy()
phase = df["Main_Phase"].copy()   # Only used for per‑phase evaluation

# Handle missing values (if any) by median imputation
if X.isnull().any().any():
    X = X.fillna(X.median())
valid = ~y.isnull()
X, y, phase = X[valid], y[valid], phase[valid]

# Manually selected top‑20 features (English column names)
top20_features = [
    'Element_at_pct_V', 'Additive_Fe', 'Additive_Al', 'Element_at_pct_C',
    'Element_at_pct_Cr', 'Element_at_pct_Li', 'Element_at_pct_Fe',
    'Element_at_pct_Mn', 'Hydrogen_Absorption_Cycles', 'Element_at_pct_Al',
    'Element_at_pct_Nb', 'Element_at_pct_Zr', 'Test_Temperature_K',
    'Lattice_Parameter_a_A', 'Second_Phase_Al', 'Initial_Hydrogen_Pressure_MPa',
    'Element_at_pct_H', 'Element_at_pct_Ti', 'Unit_Cell_Volume_A3',
    'Additive_Zr'   # additional feature to make exactly 20
]
# Keep only those that actually exist in the DataFrame
top20_features = [f for f in top20_features if f in X.columns]
X = X[top20_features]

# Random split (no stratification) to avoid rare phase classes with only 1 sample
X_train, X_test, y_train, y_test, phase_train, phase_test = train_test_split(
    X, y, phase, test_size=0.2, random_state=42
)

# 2. Simplified baseline XGBoost parameters (reduced capacity, weak regularization)
params = {
    'max_depth': 3,            # reduced from 5 to lower model capacity
    'learning_rate': 0.1,
    'n_estimators': 50,        # reduced from 150 to encourage underfitting
    'subsample': 1.0,          # no row subsampling
    'colsample_bytree': 1.0,   # no column subsampling
    'reg_alpha': 0,            # no L1 regularization
    'reg_lambda': 0,           # no L2 regularization (was 1)
    'random_state': 42,
    'verbosity': 0
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# 3. Evaluation function
def evaluate(y_true, y_pred, name=""):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:30} R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n--- Overall Evaluation ---")
train_metrics = evaluate(y_train, y_train_pred, "Training set")
test_metrics = evaluate(y_test, y_test_pred, "Test set")

print("\n--- Per‑phase Test Set Evaluation ---")
for phase_name in phase_test.unique():
    mask = (phase_test == phase_name)
    if mask.sum() >= 2:
        y_phase = y_test[mask]
        y_pred_phase = y_test_pred[mask]
        evaluate(y_phase, y_pred_phase, f"Test-{phase_name}")

print("\n========== Result of Modified Mixed Training (no phase encoding + simplified model) ==========")
print(f"Training R² = {train_metrics['R2']:.4f}")
print(f"Test R²     = {test_metrics['R2']:.4f}")
print(f"Overfitting gap = {train_metrics['R2'] - test_metrics['R2']:.4f}")