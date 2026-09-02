import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_STATE = 42
N_JOBS = 1

# ==================== 1. Load Dataset ====================
df = pd.read_csv('C14_data.csv', encoding='utf-8-sig')
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Target variable
target = 'Max_H2_Uptake_wt_pct'
features = df.columns.drop(target)

# ==================== 2. Data Preprocessing ====================
df = df.dropna(subset=[target])

# Convert boolean/int columns to numeric
for col in features:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    if df[col].isnull().any():
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

X = df[features]
y = df[target]

print(f"\nFeatures: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Standardization (SVR and MLP are sensitive to feature scales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ==================== 3. Define Models ====================
models = {
    'XGBoost': XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=0
    ),
    'GBDT': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    ),
    'SVR': SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=-1
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# ==================== 4. Cross-Validation Evaluation ====================
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

print("\n" + "=" * 70)
print("Model Screening (based on 5-fold CV R² Score)")
print("=" * 70)

for name, model in models.items():
    try:
        # Use scaled data for SVR and MLP; original data for tree-based models
        if name in ['SVR', 'MLP']:
            X_used = X_scaled
        else:
            X_used = X

        scores = cross_val_score(model, X_used, y, cv=kf, scoring='r2', n_jobs=N_JOBS)

        results[name] = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'scores': scores
        }

        print(f"\n{name}:")
        print(f"  Mean R²: {scores.mean():.4f} (± {scores.std():.4f})")
        print(f"  Fold scores: {np.array2string(scores, precision=4, floatmode='fixed')}")

    except Exception as e:
        print(f"\n{name}: Training failed - {str(e)}")
        results[name] = {
            'mean_r2': -np.inf,
            'std_r2': np.nan,
            'scores': np.array([np.nan] * 5),
            'error': str(e)
        }

# ==================== 5. Sort and Select Top 3 Models ====================
sorted_models = sorted(
    [(name, results[name]['mean_r2'], results[name]['std_r2'])
     for name in results.keys()],
    key=lambda x: x[1],
    reverse=True
)

valid_models = [item for item in sorted_models if not np.isinf(item[1])]

print("\n" + "=" * 70)
if valid_models:
    top3 = valid_models[:3]
    print("🏆 Top 3 Best Performing Models")
    print("=" * 70)
    for rank, (name, mean_r2, std_r2) in enumerate(top3, 1):
        print(f"\n{rank}. {name}")
        print(f"   CV R² Mean: {mean_r2:.4f} (± {std_r2:.4f})")
else:
    print("❌ No models trained successfully. Check data quality.")
    print("=" * 70)

# ==================== 6. Save Results ====================
result_df = pd.DataFrame([
    {
        'Model': name,
        'CV_R2_Mean': results[name]['mean_r2'],
        'CV_R2_Std': results[name]['std_r2'],
        'Rank': sorted_models.index((name, results[name]['mean_r2'], results[name]['std_r2'])) + 1
    }
    for name in results.keys()
])
result_df = result_df.sort_values('CV_R2_Mean', ascending=False).reset_index(drop=True)
result_df.to_csv('C14_model_selection_results.csv', index=False, encoding='utf-8-sig')
print("\nFull results saved to: C14_model_selection_results.csv")

if valid_models:
    top3_names = [name for name, _, _ in top3]
    with open('C14_top3_models.txt', 'w', encoding='utf-8') as f:
        for name in top3_names:
            f.write(f"{name}\n")
    print("Top 3 model names saved to: C14_top3_models.txt")

# ==================== 7. Detailed Output ====================
print("\n" + "=" * 70)
print("Complete Ranking")
print("=" * 70)
print(result_df.to_string(index=False))

# Print CV fold details for debugging
print("\n" + "=" * 70)
print("CV R² Details for Each Model")
print("=" * 70)
for name in results.keys():
    if not np.isinf(results[name]['mean_r2']):
        print(f"{name}: {np.array2string(results[name]['scores'], precision=4, floatmode='fixed')}")