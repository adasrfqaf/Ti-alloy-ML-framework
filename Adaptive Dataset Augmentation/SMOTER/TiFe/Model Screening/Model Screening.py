"""
Supplementary Code for TiFe Phase - Model Selection with SMOTER Augmentation

This script evaluates multiple models with SMOTER augmentation for the TiFe phase dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings

warnings.filterwarnings('ignore')

# ====================== Fixed Random Seed ======================
SEED = 49
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

# ====================== User Parameters ======================
EXTREME_PERCENTILES = [5, 10, 20]
N_COPIES = 1
K_NEIGHBORS = 5
NOISE_STD = 0.05

FEATURE_COUNTS = [15, 18, 20]

MODELS = {
    'SVR': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
    'SVR': XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0, n_jobs=1),
    'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    'LGBM': LGBMRegressor(n_estimators=100, random_state=SEED, verbose=-1, n_jobs=1),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=SEED,
                        early_stopping=True, verbose=False),
    'SVR': SVR(kernel='rbf', C=5, epsilon=0.1)
}


def smoter_extreme_oversample(X_train, y_train, extreme_percentile, n_copies=5,
                              k_neighbors=5, noise_std=0.05, random_state=None):
    """
    Oversample extreme low and high value samples using SMOTER-style interpolation.
    """
    if random_state is None:
        random_state = np.random.RandomState(SEED)
    X = np.array(X_train)
    y = np.array(y_train).flatten()

    low_thresh = np.percentile(y, extreme_percentile)
    high_thresh = np.percentile(y, 100 - extreme_percentile)

    low_idx = np.where(y <= low_thresh)[0]
    high_idx = np.where(y >= high_thresh)[0]

    print(f"    Low threshold: {low_thresh:.4f} (bottom {extreme_percentile}%), samples: {len(low_idx)}")
    print(f"    High threshold: {high_thresh:.4f} (top {extreme_percentile}%), samples: {len(high_idx)}")

    X_aug = []
    y_aug = []

    def augment_group(indices):
        if len(indices) == 0:
            return
        group_X = X[indices]
        if len(group_X) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(group_X)))
            nbrs.fit(group_X)
        for i, idx in enumerate(indices):
            for _ in range(n_copies):
                if len(group_X) > 1:
                    pos = np.where(indices == idx)[0][0]
                    distances, neigh_positions = nbrs.kneighbors([group_X[pos]])
                    candidates = [p for p in neigh_positions[0] if p != pos]
                    if len(candidates) == 0:
                        neigh_pos = pos
                    else:
                        neigh_pos = random_state.choice(candidates)
                    neighbor_idx = indices[neigh_pos]
                    gap = random_state.random()
                    new_X = X[idx] + gap * (X[neighbor_idx] - X[idx])
                    new_y = y[idx] + gap * (y[neighbor_idx] - y[idx])
                else:
                    new_X = X[idx].copy()
                    new_y = y[idx]
                new_X += random_state.normal(0, noise_std, len(new_X))
                new_y += random_state.normal(0, noise_std * y.std())
                X_aug.append(new_X)
                y_aug.append(new_y)

    augment_group(low_idx)
    augment_group(high_idx)

    if len(X_aug) > 0:
        X_aug = np.vstack([X, np.array(X_aug)])
        y_aug = np.concatenate([y, np.array(y_aug)])
    else:
        X_aug, y_aug = X, y
    return X_aug, y_aug


# ====================== Load Data ====================
df = pd.read_csv('TiFe_data.csv', encoding='utf-8-sig')
print("=" * 80)
print("TiFe Phase Dataset - Model Selection (Original vs SMOTER Augmentation)")
print("=" * 80)
print(f"Original data shape: {df.shape}")

target_col = 'Max_H2_Uptake_wt_pct'
feature_cols = [col for col in df.columns if col != target_col]

for col in feature_cols:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
    elif df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols].fillna(0)
y = df[target_col].fillna(0)

print(f"Number of features: {X.shape[1]}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED
)
print(f"\nOriginal training set: {len(X_train_base)}")
print(f"Test set: {len(X_test)}")

all_results = []

# ====================== 1. Baseline: Original Data ======================
print("\n" + "=" * 80)
print("[Baseline: Original Data (No Augmentation)]")
print("=" * 80)
X_curr = X_train_base.values
y_curr = y_train_base.values

for n_feat in FEATURE_COUNTS:
    if n_feat > X_curr.shape[1]:
        continue
    selector = SelectKBest(mutual_info_regression, k=n_feat)
    X_train_sel = selector.fit_transform(X_curr, y_curr)
    X_test_sel = selector.transform(X_test.values)

    for name, model in MODELS.items():
        try:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_sel, y_curr)
            y_train_pred = model_clone.predict(X_train_sel)
            y_test_pred = model_clone.predict(X_test_sel)

            train_r2 = r2_score(y_curr, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            overfit = train_r2 - test_r2

            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_r2 = cross_val_score(model_clone, X_train_sel, y_curr, cv=cv, scoring='r2')

            all_results.append({
                'Data_Mode': 'Original',
                'Features': n_feat,
                'Model': name,
                'test_R2': test_r2,
                'overfit': overfit,
                'cv_R2_mean': cv_r2.mean(),
                'cv_R2_std': cv_r2.std()
            })
        except Exception as e:
            pass

# ====================== 2. SMOTER Augmentation ======================
print("\n" + "=" * 80)
print("[SMOTER Augmentation Experiments]")
print("=" * 80)

for pct in EXTREME_PERCENTILES:
    print(f"\nExtreme Percentile: {pct}% (both tails)")

    X_train_aug, y_train_aug = smoter_extreme_oversample(
        X_train_base.values, y_train_base.values,
        extreme_percentile=pct, n_copies=N_COPIES,
        k_neighbors=K_NEIGHBORS, noise_std=NOISE_STD,
        random_state=rng
    )
    print(f"Augmented training set: {len(X_train_aug)}")

    for n_feat in FEATURE_COUNTS:
        if n_feat > X_train_aug.shape[1]:
            continue
        selector = SelectKBest(mutual_info_regression, k=n_feat)
        X_train_sel = selector.fit_transform(X_train_aug, y_train_aug)
        X_test_sel = selector.transform(X_test.values)

        for name, model in MODELS.items():
            try:
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_sel, y_train_aug)
                y_train_pred = model_clone.predict(X_train_sel)
                y_test_pred = model_clone.predict(X_test_sel)

                train_r2 = r2_score(y_train_aug, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                overfit = train_r2 - test_r2

                cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
                cv_r2 = cross_val_score(model_clone, X_train_sel, y_train_aug, cv=cv, scoring='r2')

                all_results.append({
                    'Data_Mode': f'{pct}% tails',
                    'Features': n_feat,
                    'Model': name,
                    'test_R2': test_r2,
                    'overfit': overfit,
                    'cv_R2_mean': cv_r2.mean(),
                    'cv_R2_std': cv_r2.std()
                })
            except Exception as e:
                pass

# ====================== Results Summary ======================
results_df = pd.DataFrame(all_results)

# Get best result per model (by CV R²)
best_per_model = results_df.loc[results_df.groupby('Model')['cv_R2_mean'].idxmax()]
best_per_model = best_per_model.sort_values('cv_R2_mean', ascending=False)

print("\n" + "=" * 80)
print("Top 3 Models (Best Configuration per Model, by CV R²)")
print("=" * 80)

# Select top 3 models
top3_models = best_per_model.head(3)

for idx, row in top3_models.iterrows():
    print(f"\n{'─' * 60}")
    print(f"Model: {row['Model']}")
    print(f"{'─' * 60}")
    print(f"  Data Mode:   {row['Data_Mode']}")
    print(f"  Features:    {int(row['Features'])}")
    print(f"  CV R²:       {row['cv_R2_mean']:.4f} ± {row['cv_R2_std']:.4f}")
    print(f"  Test R²:     {row['test_R2']:.4f}")
    print(f"  Overfit:     {row['overfit']:.4f}")

# Save results
results_df.to_csv('TiFe_model_selection_results.csv', index=False)
best_per_model.to_csv('TiFe_best_per_model.csv', index=False)

print("\n" + "=" * 80)
print("Results saved to:")
print("  - TiFe_model_selection_results.csv (all results)")
print("  - TiFe_best_per_model.csv (best config per model)")