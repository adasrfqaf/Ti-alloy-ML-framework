"""
XGBoost Fine-Tuning on Best Configuration (SMOTER 20%, 15 features)
Single-stage local grid search around best parameters.
Outputs train/test metrics, overfitting, and CV R².
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ====================== Configuration ======================
# Best configuration from previous search
SMOTER_RATIO = 0.20
N_FEATURES = 15
CV_FOLDS = 3

# Best hyperparameters from previous search (before fine-tuning)
BEST_INITIAL_PARAMS = {
    'gamma': 0.0,
    'max_depth': 4,
    'min_child_weight': 3,
    'reg_alpha': 0.05,
    'reg_lambda': 0.8,
    'colsample_bytree': 0.7,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.85
}

# ====================== Fine-Tuning Grid (local) ======================
# Narrow ranges around best values
FINE_TUNE_GRID = {
    'learning_rate': [0.05],
    'n_estimators': [280],
    'subsample': [0.85],
    'colsample_bytree': [0.7],
    'max_depth': [4],
    'min_child_weight': [3],
    'gamma': [0.004],
    'reg_alpha': [0.4],
    'reg_lambda': [ 0.6]
}


def load_bcc_data():
    """Load C14 Alloy Screening dataset."""
    df = pd.read_csv('BCC_data.csv', encoding='utf-8-sig')
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
    X = df.select_dtypes(include=[np.number]).drop(columns=['Max_H2_Uptake_wt_pct'], errors='ignore')
    y = df['Max_H2_Uptake_wt_pct'].values
    X = X.loc[:, X.nunique() > 1]
    X.fillna(X.mean(), inplace=True)
    return X, y


def bilateral_smoter_interpolate(X_train, y_train, low_ratio=0.20, high_ratio=0.20, k=5):
    """SMOTER interpolation for both tails."""
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    low_th = np.percentile(y, 100 * low_ratio)
    high_th = np.percentile(y, 100 * (1 - high_ratio))
    minority_idx = np.where((y <= low_th) | (y >= high_th))[0]

    print(f"  Low: ≤ {low_th:.3f} (bottom {low_ratio*100:.0f}%), High: ≥ {high_th:.3f} (top {high_ratio*100:.0f}%)")
    print(f"  Selected samples: {len(minority_idx)}")

    if len(minority_idx) < 2:
        print("  Warning: Insufficient samples, returning original data")
        return X_train, y_train

    nbrs = NearestNeighbors(n_neighbors=min(k, len(minority_idx) - 1)).fit(X[minority_idx])
    syn_X, syn_y = [], []
    for i in minority_idx:
        _, neighbors = nbrs.kneighbors(X[i].reshape(1, -1))
        n = minority_idx[np.random.choice(neighbors[0][1:])]
        gap = np.random.rand()
        syn_X.append(X[i] + gap * (X[n] - X[i]))
        syn_y.append(y[i] + gap * (y[n] - y[i]))
    X_aug = np.vstack([X, syn_X])
    y_aug = np.hstack([y, syn_y])
    if isinstance(X_train, pd.DataFrame):
        X_aug = pd.DataFrame(X_aug, columns=X_train.columns)
    return X_aug, y_aug


def get_top_features(X_train, y_train, n_features):
    """Select top n_features using Random Forest importance."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:n_features]
    return X_train.columns[sorted_idx].tolist()


def main():
    # Load data
    X, y = load_bcc_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Apply SMOTER augmentation
    print("Applying SMOTER 20% augmentation...")
    X_train_aug, y_train_aug = bilateral_smoter_interpolate(X_train, y_train, low_ratio=SMOTER_RATIO, high_ratio=SMOTER_RATIO, k=5)
    print(f"Augmented training size: {len(X_train_aug)} ({len(X_train_aug)/len(X_train):.2f}x)")

    # Select top N features
    print(f"Selecting top {N_FEATURES} features...")
    selected_features = get_top_features(X_train_aug, y_train_aug, N_FEATURES)
    print("Selected features:", selected_features)
    X_train_sel = X_train_aug[selected_features]
    X_test_sel = X_test[selected_features]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # ---------------------- Fine-Tuning Grid Search ----------------------
    print("\n=== Fine-Tuning Grid Search (single-stage) ===")
    print(f"Grid size: {np.prod([len(v) for v in FINE_TUNE_GRID.values()])} combinations")

    base_model = XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0)  # n_jobs=1 to avoid encoding issues
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=FINE_TUNE_GRID,
        cv=CV_FOLDS,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train_aug)

    best_params = grid_search.best_params_
    best_cv_r2 = grid_search.best_score_

    # Train final model with best params
    final_model = XGBRegressor(**best_params, random_state=SEED, n_jobs=1, verbosity=0)
    final_model.fit(X_train_scaled, y_train_aug)

    # Predictions
    y_train_pred = final_model.predict(X_train_scaled)
    y_test_pred = final_model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train_aug, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_aug, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train_aug, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    overfit = train_r2 - test_r2

    # CV R² on final model (3-fold)
    cv_scores = cross_val_score(final_model, X_train_scaled, y_train_aug, cv=CV_FOLDS, scoring='r2')
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    # ---------------------- Output ----------------------
    print("\n" + "="*60)
    print("Final Fine-Tuned XGBoost Model - Performance Summary")
    print("="*60)
    print(f"SMOTER ratio: {SMOTER_RATIO*100:.0f}%")
    print(f"Number of features: {N_FEATURES}")
    print("\n[Best Hyperparameters (after fine-tuning)]")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\n[Evaluation Metrics]")
    print(f"  Train R²:    {train_r2:.4f}")
    print(f"  Test R²:     {test_r2:.4f}")
    print(f"  Train RMSE:  {train_rmse:.4f}")
    print(f"  Test RMSE:   {test_rmse:.4f}")
    print(f"  Train MAE:   {train_mae:.4f}")
    print(f"  Test MAE:    {test_mae:.4f}")
    print(f"  Overfitting (Train-Test R² gap): {overfit:.4f}")
    print(f"  CV R² Mean:  {cv_mean:.4f}  (3-fold CV)")
    print(f"  CV R² Std:   {cv_std:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()