"""
SVR Model Fine-Tuning with SMOTER Augmentation and Kernel Selection (C14 Alloy Screening Dataset)
Traverses SMOTER ratios (5%, 10%, 20%) and feature counts (15, 18, 20).
Grid search includes kernel='linear' and 'rbf' with regularized hyperparameters.
Outputs comprehensive evaluation metrics for the global best model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ====================== User Parameters ======================
SMOTER_RATIOS = [0.05, 0.10, 0.20]
FEATURE_COUNTS = [15, 18, 20]
CV_FOLDS = 3

# ====================== SVR Grid (includes kernel, regularized) ======================
PARAM_GRID = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 0.5, 1.0, 2.0],
    'gamma': ['scale', 0.005, 0.01, 0.05],
    'epsilon': [0.2, 0.3, 0.5]
}


def load_bcc_data():
    """Load C14 Alloy Screening dataset with English column names."""
    df = pd.read_csv('BCC_data.csv', encoding='utf-8-sig')
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
    X = df.select_dtypes(include=[np.number]).drop(columns=['Max_H2_Uptake_wt_pct'], errors='ignore')
    y = df['Max_H2_Uptake_wt_pct'].values
    X = X.loc[:, X.nunique() > 1]
    X.fillna(X.mean(), inplace=True)
    return X, y


def bilateral_smoter_interpolate(X_train, y_train, low_ratio=0.20, high_ratio=0.20, k=5):
    """SMOTER interpolation for both low and high target value tails."""
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
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:n_features]
    return X_train.columns[sorted_idx].tolist()


def evaluate_svr_config(X_train, y_train, X_test, y_test, param_grid, cv=3):
    """
    Perform grid search on SVR with given data and return best model and metrics.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = SVR(max_iter=5000)  # kernel will be chosen by grid
    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=1,
        verbose=0
    )
    gs.fit(X_train_scaled, y_train)

    best_model = gs.best_estimator_
    best_params = gs.best_params_
    # mean and std from cv_results_
    best_cv_mean = gs.best_score_
    best_cv_std = gs.cv_results_['std_test_score'][gs.best_index_]

    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    overfit = train_r2 - test_r2

    # Recompute CV R² on final model (consistent)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='r2')
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    return {
        'best_model': best_model,
        'best_params': best_params,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfit': overfit,
        'scaler': scaler
    }


def main():
    # Load and split data
    X, y = load_bcc_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    all_results = []

    for ratio in SMOTER_RATIOS:
        print(f"\n{'='*60}")
        print(f"Processing SMOTER ratio: {ratio*100:.0f}%")
        print('='*60)

        # Augment
        X_train_aug, y_train_aug = bilateral_smoter_interpolate(
            X_train, y_train, low_ratio=ratio, high_ratio=ratio, k=5
        )
        print(f"Augmented training size: {len(X_train_aug)} ({len(X_train_aug)/len(X_train):.2f}x)")

        # Compute feature importance on augmented set
        print("  Computing feature importance...")
        # Get all top features up to max count, then subset per n_feat
        max_features = max(FEATURE_COUNTS)
        top_features_all = get_top_features(X_train_aug, y_train_aug, max_features)

        for n_feat in FEATURE_COUNTS:
            print(f"\n--- Feature Count: {n_feat} ---")
            selected_features = top_features_all[:n_feat]
            X_train_sel = X_train_aug[selected_features]
            X_test_sel = X_test[selected_features]

            metrics = evaluate_svr_config(
                X_train_sel, y_train_aug, X_test_sel, y_test,
                param_grid=PARAM_GRID, cv=CV_FOLDS
            )

            print(f"  Test R²: {metrics['test_r2']:.4f}, CV R²: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

            all_results.append({
                'SMOTER': f"{ratio*100:.0f}%",
                'n_feat': n_feat,
                'cv_mean': metrics['cv_mean'],
                'cv_std': metrics['cv_std'],
                'test_r2': metrics['test_r2'],
                'overfit': metrics['overfit'],
                'train_r2': metrics['train_r2'],
                'train_rmse': metrics['train_rmse'],
                'test_rmse': metrics['test_rmse'],
                'train_mae': metrics['train_mae'],
                'test_mae': metrics['test_mae'],
                'best_params': metrics['best_params'],
                'selected_features': selected_features
            })

    # ====================== Summary ======================
    print("\n" + "="*80)
    print("Summary of All Configurations (sorted by Test R²)")
    print("="*80)

    df_summary = pd.DataFrame([
        {
            'SMOTER': r['SMOTER'],
            'n_feat': r['n_feat'],
            'CV_R2': r['cv_mean'],
            'CV_Std': r['cv_std'],
            'Test_R2': r['test_r2'],
            'Overfit': r['overfit']
        }
        for r in all_results
    ])
    df_summary = df_summary.sort_values('Test_R2', ascending=False)
    print(df_summary.to_string(index=False))

    # Best overall
    best_entry = max(all_results, key=lambda x: x['test_r2'])
    print(f"\n{'='*80}")
    print("Global Best Model")
    print('='*80)
    print(f"SMOTER ratio: {best_entry['SMOTER']}")
    print(f"Number of features: {best_entry['n_feat']}")
    print("Selected features:", best_entry['selected_features'])
    print("\n[Best Hyperparameters]")
    for k, v in best_entry['best_params'].items():
        print(f"  {k}: {v}")
    print("\n[Evaluation Metrics]")
    print(f"  Train R²:    {best_entry['train_r2']:.4f}")
    print(f"  Test R²:     {best_entry['test_r2']:.4f}")
    print(f"  Train RMSE:  {best_entry['train_rmse']:.4f}")
    print(f"  Test RMSE:   {best_entry['test_rmse']:.4f}")
    print(f"  Train MAE:   {best_entry['train_mae']:.4f}")
    print(f"  Test MAE:    {best_entry['test_mae']:.4f}")
    print(f"  Overfitting: {best_entry['overfit']:.4f}")
    print(f"  CV R² Mean:  {best_entry['cv_mean']:.4f}  (3-fold CV)")
    print(f"  CV R² Std:   {best_entry['cv_std']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()