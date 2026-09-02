import pandas as pd
import numpy as np
import random
import os
import re
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ====================== Global plotting style for SCI journals ======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# ====================== Fixed random seed ======================
SEED = 49


def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(SEED)


# ====================== Column name cleaning ======================
def clean_column_names(df):
    """
    Remove special characters from column names and replace with underscores.
    """
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        cleaned = re.sub(r'[\[\]\(\),;:\s]+', '_', str(col))
        cleaned = cleaned.strip('_')
        if not cleaned:
            cleaned = col
        new_cols[col] = cleaned
    df.rename(columns=new_cols, inplace=True)
    return df


# ====================== Manual SMOGN implementation ======================
def smogn_augmentation(X, y, smoter_ratio=1.0, noise_ratio=0.05, k=7,
                       bins=10, extreme_factor=2.0, dist_threshold_factor=0.8,
                       random_state=SEED):
    """
    Synthetic Minority Over-sampling for Regression with Gaussian Noise (SMOGN).
    Generates synthetic samples for both minority and majority regions.
    """
    np.random.seed(random_state)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X = X.copy()
    y = y.copy()
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)

    if n_generate <= 0:
        return X, y

    # Determine bin edges based on percentile to identify rare/extreme cases
    y_percentile = np.percentile(y, np.linspace(0, 100, bins + 1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor          # weight for lowest extreme
    bin_weights[-1] = extreme_factor         # weight for highest extreme
    sample_weights = bin_weights[bin_indices - 1] if bins > 1 else np.ones(n_original)
    sample_weights = sample_weights / sample_weights.sum()

    # Nearest neighbors for distance calculation
    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)

    # Compute global average distance for thresholding
    all_distances = []
    for i in range(n_original):
        dist, _ = nn.kneighbors(X.iloc[i].values.reshape(1, -1), n_neighbors=k + 1)
        all_distances.append(dist[0][1:].mean())
    global_avg_dist = np.mean(all_distances)
    dist_threshold = global_avg_dist * dist_threshold_factor

    X_list = [X]
    y_list = [y]

    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y.iloc[idx]

        distances, indices = nn.kneighbors(x_seed.reshape(1, -1), n_neighbors=k + 1)
        neighbor_indices = indices[0][1:]
        avg_dist_to_neighbors = distances[0][1:].mean()

        # If seed is in a sparse region, add Gaussian noise; otherwise interpolate
        if avg_dist_to_neighbors > dist_threshold:
            x_new = x_seed.copy()
            for i_col, col in enumerate(X.columns):
                std_col = X[col].std()
                if std_col > 0:
                    x_new[i_col] += np.random.normal(0, noise_ratio * std_col)
            y_new = y_seed
        else:
            neighbor_idx = np.random.choice(neighbor_indices)
            x_neighbor = X.iloc[neighbor_idx].values
            y_neighbor = y.iloc[neighbor_idx]
            lam = np.random.uniform()
            x_new = x_seed + lam * (x_neighbor - x_seed)
            y_new = y_seed + lam * (y_neighbor - y_seed)

        X_list.append(pd.DataFrame([x_new], columns=X.columns))
        y_list.append([y_new])

    X_aug = pd.concat(X_list, ignore_index=True)
    y_aug = np.concatenate(y_list)
    return X_aug, y_aug


# ====================== Data preprocessing ======================
def preprocess_dataframe(df, target_col):
    """
    Clean column names, convert bools to ints, drop constant columns,
    and keep only numeric features.
    """
    df = clean_column_names(df)
    if target_col not in df.columns:
        cleaned_target = re.sub(r'[\[\]\(\),;:\s]+', '_', target_col).strip('_')
        if cleaned_target in df.columns:
            target_col = cleaned_target
            print(f"Target column auto‑mapped to: {target_col}")
        else:
            raise ValueError(f"Target column '{target_col}' not found")
    df = df.copy()
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"Target column {target_col} is not numeric")
    df = df[numeric_cols]
    for col in df.columns:
        if df[col].nunique() <= 1:
            print(f"Dropping constant column: {col}")
            df.drop(columns=[col], inplace=True)
    return df, target_col


# ====================== Feature selection (cumulative importance 95% using XGBoost) ======================
def select_features_by_importance(X_train, y_train, target_ratio=0.95):
    """
    Select features that cumulatively account for target_ratio of total importance.
    """
    xgb_temp = xgb.XGBRegressor(
        n_estimators=100,
        random_state=SEED,
        verbosity=0,
        n_jobs=1
    )
    xgb_temp.fit(X_train, y_train)
    importances = xgb_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[indices])
    n_selected = np.searchsorted(cumsum, target_ratio) + 1
    selected_features = X_train.columns[indices[:n_selected]]
    print(f"Cumulative importance of {target_ratio * 100:.0f}% requires {n_selected} features")
    print("Selected features:", list(selected_features))
    return selected_features


# ====================== Fine grid search for XGBoost (with CV R² mean ± std) ======================
def fine_grid_search_xgb(X_train, y_train, X_test, y_test, cv=5):
    """
    Perform a predefined fine‑grid search over hyperparameters.
    Returns the best model and performance metrics.
    """
    param_grid = {
        'n_estimators': [273],
        'max_depth': [4],
        'learning_rate': [0.29],
        'subsample': [0.84],
        'colsample_bytree': [0.8],
        'min_child_weight': [7],
        'reg_alpha': [1.56],
        'reg_lambda': [0.921]
    }
    xgb_model = xgb.XGBRegressor(
        random_state=SEED,
        verbosity=0,
        n_jobs=1
    )
    grid = GridSearchCV(
        xgb_model, param_grid, cv=cv, scoring='r2',
        n_jobs=1, verbose=1
    )
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_

    cv_results = grid.cv_results_
    best_index = grid.best_index_
    cv_mean = cv_results['mean_test_score'][best_index]
    cv_std = cv_results['std_test_score'][best_index]

    y_train_pred = best_xgb.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    y_test_pred = best_xgb.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    best_params = grid.best_params_
    return best_xgb, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae, best_params, cv_mean, cv_std


# ====================== Main program ======================
if __name__ == "__main__":
    # File path (adjust as needed)
    file_path = r"BCC_data.csv"
    target_col = "Max_H2_Uptake_wt_pct"

    SMOTER_RATIO = 3
    NOISE_RATIO = 0.05

    # 1. Load and preprocess
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    df, target_col = preprocess_dataframe(df, target_col)
    print(f"After preprocessing shape: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Fixed train/test split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_orig = X_train_orig.reset_index(drop=True)
    y_train_orig = y_train_orig.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Original training set: {X_train_orig.shape}, test set: {X_test.shape}")

    # 3. Feature selection based on original training data
    selected_features = select_features_by_importance(X_train_orig, y_train_orig, target_ratio=0.95)
    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]
    print(f"After feature selection, training set: {X_train_orig.shape}")

    # 4. SMOGN augmentation
    print(f"\n{'=' * 70}")
    print(f"SMOGN augmentation: smoter_ratio = {SMOTER_RATIO} (total samples become {1 + SMOTER_RATIO}×), noise_ratio = {NOISE_RATIO}")
    print('=' * 70)
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=SMOTER_RATIO,
        noise_ratio=NOISE_RATIO,
        k=7,
        bins=10,
        extreme_factor=2.0,
        dist_threshold_factor=0.8
    )
    print(f"Augmented training set size: {X_train_aug.shape[0]}")

    # 5. Fine grid search
    print("Performing fine grid search (5‑fold CV)...")
    best_model, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae, best_params, cv_mean, cv_std = fine_grid_search_xgb(
        X_train_aug, y_train_aug, X_test, y_test, cv=5
    )

    # 6. Compute overfitting degree
    overfit = train_r2 - test_r2

    # 7. Print all requested metrics (for viewing only, not saved)
    print("\n" + "=" * 70)
    print("Final model performance evaluation")
    print("=" * 70)
    print(f"Best hyperparameters: {best_params}")
    print(f"\nCross‑validation R² (5‑fold): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"\nTraining set metrics:")
    print(f"  R²   = {train_r2:.4f}")
    print(f"  RMSE = {train_rmse:.4f}")
    print(f"  MAE  = {train_mae:.4f}")
    print(f"\nTest set metrics:")
    print(f"  R²   = {test_r2:.4f}")
    print(f"  RMSE = {test_rmse:.4f}")
    print(f"  MAE  = {test_mae:.4f}")
    print(f"\nOverfitting (train R² - test R²) = {overfit:.4f}")

    # ====================== 8. SHAP explanation and save figure ======================
    print("\n" + "=" * 120)
    print("Generating SHAP summary plot (Times New Roman, top 10 features)...")
    print("=" * 120)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'

    sample_size = min(200, len(X_train_aug))
    X_sample = X_train_aug.sample(n=sample_size, random_state=SEED)

    # Use column names directly (already in English)
    translated_names = list(X_sample.columns)

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=translated_names,
        max_display=10,
        show=False
    )

    ax = plt.gca()
    # Remove vertical dashed line at x=0
    for line in ax.lines:
        if line.get_linestyle() == '--' and len(line.get_xdata()) > 0 and abs(line.get_xdata()[0]) < 1e-6:
            line.remove()

    ax.tick_params(axis='both', labelsize=18, width=2, length=8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)

    ax.set_xlabel('SHAP value (impact on model output)',
                  fontfamily='Times New Roman', fontsize=18, fontweight='bold')
    ax.set_ylabel('Features',
                  fontfamily='Times New Roman', fontsize=18, fontweight='bold')

    # Colorbar customization (as before)
    fig = plt.gcf()
    cbar_ax = None
    for ax_ in fig.axes:
        if ax_ is not ax:
            if ax_.get_ylabel() == 'Feature value':
                cbar_ax = ax_
                break
    if cbar_ax is None and len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]

    if cbar_ax is not None:
        cbar_ax.tick_params(labelsize=16, width=2, length=6)
        for tick in cbar_ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontweight('bold')
            tick.set_fontsize(12)
        cbar_ax.set_ylabel('Feature value', fontfamily='Times New Roman',
                           fontsize=18, fontweight='bold')
        for text in cbar_ax.texts:
            if text.get_text() in ['Low', 'High', 'low', 'high']:
                text.set_fontfamily('Times New Roman')
                text.set_fontweight('bold')
                text.set_fontsize(18)

    for text in ax.texts:
        if text.get_text() in ['Low', 'High', 'low', 'high']:
            text.set_fontfamily('Times New Roman')
            text.set_fontweight('bold')
            text.set_fontsize(12)

    plt.tight_layout()
    plt.savefig('shap_summary_plot_BCC.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(" Saved SHAP summary plot (top 10): shap_summary_plot_BCC.png")