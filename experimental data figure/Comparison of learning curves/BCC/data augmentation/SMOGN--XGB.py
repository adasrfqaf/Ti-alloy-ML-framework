# =============================================================================
# XGBoost Depth Overfitting Analysis (Based on SMOGN-Augmented Data)
# Fixed all best parameters except max_depth to observe depth's effect on overfitting
# Output: depth_overfitting_smogn.png
# All comments written in English, only PNG output retained
# =============================================================================

import pandas as pd
import numpy as np
import random
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ====================== Fixed Global Matplotlib & Random Seed Settings ======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

SEED = 49

def set_global_seed(seed=42):
    """Fix random seeds for full reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_global_seed(SEED)

# ====================== Clean special characters in dataframe column names ======================
def clean_column_names(df):
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

# ====================== SMOGN sample augmentation algorithm implementation ======================
def smogn_augmentation(X, y, smoter_ratio=1.0, noise_ratio=0.05, k=7,
                       bins=10, extreme_factor=2.0, dist_threshold_factor=0.8,
                       random_state=SEED):
    np.random.seed(random_state)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X = X.copy()
    y = y.copy()
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)

    if n_generate <= 0:
        return X, y

    y_percentile = np.percentile(y, np.linspace(0, 100, bins + 1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices - 1] if bins > 1 else np.ones(n_original)
    sample_weights = sample_weights / sample_weights.sum()

    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean')
    nn.fit(X.values)

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

# ====================== Raw dataset preprocessing pipeline ======================
def preprocess_dataframe(df, target_col):
    df = clean_column_names(df)
    if target_col not in df.columns:
        cleaned_target = re.sub(r'[\[\]\(\),;:\s]+', '_', target_col).strip('_')
        if cleaned_target in df.columns:
            target_col = cleaned_target
            print(f"Target column auto-mapped to: {target_col}")
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
            print(f"Removing constant column: {col}")
            df.drop(columns=[col], inplace=True)
    return df, target_col

# ====================== Feature selection based on cumulative feature importance (95% threshold) ======================
def select_features_by_importance(X_train, y_train, target_ratio=0.95):
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
    print(f"Cumulative importance {target_ratio * 100}% requires {n_selected} features")
    print("Selected features:", list(selected_features))
    return selected_features

# ====================== Main execution program ======================
if __name__ == "__main__":
    # File directory, modify this path according to your local file location
    file_path = r"BCC_data.csv"
    target_col = "Max_H2_Uptake_wt_pct"

    # Fixed hyperparameters for SMOGN augmentation
    SMOTER_RATIO = 3
    NOISE_RATIO = 0.05

    # Step 1: Load and clean raw dataset
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    df, target_col = preprocess_dataframe(df, target_col)
    print(f"Data shape after preprocessing: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 2: Train-test split with fixed random state
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_orig = X_train_orig.reset_index(drop=True)
    y_train_orig = y_train_orig.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Original training set size: {X_train_orig.shape}, Test set size: {X_test.shape}")

    # Step 3: Filter features by cumulative importance
    selected_features = select_features_by_importance(X_train_orig, y_train_orig, target_ratio=0.95)
    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]
    print(f"Training set size after feature selection: {X_train_orig.shape}")

    # Step 4: Execute SMOGN data augmentation on training set
    print(f"\nApplying SMOGN augmentation: smoter_ratio={SMOTER_RATIO}, noise_ratio={NOISE_RATIO}")
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=SMOTER_RATIO,
        noise_ratio=NOISE_RATIO,
        k=7,
        bins=10,
        extreme_factor=2.0,
        dist_threshold_factor=0.8
    )
    print(f"Augmented training dataset sample count: {X_train_aug.shape[0]}")

    # Step 5: Optimized fixed XGBoost hyperparameters from grid search
    best_params_fixed = {
        'n_estimators': 273,
        'learning_rate': 0.29,
        'subsample': 0.84,
        'colsample_bytree': 0.8,
        'min_child_weight': 7,
        'reg_alpha': 1.56,
        'reg_lambda': 0.921,
        'random_state': SEED,
        'verbosity': 0,
        'n_jobs': 1
    }

    # List of tree depth values for traversal test
    depth_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Initialize metric storage lists
    train_r2_list = []
    test_r2_list = []
    delta_r2_list = []

    print("\nStart evaluating model performance under different tree depth...")
    for depth in depth_list:
        print(f"  Running model with max_depth = {depth} ...")
        params = best_params_fixed.copy()
        params['max_depth'] = depth
        model = xgb.XGBRegressor(**params)

        model.fit(X_train_aug, y_train_aug)

        y_train_pred = model.predict(X_train_aug)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train_aug, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        delta = train_r2 - test_r2

        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)
        delta_r2_list.append(delta)

    # ====================== Plot drawing module (Journal standard version) ======================
    fig, ax1 = plt.subplots(figsize=(14, 9))

    # Journal-friendly color palette: Azure(training), Coral red(test), Fresh green(bar)
    color_train = "#4EA8C0"
    color_test = "#FFA453"
    color_delta_bar = "#B6B0D9"

    # Plot training R² (circle marker) & test R² (square marker) + set layer zorder
    line1, = ax1.plot(depth_list, train_r2_list, 'o-', color=color_train,
                      linewidth=3.5, markersize=12, label='Training R$^2$', zorder=5)
    line2, = ax1.plot(depth_list, test_r2_list, 's-', color=color_test,
                      linewidth=3.5, markersize=12, label='Test R$^2$', zorder=4)

    # Annotate training R² values above each data point
    for x, y in zip(depth_list, train_r2_list):
        ax1.text(x, y + 0.007, f"{y:.3f}", ha="center", va="bottom",
                 fontsize=16, color=color_train, fontweight="bold", fontfamily="Times New Roman", zorder=6)

    # Increase vertical spacing for test labels to prevent crowding with test line
    for x, y in zip(depth_list, test_r2_list):
        ax1.text(x, y - 0.022, f"{y:.3f}", ha="center", va="top",
                 fontsize=16, color=color_test, fontweight="bold", fontfamily="Times New Roman", zorder=6)

    # Configure primary left axis for R² score
    ax1.set_xlabel("Tree Depth (max_depth)", fontsize=32, fontweight="bold", fontfamily="Times New Roman")
    ax1.set_ylabel("R$^2$ Score", fontsize=32, fontweight="bold", fontfamily="Times New Roman")
    ax1.tick_params(axis="both", labelsize=20, width=2, length=8, direction="in")
    for tick_label in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick_label.set(fontfamily="Times New Roman", fontweight="bold")
    ax1.set_xlim(0.5, 8.5)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=24)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # Create secondary y-axis for ΔR² bars + low zorder
    ax2 = ax1.twinx()
    bars = ax2.bar(depth_list, delta_r2_list, width=0.42, alpha=0.65, color=color_delta_bar,
                   label="ΔR$^2$ (Train - Test)", zorder=3)
    ax2.set_ylabel("ΔR$^2$ (Overfitting degree)", fontsize=32, fontweight="bold",
                   fontfamily="Times New Roman", color=color_delta_bar)
    ax2.tick_params(axis="y", labelcolor=color_delta_bar, labelsize=20, width=2, length=8, direction="in")
    for tick_label in ax2.get_yticklabels():
        tick_label.set(fontfamily="Times New Roman", fontweight="bold")

    max_delta = max(delta_r2_list)
    ax2.set_ylim(bottom=0, top=max_delta * 1.5)

    # Place ΔR² text labels on top of each bar
    for bar, value in zip(bars, delta_r2_list):
        bar_height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, bar_height + max_delta * 0.025,
                 f"{value:.3f}", ha="center", va="bottom", fontsize=14,
                 color=color_delta_bar, fontweight="bold", fontfamily="Times New Roman", zorder=4)

    # Move combined legend to lower right corner + set legend to topmost layer (zorder=100)
    from matplotlib.patches import Patch
    bar_patch = Patch(color=color_delta_bar, alpha=0.65, label="ΔR$^2$ (Train - Test)")
    legend = ax1.legend(handles=[line1, line2, bar_patch], loc="lower right", fontsize=20, frameon=True, shadow=False)
    legend.set_zorder(100)  # Legend floating on the top of all elements
    for txt in legend.get_texts():
        txt.set(fontfamily="Times New Roman", fontweight="bold")
    legend.get_frame().set_linewidth(2)

    # Figure title
    plt.title("Effect of Tree Depth on Model\nPerformance and Overfitting (C14 Alloy Screening with SMOGN)",
              fontsize=32, fontweight="bold", fontfamily="Times New Roman", pad=28)

    # Layout adjustment
    plt.subplots_adjust(bottom=0.15, right=0.95)
    plt.tight_layout(pad=3.0)

    # Only export high resolution PNG file
    plt.savefig("depth_overfitting_smogn.png", dpi=600, bbox_inches="tight")
    print("High-resolution PNG figure has been saved successfully for journal submission")
    plt.show()

    # Output data summary on console
    print("\n=== Depth Sensitivity Analysis Summary ===")
    print(f"{'Depth':<6}{'Train R²':<10}{'Test R²':<10}{'ΔR²':<8}")
    for d, tr, te, de in zip(depth_list, train_r2_list, test_r2_list, delta_r2_list):
        print(f"{d:<6}{tr:.4f}     {te:.4f}     {de:.4f}")