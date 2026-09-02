# =============================================================================
# GBDT Depth Sensitivity Analysis (TiFe Dataset, SMOGN Augmentation, Fixed Optimal Parameters)
# Vary max_depth only to observe overfitting trend
# Output: gbdt_depth_sensitivity_TiFe.png / pdf
# =============================================================================

import pandas as pd
import numpy as np
import random
import os
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ====================== Fix Random Seed ======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

SEED = 49
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_global_seed(SEED)

# ====================== Column Name Cleaning ======================
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

# ====================== SMOGN Oversampling ======================
def smogn_augmentation(X, y, smoter_ratio=1.0, noise_ratio=0.03, k=7,
                       bins=10, extreme_factor=1.8, dist_threshold_factor=0.5,
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

# ====================== Data Preprocessing ======================
def preprocess_dataframe(df, target_col):
    df = clean_column_names(df)
    if target_col not in df.columns:
        cleaned_target = re.sub(r'[\[\]\(\),;:\s]+', '_', target_col).strip('_')
        if cleaned_target in df.columns:
            target_col = cleaned_target
            print(f"Target column auto-mapped to: {target_col}")
        else:
            raise ValueError(f"Target column '{target_col}' does not exist")
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
            print(f"Drop constant column: {col}")
            df.drop(columns=[col], inplace=True)
    return df, target_col

# ====================== Feature Selection ======================
def select_features_by_count(X_train, y_train, n_features=15):
    temp_model = XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0)
    temp_model.fit(X_train, y_train)
    importances = temp_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    n_select = min(n_features, len(indices))
    selected_features = X_train.columns[indices[:n_select]]
    print(f"Feature importance ranking (XGBoost), top {n_select} features selected")
    print("Selected features:", list(selected_features))
    return selected_features

# ====================== Main Program (Depth Sensitivity Analysis) ======================
if __name__ == "__main__":
    # ----------------- Fixed Configuration -----------------
    FEATURE_COUNT = 18
    SMOTER_RATIO = 4
    NOISE_RATIO = 0.08
    FILE_PATH = r"TiFe_data.csv"
    TARGET_COL = "Max_H2_Uptake_wt_pct"

    # Fixed optimal hyperparameters (except max_depth)
    FIXED_PARAMS = {
        'n_estimators': 190,
        'learning_rate': 0.085,
        'subsample': 0.8,
        'min_samples_split': 15,
        'min_samples_leaf': 7,
        'random_state': SEED,
        'verbose': 0
    }

    DEPTH_LIST = list(range(1, 11))   # Depth 1~10
    x_positions = np.arange(len(DEPTH_LIST))

    # ----------------- Data Loading and Preprocessing -----------------
    print("=" * 70)
    print("GBDT Depth Sensitivity Analysis (TiFe Dataset, SMOGN Augmentation)")
    print("=" * 70)

    df = pd.read_csv(FILE_PATH)
    print(f"Raw data shape: {df.shape}")
    df, TARGET_COL = preprocess_dataframe(df, TARGET_COL)
    print(f"Preprocessed shape: {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train/test split (fixed split)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_orig = X_train_orig.reset_index(drop=True)
    y_train_orig = y_train_orig.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Original training set: {X_train_orig.shape}, test set: {X_test.shape}")

    # Feature selection
    selected_features = select_features_by_count(X_train_orig, y_train_orig, n_features=FEATURE_COUNT)
    X_train_orig = X_train_orig[selected_features]
    X_test = X_test[selected_features]
    print(f"Training set after feature selection: {X_train_orig.shape}")

    # ----------------- SMOGN Augmentation (once only) -----------------
    print(f"\nPerforming SMOGN oversampling: smoter_ratio = {SMOTER_RATIO}, noise_ratio = {NOISE_RATIO}")
    X_train_aug, y_train_aug = smogn_augmentation(
        X_train_orig, y_train_orig,
        smoter_ratio=SMOTER_RATIO,
        noise_ratio=NOISE_RATIO,
        k=7,
        bins=10,
        extreme_factor=1.8,
        dist_threshold_factor=0.5
    )
    print(f"Augmented training set size: {X_train_aug.shape[0]}")

    # ----------------- Standardization -----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    # ----------------- Loop Over Different Depths -----------------
    train_r2_list = []
    test_r2_list = []
    delta_r2_list = []

    print("\nStart looping over different max_depth values for performance evaluation...")
    for depth in DEPTH_LIST:
        print(f"  Training max_depth = {depth} ...")
        params = FIXED_PARAMS.copy()
        params['max_depth'] = depth
        model = GradientBoostingRegressor(**params)
        model.fit(X_train_scaled, y_train_aug)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_r2 = r2_score(y_train_aug, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        delta = train_r2 - test_r2

        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)
        delta_r2_list.append(delta)

    # ====================== Plotting ======================
    # Custom color scheme
    color_train = "#4EA8C0"      # Training set
    color_test = "#FFA453"       # Test set
    color_delta = "#B6B0D9"      # ΔR²

    fig, ax1 = plt.subplots(figsize=(14, 9))

    # Plot lines
    line1, = ax1.plot(x_positions, train_r2_list, 'o-', color=color_train,
                      linewidth=3.5, markersize=12, label='Training R²')
    line2, = ax1.plot(x_positions, test_r2_list, 's-', color=color_test,
                      linewidth=3.5, markersize=12, label='Test R²')

    # Training R² labels - above the line
    offset_train = 0.02
    for x, y_val in zip(x_positions, train_r2_list):
        ax1.text(x, y_val + offset_train, f'{y_val:.3f}', ha='center', va='bottom',
                 fontsize=20, color=color_train, fontweight='bold', fontfamily='Times New Roman')

    # Test R² labels - below the line
    offset_test = 0.02
    for x, y_val in zip(x_positions, test_r2_list):
        ax1.text(x, y_val - offset_test, f'{y_val:.3f}', ha='center', va='top',
                 fontsize=20, color=color_test, fontweight='bold', fontfamily='Times New Roman')

    # Axis settings
    ax1.set_xlabel('Tree Depth (max_depth)', fontsize=32, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('R² Score', fontsize=32, fontweight='bold', fontfamily='Times New Roman')

    ax1.tick_params(axis='both', labelsize=20, width=2, length=8)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(d) for d in DEPTH_LIST], fontfamily='Times New Roman', fontweight='bold', fontsize=20)
    for tick in ax1.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')

    # R² y-axis range 0~1.05 with ticks 0~1.0
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                         fontfamily='Times New Roman', fontweight='bold', fontsize=24)

    # Bold axis spines
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # Twin axis for ΔR² bar chart (expanded range to avoid overlapping with lines)
    ax2 = ax1.twinx()
    bars = ax2.bar(x_positions, delta_r2_list, width=0.4, alpha=0.6,
                   color=color_delta, label='ΔR² (Train - Test)')
    ax2.set_ylabel('ΔR² (Overfitting degree)', fontsize=32, fontweight='bold',
                   fontfamily='Times New Roman', color=color_delta)
    ax2.tick_params(axis='y', labelcolor=color_delta, labelsize=20, width=2, length=8)
    for tick in ax2.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')

    # Expand ΔR² y-axis range to keep bars below the lines
    max_delta = max(delta_r2_list) if max(delta_r2_list) > 0 else 0.001
    ax2.set_ylim(0, max_delta * 2.2)

    # Bar value labels - on top of bars
    bar_offset = max_delta * 0.02
    for x, v in zip(x_positions, delta_r2_list):
        ax2.text(x, v + bar_offset, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=20, color=color_delta, fontweight='bold', fontfamily='Times New Roman')

    # Combined legend, placed at center right with slight upward offset
    from matplotlib.patches import Patch
    handles = [
        line1,
        line2,
        Patch(facecolor=color_delta, alpha=0.6, label='ΔR² (Train - Test)')
    ]
    legend = ax1.legend(handles=handles, loc='center right', bbox_to_anchor=(1.0, 0.58),
                        fontsize=20, frameon=True, shadow=True)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')
    # Bold legend frame
    legend.get_frame().set_linewidth(2)

    # Chart title
    plt.title('Effect of Tree Depth on Model \nPerformance and Overfitting (TiFe with SMOGN)',
              fontsize=32, fontweight='bold', fontfamily='Times New Roman', pad=25)

    # Global padding
    plt.tight_layout(pad=3.0)

    # Save high-resolution figure and PDF
    plt.savefig('gbdt_depth_sensitivity_TiFe.png', dpi=600, bbox_inches='tight')
    print("\nFigure saved: full English Times New Roman font, large font size, R² range 0~1.05")
    plt.show()

    # Print result table
    print("\n=== Depth Sensitivity Analysis Results ===")
    print("Depth\tTrain R²\tTest R²\tΔR²")
    for d, tr, te, de in zip(DEPTH_LIST, train_r2_list, test_r2_list, delta_r2_list):
        print(f"{d}\t{tr:.4f}\t{te:.4f}\t{de:.4f}")