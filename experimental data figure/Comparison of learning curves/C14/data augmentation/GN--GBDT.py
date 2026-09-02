# =============================================================================
# Complete Script: GBDT Depth Sensitivity Analysis (sigma=0.1, copies=3 Gaussian Noise Augmentation)
# Dataset: C14 English feature names version
# Modify: Labels shift left to avoid covering lines, twin Y-axis compressed to prevent overlap
# Value labels placed at left-middle of each data point / inside bar center
# Output image only: gbdt_depth_sensitivity_C14_gaussian_final.png
# All comments in English, Fixed seed = 49, auto clean old cache csv files
# =============================================================================
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

# -------------------------- Global Font & Style Configuration --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Fix all random seeds for reproducibility
SEED = 49
def set_global_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_global_seed(SEED)

# ====================== Global Experiment Settings ======================
FILE_PATH = r'C14_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 18
# Target augmentation parameters
SIGMA_TARGET = 0.1
COPY_TARGET = 3

# Fixed hyperparameter grid for random search (all fixed values, no actual search)
PARAM_DIST = {
    'n_estimators': [100],
    'learning_rate': [0.11],
    'max_depth': [7],
    'min_samples_split': [8],
    'min_samples_leaf': [2],
    'max_features': [0.55],
    'subsample': [0.9],
    'loss': ['squared_error'],
    'validation_fraction': [0.2],
    'n_iter_no_change': [25]
}
N_ITER = 50

# Gaussian noise augmentation hyperparameters
SIGMA_RATIOS = [0.02, 0.05, 0.08, 0.1]
EXPANSION_FACTORS = [2, 3, 4, 5, 6]

# Auto clean historical cache files (no new files will be created)
cache_file_list = [
    'X_train_aug_0.1_3.csv',
    'y_train_aug_0.1_3.csv',
    'X_test_sel.csv',
    'y_test.csv'
]
for f_name in cache_file_list:
    if os.path.exists(f_name):
        os.remove(f_name)
        print(f"Old cache file removed: {f_name}")

# ====================== Data Loading & Preprocessing Function ======================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if TARGET_COL not in df.columns:
        raise ValueError("Target column not found in dataset")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    # Convert boolean features to integer
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    # Retain only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    X = X.fillna(0)
    y = np.nan_to_num(y)
    return X, y

# Select top N important features via Random Forest
def select_top_features_rf(X_train, y_train, top_n=18):
    from sklearn.ensemble import RandomForestRegressor
    temp_model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE,
        n_jobs=1, max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    temp_model.fit(X_scaled, y_train)
    importance = temp_model.feature_importances_
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    return top_features

# Add relative Gaussian noise to feature matrix for data augmentation
def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=2):
    if n_copies == 0 or sigma_ratio == 0.0:
        return X, y
    X_noisy_list = [X]
    y_noisy_list = [y]
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            col_std = stds[col]
            if col_std > 0:
                noise = np.random.normal(0, sigma_ratio * col_std, len(X))
                X_copy[col] = X_copy[col] + noise
        X_noisy_list.append(X_copy)
        y_noisy_list.append(y)
    X_aug = pd.concat(X_noisy_list, ignore_index=True)
    y_aug = np.concatenate(y_noisy_list)
    return X_aug, y_aug

# Calculate R2, MAE, RMSE for regression model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse

# ====================== Main Experiment Pipeline ======================
print("=" * 100)
print("GBDT Depth Analysis Pipeline (Gaussian Noise Augmentation sigma=0.1, copies=3 | SEED=49)")
print("=" * 100)

# Load raw dataset
X, y = load_data(FILE_PATH)
print(f"Total samples: {len(X)}, total raw features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

# Feature selection based on RF importance
top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"Training set shape after feature selection: {X_train_sel.shape}")
print(f"Selected top {TOP_N_FEATURES} features:")
for i, f in enumerate(top_features, 1):
    print(f"  {i}. {f}")

# Calculate feature standard deviation for noise generation
feature_stds = X_train_sel.std(axis=0)

# ===== MODIFIED: Store augmented data in memory, no CSV writing =====
X_train_aug_final = None
y_train_aug_final = None

print("\nIterate all augmentation combinations to generate target dataset")
for sigma_ratio in SIGMA_RATIOS:
    for n_copies in EXPANSION_FACTORS:
        print(f"  Process sigma={sigma_ratio}, copies={n_copies} ...")
        if sigma_ratio == 0.0 or n_copies == 0:
            X_train_aug = X_train_sel
            y_train_aug = y_train
        else:
            X_train_aug, y_train_aug = add_relative_gaussian_noise(
                X_train_sel, y_train, sigma_ratio, feature_stds, n_copies=n_copies
            )
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_sel)
        # Random Search CV (fixed params, only fit)
        gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE)
        random_search = RandomizedSearchCV(
            gbdt, PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
            scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=0
        )
        random_search.fit(X_train_scaled, y_train_aug)
        # Save target combination in memory (no file output)
        if sigma_ratio == SIGMA_TARGET and n_copies == COPY_TARGET:
            X_train_aug_final = X_train_aug
            y_train_aug_final = y_train_aug
            print("  Target augmented dataset stored in memory (no CSV saved)")

# Ensure target data exists
if X_train_aug_final is None or y_train_aug_final is None:
    raise RuntimeError("Target augmented data not generated! Check parameters.")

print("Target augmented dataset ready in memory")

# ====================== GBDT Fixed Hyperparameters (only max_depth changed) ======================
FIXED_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.11,
    'min_samples_split': 8,
    'min_samples_leaf': 2,
    'max_features': 0.55,
    'subsample': 0.9,
    'loss': 'squared_error',
    'validation_fraction': 0.2,
    'n_iter_no_change': 25,
    'random_state': SEED,
    'verbose': 0
}
DEPTH_LIST = list(range(1, 11))
x_positions = np.arange(len(DEPTH_LIST))

# Standardize augmented train & original test set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_aug_final)   # Use in-memory data
X_test_scaled = scaler.transform(X_test_sel)               # X_test_sel already in memory

# Store evaluation metrics for each tree depth
train_r2_list = []
test_r2_list = []
delta_r2_list = []

print("\nStart training models with different max_depth to record performance metrics...")
for depth in DEPTH_LIST:
    print(f"  Train model with max_depth = {depth} ...")
    params = FIXED_PARAMS.copy()
    params['max_depth'] = depth
    model = GradientBoostingRegressor(**params)
    model.fit(X_train_scaled, y_train_aug_final)            # Use in-memory target y
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_r2 = r2_score(y_train_aug_final, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    delta = train_r2 - test_r2
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    delta_r2_list.append(delta)

# ====================== Drawing Module (Revised Layout) ======================
# Color palette:
#   color_train: Light blue for training R² curve
#   color_test:  Coral orange for test R² curve
#   color_delta: Light purple-gray for overfitting ΔR² bars
color_train = "#4EA8C0"
color_test = "#FFA453"
color_delta = "#B6B0D9"

fig, ax1 = plt.subplots(figsize=(14, 9))

# Plot training and test R² curves
line1, = ax1.plot(x_positions, train_r2_list, 'o-', linewidth=3.5, markersize=12,
                  color=color_train, label='Training R$^2$', zorder=5)
line2, = ax1.plot(x_positions, test_r2_list, 's-', linewidth=3.5, markersize=12,
                  color=color_test, label='Test R$^2$', zorder=4)

# Label config: shift left to avoid covering markers
label_left_offset = 0.02
vertical_gap = 0.03
# Training value: above point, left side
for x, y in zip(x_positions, train_r2_list):
    ax1.text(x + label_left_offset, y + vertical_gap, f'{y:.3f}', ha='center', va='bottom',
             fontsize=20, color=color_train, fontweight='bold', fontfamily='Times New Roman', zorder=6)
# Test value: below point, left side
for x, y in zip(x_positions, test_r2_list):
    ax1.text(x + label_left_offset, y - vertical_gap, f'{y:.3f}', ha='center', va='top',
             fontsize=20, color=color_test, fontweight='bold', fontfamily='Times New Roman', zorder=6)

# Left primary axis configuration (R² Score)
ax1.set_xlabel('Tree Depth (max_depth)', fontsize=32, fontweight='bold', fontfamily='Times New Roman')
ax1.set_ylabel('R$^2$ Score', fontsize=32, fontweight='bold', fontfamily='Times New Roman')
ax1.tick_params(axis='both', labelsize=20, width=2, length=8)
ax1.set_xticks(x_positions)
ax1.set_xticklabels([str(d) for d in DEPTH_LIST], fontfamily='Times New Roman', fontweight='bold', fontsize=24)
ax1.set_ylim(-0.05, 1.1)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                    fontfamily='Times New Roman', fontweight='bold', fontsize=20)
# Bold axis border lines
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Twin secondary axis: compress upper limit to separate bars & lines
ax2 = ax1.twinx()
bars = ax2.bar(x_positions, delta_r2_list, width=0.4, alpha=0.6,
               color=color_delta, label='ΔR$^2$ (Train - Test)', zorder=3)
ax2.set_ylabel('ΔR$^2$ (Overfitting degree)', fontsize=32, fontweight='bold',
               fontfamily='Times New Roman', color=color_delta)
ax2.tick_params(axis='y', labelcolor=color_delta, labelsize=20, width=2, length=8)
# Fixed right axis max value to avoid overlapping curve area
ax2.set_ylim(0, 0.32)

# Bar value inside vertical center, left offset
for bar, val in zip(bars, delta_r2_list):
    bar_h = bar.get_height()
    bar_x_center = bar.get_x() + bar.get_width() / 2
    ax2.text(bar_x_center + label_left_offset, bar_h + 0.01 , f'{val:.3f}', ha='center', va='center',
             fontsize=20, color=color_delta, fontweight='bold', fontfamily='Times New Roman', zorder=4)

# Merge legend, placed upper left
handles = [
    line1,
    line2,
    Patch(facecolor=color_delta, alpha=0.6, label='ΔR$^2$ (Train - Test)')
]
legend = ax1.legend(handles=handles, loc='center right', fontsize=20, frameon=True)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_facecolor('white')
legend.set_zorder(100)
for text in legend.get_texts():
    text.set_fontfamily('Times New Roman')
    text.set_fontweight('bold')

# Figure main title
plt.title('Effect of Tree Depth on Model\n Performance and Overfitting (C14 with GN)',
          fontsize=32, fontweight='bold', fontfamily='Times New Roman', pad=25)

plt.tight_layout(pad=3.0)
# Only save PNG image, no extra file output
plt.savefig('gbdt_depth_sensitivity_C14_gaussian_final.png', dpi=600, bbox_inches='tight')
plt.show()

# Print depth sensitivity result table (console only, no file)
print("\n=== Depth Sensitivity Analysis Result ===")
print("Depth\tTrain R²\tTest R²\tΔR²")
for d, tr, te, de in zip(DEPTH_LIST, train_r2_list, test_r2_list, delta_r2_list):
    print(f"{d}\t{tr:.4f}\t{te:.4f}\t{de:.4f}")