# =============================================================================
# GBDT Tree Depth Sensitivity Analysis for TiFe Raw Unaugmented Dataset
# Fix all hyperparameters from GridSearch, only change max_depth for comparison
# Dual-axis figure: R² line chart + ΔR² overfitting bar chart
# All comments written in English, output PNG & PDF vector graph
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# -------------------------- Global Matplotlib Font & Axis Style Setup --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ==================== Global Configuration Parameters ====================
RANDOM_STATE = 42
N_FEATURES = 20
TEST_SIZE = 0.2
CV_FOLDS = 5
DEPTH_LIST = list(range(1, 11))  # Test tree depth from 1 to 10

# ==================== Step 1: Load Raw TiFe Dataset & Remove Redundant Columns ====================
file_path = r'TiFe_data.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
print("Dataset shape:", df.shape)

target = 'Max_H2_Uptake_wt_pct'
df = df.dropna(subset=[target])

# Remove redundant additive columns (identical to element percentage columns)
additive_cols = [col for col in df.columns if col.startswith('Additive_')]
cols_to_drop = []
for add_col in additive_cols:
    elem_col = add_col.replace('Additive_', 'Element_at_pct_')
    if elem_col in df.columns and df[add_col].equals(df[elem_col]):
        cols_to_drop.append(add_col)
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Removed redundant additive columns: {cols_to_drop}")

features = df.columns.drop(target)
X = df[features]
y = df[target]

print(f"\nTarget variable: {target}")
print(f"Number of features after cleaning: {X.shape[1]}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ==================== Step 2: Feature Importance Screening ====================
print("\n=== Calculating feature importance and selecting top features ===")
gbdt_init = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
gbdt_init.fit(X_train, y_train)

importances = gbdt_init.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)

print(f"\nFeature importance ranking (Top 20):")
for i, (idx, row) in enumerate(importance_df.head(20).iterrows(), 1):
    print(f"{i}. {row['feature']} : {row['importance']:.4f}")

# Select top N important features
selected_features = importance_df.head(N_FEATURES)['feature'].tolist()
print(f"\nSelected top {N_FEATURES} features:")
print(selected_features)

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# ==================== Step 3: Grid Search to Get Fixed Base Hyperparameters ====================
param_grid = {
    'n_estimators': [150],
    'learning_rate': [0.05],
    'max_depth': [3],
    'min_samples_split': [5],
    'min_samples_leaf': [4],
    'max_features': ['sqrt'],
    'subsample': [0.8]
}

base_gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE)

print("\n=== Starting Grid Search (local fine-tuning) ===")
grid_search = GridSearchCV(
    estimator=base_gbdt,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=CV_FOLDS,
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_train_sel, y_train)

print("\n=== Best Fixed Hyperparameters (only max_depth will be modified in loop) ===")
best_params = grid_search.best_params_
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Best CV MSE: {-grid_search.best_score_:.4f}")

# ==================== Step 4: Traverse All Tree Depths, Fix Other Params ====================
train_r2_list = []
test_r2_list = []
delta_r2_list = []

print("\nStart training GBDT models with different max_depth (all other params fixed)...")
for depth in DEPTH_LIST:
    print(f"  Processing max_depth = {depth} ...")
    current_params = best_params.copy()
    current_params['max_depth'] = depth
    temp_model = GradientBoostingRegressor(**current_params)
    temp_model.fit(X_train_sel, y_train)

    y_train_pred = temp_model.predict(X_train_sel)
    y_test_pred = temp_model.predict(X_test_sel)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    delta_gap = train_r2 - test_r2

    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    delta_r2_list.append(delta_gap)

# ==================== Step 5: Evaluate Model With Optimal max_depth ====================
final_model = grid_search.best_estimator_
y_train_pred = final_model.predict(X_train_sel)
y_test_pred = final_model.predict(X_test_sel)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
overfit_gap = train_r2 - test_r2

# 5-fold cross validation R² score
cv_r2_scores = cross_val_score(final_model, X_train_sel, y_train, cv=CV_FOLDS, scoring='r2')
cv_r2_mean = cv_r2_scores.mean()
cv_r2_std = cv_r2_scores.std()

print("\n" + "="*60)
print(f"Model Evaluation Results (Features={N_FEATURES})")
print("="*60)
print(f"Training RMSE:  {train_rmse:.4f}")
print(f"Testing RMSE:   {test_rmse:.4f}")
print(f"Training MAE:   {train_mae:.4f}")
print(f"Testing MAE:    {test_mae:.4f}")
print(f"Training R²:    {train_r2:.4f}")
print(f"Testing R²:     {test_r2:.4f}")
print(f"Overfitting:    {overfit_gap:.4f}")
print(f"5-Fold CV R² Mean: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
if overfit_gap > 0.05:
    print("  → Slight overfitting detected.")
elif overfit_gap < -0.05:
    print("  → Test set outperforms training set.")
else:
    print("  → Good generalization.")
print("="*60)

# ==================== Step 6: Unified Drawing Module (Revised label & axis layout) ====================
fig, ax1 = plt.subplots(figsize=(14, 9))

# Custom fixed color palette
color_train = "#4EA8C0"    # Azure blue for training curve
color_test = "#FFA453"     # Coral orange for test curve
color_delta = "#B6B0D9"    # Lavender for ΔR² bar chart

# Plot training & test R² lines with layer priority
line1, = ax1.plot(DEPTH_LIST, train_r2_list, 'o-', color=color_train,
                  linewidth=3.5, markersize=12, label='Training R$^2$', zorder=5)
line2, = ax1.plot(DEPTH_LIST, test_r2_list, 's-', color=color_test,
                  linewidth=3.5, markersize=12, label='Test R$^2$', zorder=4)

# ========== 1. Reset label offset: place training data labels upward, test data labels downward to fully separate texts from bars ==========
offset_train_up = 0.012   # Larger upward offset for training R² labels
offset_test_down = 0.012  # Downward offset for test R² labels to avoid overlapping top values of lines
for x, y_val in zip(DEPTH_LIST, train_r2_list):
    ax1.text(x, y_val + offset_train_up, f'{y_val:.3f}', ha='center', va='bottom',
             fontsize=16, color=color_train, fontweight='bold',
             fontfamily='Times New Roman', zorder=6)
for x, y_val in zip(DEPTH_LIST, test_r2_list):
    ax1.text(x, y_val - offset_test_down, f'{y_val:.3f}', ha='center', va='top',
             fontsize=16, color=color_test, fontweight='bold',
             fontfamily='Times New Roman', zorder=6)

# Left Y-axis formatting for R² Score curves
ax1.set_xlabel('Tree Depth (max_depth)', fontsize=28,
               fontfamily='Times New Roman', fontweight='bold', labelpad=12)
ax1.set_ylabel('R$^2$ Score', fontsize=28,
               fontfamily='Times New Roman', fontweight='bold', labelpad=12)
ax1.tick_params(axis='both', labelsize=18, width=2, length=8)
ax1.set_xticks(DEPTH_LIST)
ax1.set_xticklabels([str(d) for d in DEPTH_LIST],
                    fontfamily='Times New Roman', fontweight='bold')

# ========== 2. Increase top margin of left Y-axis to prevent data points from touching the upper boundary ==========
ymin = 0.60
ymax_raw = max(max(train_r2_list), max(test_r2_list)) + 0.06
ax1.set_ylim(ymin, ymax_raw)
ytick_step = 0.1
yticks = np.arange(0.50, 1.01, ytick_step)
ax1.set_yticks(yticks)
for tick in ax1.get_yticklabels():
    tick.set_fontfamily('Times New Roman')
    tick.set_fontweight('bold')

# Set bold linewidth for all axis spines
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Create secondary twin Y-axis for overfitting bar chart
ax2 = ax1.twinx()
bars = ax2.bar(DEPTH_LIST, delta_r2_list, width=0.4, alpha=0.6,
               color=color_delta, zorder=3)
ax2.set_ylabel('ΔR$^2$ (Overfitting degree)', fontsize=28,
               fontfamily='Times New Roman', fontweight='bold',
               color=color_delta, labelpad=15)
ax2.tick_params(axis='y', labelcolor=color_delta, labelsize=18, width=2, length=8)
for tick in ax2.get_yticklabels():
    tick.set_fontfamily('Times New Roman')
    tick.set_fontweight('bold')

# ========== 3. Fix the range of right-side ΔR² axis from 0 to 0.3 with matched ticks and proper vertical scaling ==========
ax2.set_ylim(0, 0.32)
ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

# Fine-tune offset for value labels on bar columns
bar_offset = 0.005
for bar, val in zip(bars, delta_r2_list):
    bar_h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, bar_h + bar_offset, f'{val:.3f}',
             ha='center', va='bottom', fontsize=15, color=color_delta,
             fontweight='bold', fontfamily='Times New Roman', zorder=6)

# ========== 4. Adjust legend position: move outward & slightly upward at upper right to avoid covering curve data ==========
bar_legend = Patch(facecolor=color_delta, alpha=0.6, label='ΔR$^2$ (Train - Test)')
handles = [line1, line2, bar_legend]
legend = ax1.legend(handles=handles, loc='center right',
                    bbox_to_anchor=(1.0, 0.1), fontsize=18, frameon=True, shadow=True)
legend.set_zorder(100)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_linewidth(2)
for text in legend.get_texts():
    text.set_fontfamily('Times New Roman')
    text.set_fontweight('bold')

# Overall figure title
plt.title('Effect of Tree Depth on XGBoost Performance (TiFe Raw Dataset)',
          fontsize=30, fontfamily='Times New Roman', fontweight='bold', pad=35)

# Optimize layout to eliminate label clipping
plt.tight_layout(pad=3.0)

# Export high-resolution PNG and vector PDF files for journal submission
plt.savefig('gbdt_depth_sensitivity_TiFe_raw.png', dpi=600, bbox_inches='tight')
print("\nFigure saved: gbdt_depth_sensitivity_TiFe_raw.png / .pdf")
plt.show()