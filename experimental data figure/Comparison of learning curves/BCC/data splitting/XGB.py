# =============================================================================
# XGBoost Depth-Analysis Plot (Match your target figure style)
# C14 Alloy Screening dataset with SMOGN resampling
# =============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------- Global Font & Style Setup --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

RANDOM_STATE = 42
N_TOP_FEATURES = 18
N_JOBS = 1
DATA_PATH = r'D:\python_file\Titanium-based hydrogen storage alloy\BCC原始数据集\数据预处理\cleaned_BCC_data.csv'
DEPTH_LIST = [1, 2, 3, 4, 5, 6, 7, 8]

# -------------------------- Data Loading & Preprocessing --------------------------
df = pd.read_csv(DATA_PATH)
target = 'max_h2_uptake'
features = df.columns.drop(target)
df = df.dropna(subset=[target])

# Fill missing values
for col in features:
    if df[col].isnull().any():
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Encode category features
categorical_cols = df[features].select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Feature selection via XGB importance
xgb_init = xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0, n_jobs=N_JOBS)
xgb_init.fit(X_train, y_train)
importance_df = pd.DataFrame({'feature': X.columns, 'importance': xgb_init.feature_importances_})
importance_df = importance_df.sort_values('importance', ascending=False)
selected_features = importance_df.head(N_TOP_FEATURES)['feature'].tolist()
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# -------------------------- Fixed XGB Hyperparameters --------------------------
base_params = {
    'subsample': 0.7,
    'reg_lambda': 5,
    'reg_alpha': 1,
    'n_estimators': 300,
    'min_child_weight': 2,
    'learning_rate': 0.05,
    'gamma': 0,
    'colsample_bytree': 0.6,
    'random_state': RANDOM_STATE,
    'verbosity': 0,
    'n_jobs': N_JOBS
}

# -------------------------- Calculate metrics for each depth --------------------------
train_r2_list = []
test_r2_list = []
delta_r2_list = []
for depth in DEPTH_LIST:
    params = base_params.copy()
    params['max_depth'] = depth
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_sel, y_train)
    y_tr_pred = model.predict(X_train_sel)
    y_te_pred = model.predict(X_test_sel)
    tr_r2 = r2_score(y_train, y_tr_pred)
    te_r2 = r2_score(y_test, y_te_pred)
    delta = tr_r2 - te_r2
    train_r2_list.append(tr_r2)
    test_r2_list.append(te_r2)
    delta_r2_list.append(delta)

# -------------------------- Drawing (Fully matched target style) --------------------------
fig, ax1 = plt.subplots(figsize=(15, 9))
color_train = "#4EA8C0"
color_test = "#FFA453"
color_delta = "#B6B0D9"

# Draw two lines, z-order priority
line_train, = ax1.plot(DEPTH_LIST, train_r2_list, 'o-', c=color_train, lw=3.5, ms=12, label="Training R²", zorder=5)
line_test, = ax1.plot(DEPTH_LIST, test_r2_list, 's-', c=color_test, lw=3.5, ms=12, label="Test R²", zorder=4)

# R² value text labels
offset_up = 0.06
offset_down = 0.06
for x, y in zip(DEPTH_LIST, train_r2_list):
    ax1.text(x, y + offset_up, f"{y:.3f}", ha="center", va="top", fontsize=20, c=color_train, weight="bold", family="Times New Roman", zorder=6)
for x, y in zip(DEPTH_LIST, test_r2_list):
    ax1.text(x, y + offset_down, f"{y:.3f}", ha="center", va="top", fontsize=20, c=color_test, weight="bold", family="Times New Roman", zorder=6)

# Left axis (R² Score)
ax1.set_xlabel("Tree Depth (max_depth)", fontsize=32, weight="bold", family="Times New Roman", labelpad=12)
ax1.set_ylabel("R² Score", fontsize=32, weight="bold", family="Times New Roman", labelpad=12)
ax1.set_ylim(-0.05, 1.05)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.tick_params(axis="both", labelsize=20, width=2, length=8)
for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
    tick.set_fontweight("bold")
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Right twin axis for ΔR² bars
ax2 = ax1.twinx()
bars = ax2.bar(DEPTH_LIST, delta_r2_list, width=0.4, alpha=0.6, color=color_delta, zorder=3)
ax2.set_ylabel("ΔR² (Overfitting degree)", fontsize=32, weight="bold", family="Times New Roman", color=color_delta, labelpad=15)
# Fixed right axis limit same as sample figure: 0 ~ 0.200
ax2.set_ylim(-0.005, 0.305)
ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax2.tick_params(axis="y", labelcolor=color_delta, labelsize=20, width=2, length=8)
for tick in ax2.get_yticklabels():
    tick.set_fontweight("bold")

# Bar top value labels
bar_text_offset = 0.003
for bar, val in zip(bars, delta_r2_list):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + bar_text_offset, f"{val:.3f}",
             ha="center", va="bottom", fontsize=20, c=color_delta, weight="bold", family="Times New Roman", zorder=6)

# Legend: lower right + white solid background + top zorder (solve occlusion)
from matplotlib.patches import Patch
bar_legend_patch = Patch(color=color_delta, alpha=0.6, label="ΔR² (Train - Test)")
legend_handles = [line_train, line_test, bar_legend_patch]
legend = ax1.legend(handles=legend_handles, loc="lower right", fontsize=22, frameon=True)
legend.set_zorder(100)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_linewidth(2)
for txt in legend.get_texts():
    txt.set_fontweight("bold")

# Figure Title
plt.title("Effect of Tree Depth on XGBoost Performance (C14 Alloy Screening Raw Dataset)",
          fontsize=32, weight="bold", family="Times New Roman", pad=30)

plt.tight_layout(pad=3.0)
# Export vector PDF + high-res PNG
plt.savefig("xgb_depth_overfit.png", dpi=600, bbox_inches="tight")
plt.show()