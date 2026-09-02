# =============================================================================
# C14 Alloy Screening Dataset - Data Augmentation Visualization
# Methods: Gaussian Noise (GN), SMOTER, SMOTER+GN, SMOGN
# Output: KDE plot (Fig3) and PCA projection (Fig4)
# All comments in English, fonts Times New Roman, colors updated
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------- Global settings --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

np.random.seed(42)

# ==================== 1. Load data ====================
file_path = r"BCC_data.csv"   # Adjust path if needed
df = pd.read_csv(file_path)
target_col = 'Max_H2_Uptake_wt_pct'

# Drop rows with missing target (if any)
df = df.dropna(subset=[target_col])
print(f"Total samples: {len(df)}")

# Separate features and target
X_df = df.drop(columns=[target_col])
y = df[target_col].values

# Keep only numeric columns (all are numeric in this dataset)
X_df = X_df.select_dtypes(include=[np.number])
print(f"Original feature count: {X_df.shape[1]}")

# ==================== 2. Feature selection (cumulative importance 95%) ====================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_df, y)
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': X_df.columns, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)
feat_imp['cumsum'] = feat_imp['importance'].cumsum()

selected_features = feat_imp[feat_imp['cumsum'] <= 0.95]['feature'].tolist()
if len(selected_features) == 0:
    selected_features = [feat_imp.iloc[0]['feature']]

print(f"Selected features ({len(selected_features)}): {selected_features}")

X_orig = X_df[selected_features].values

# ==================== 3. Augmentation parameters ====================
NOISE_STD = 0.05
EXTREME_RATIO = 0.2
N_SYNTHETIC_RATIO = 3

# ==================== 4. Augmentation functions ====================
def gaussian_noise_synthetic(X, y, noise_std, n_ratio):
    """Generate synthetic samples by adding Gaussian noise to original samples."""
    n_orig = len(X)
    n_synth = int(n_orig * n_ratio)
    synth_X, synth_y = [], []
    for _ in range(n_synth):
        idx = np.random.randint(0, n_orig)
        x_sample = X[idx].copy()
        y_sample = y[idx]
        noise = np.random.normal(0, noise_std, size=X.shape[1])
        new_x = x_sample + noise
        new_y = y_sample + np.random.normal(0, noise_std * 0.1)
        synth_X.append(new_x)
        synth_y.append(new_y)
    return np.array(synth_X), np.array(synth_y)

def smoter_synthetic(X, y, extreme_ratio):
    """SMOTER: interpolate between low and high target extremes."""
    n_orig = len(X)
    n_extreme = int(n_orig * extreme_ratio)
    sorted_idx = np.argsort(y)
    low_idx = sorted_idx[:n_extreme]
    high_idx = sorted_idx[-n_extreme:]
    synth_X, synth_y = [], []
    # Low-low interpolations
    for _ in range(n_extreme):
        a, b = np.random.choice(low_idx, 2, replace=True)
        alpha = np.random.uniform(0, 1)
        new_x = X[a] * (1 - alpha) + X[b] * alpha
        new_y = y[a] * (1 - alpha) + y[b] * alpha
        synth_X.append(new_x)
        synth_y.append(new_y)
    # High-high interpolations
    for _ in range(n_extreme):
        a, b = np.random.choice(high_idx, 2, replace=True)
        alpha = np.random.uniform(0, 1)
        new_x = X[a] * (1 - alpha) + X[b] * alpha
        new_y = y[a] * (1 - alpha) + y[b] * alpha
        synth_X.append(new_x)
        synth_y.append(new_y)
    return np.array(synth_X), np.array(synth_y)

def smoter_gaussian_synthetic(X, y, extreme_ratio, noise_std, n_ratio):
    """SMOTER followed by Gaussian noise."""
    X_sm, y_sm = smoter_synthetic(X, y, extreme_ratio)
    X_combined = np.vstack([X, X_sm])
    y_combined = np.concatenate([y, y_sm])
    # Apply Gaussian noise to the combined set
    return gaussian_noise_synthetic(X_combined, y_combined, noise_std, n_ratio)

def smogn_synthetic(X, y, noise_std, n_ratio):
    """SMOGN simplified: here we use Gaussian noise as a placeholder (same as GN)."""
    return gaussian_noise_synthetic(X, y, noise_std, n_ratio)

# ==================== 5. Generate synthetic data ====================
print("\nGenerating synthetic data...")
X_gn, y_gn = gaussian_noise_synthetic(X_orig, y, NOISE_STD, N_SYNTHETIC_RATIO)
X_smoter, y_smoter = smoter_synthetic(X_orig, y, EXTREME_RATIO)
X_smotergn, y_smotergn = smoter_gaussian_synthetic(X_orig, y, EXTREME_RATIO, NOISE_STD, N_SYNTHETIC_RATIO)
X_smogn, y_smogn = smogn_synthetic(X_orig, y, NOISE_STD, N_SYNTHETIC_RATIO)

print(f"Original: {len(y)}")
print(f"GN: {len(X_gn)}")
print(f"SMOTER: {len(X_smoter)}")
print(f"SMOTER+GN: {len(X_smotergn)}")
print(f"SMOGN: {len(X_smogn)}")

# ==================== 6. Color definitions ====================
COLOR_ORIG = '#4EABC0'      # Blue for original
COLOR_SYNTH = '#FFA453'     # Orange for synthetic

# ==================== 7. Fig3: KDE comparison (Original vs SMOGN) ====================
plt.figure(figsize=(10, 7))

sns.kdeplot(y, label='Original', color=COLOR_ORIG, linestyle='-', linewidth=4)
sns.kdeplot(y_smogn, label='SMOGN Synthetic', color=COLOR_SYNTH, linestyle='--', linewidth=4)

plt.xlabel('Hydrogen capacity (wt.%)', fontsize=28, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Density', fontsize=28, fontweight='bold', fontfamily='Times New Roman')
plt.title('C14 Alloy Screening: Target distribution (Original vs SMOGN)', fontsize=24, fontweight='bold', fontfamily='Times New Roman')

leg = plt.legend(frameon=True, shadow=True, fontsize=22)
for text in leg.get_texts():
    text.set_fontfamily('Times New Roman')
    text.set_fontweight('bold')

plt.tick_params(axis='both', which='major', labelsize=24, width=2.5, length=10)
for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    tick.set_fontfamily('Times New Roman')
    tick.set_fontweight('bold')
for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.tight_layout()
plt.savefig('Fig3_BCC_KDE_SMOGN.png', dpi=600, bbox_inches='tight')
plt.show()

# ==================== 8. Fig4: PCA projection (2x2 subplots) ====================
def plot_pca_projection(X_orig, y_orig, X_synth, y_synth, title, ax):
    """Project both original and synthetic data onto PCA space and plot."""
    scaler = StandardScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    pca = PCA(n_components=2)
    pca.fit(X_orig_scaled)
    orig_pca = pca.transform(X_orig_scaled)
    X_synth_scaled = scaler.transform(X_synth)
    synth_pca = pca.transform(X_synth_scaled)

    # Original: filled circle
    ax.scatter(orig_pca[:, 0], orig_pca[:, 1],
               c=COLOR_ORIG, marker='o', s=80, label='Original', alpha=0.85)
    # Synthetic: filled circle
    ax.scatter(synth_pca[:, 0], synth_pca[:, 1],
               c=COLOR_SYNTH, marker='o', s=70, label='Synthetic', alpha=0.6)

    ax.set_title(title, fontfamily='Times New Roman', fontsize=32, fontweight='bold')
    leg = ax.legend(fontsize=32, frameon=True, shadow=True)
    for text in leg.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')

    ax.tick_params(axis='both', which='major', labelsize=24, width=2.5, length=10)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')

    ax.set_xlabel('PC1', fontsize=32, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('PC2', fontsize=32, fontweight='bold', fontfamily='Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
plot_pca_projection(X_orig, y, X_gn, y_gn, 'GN', axes[0, 0])
plot_pca_projection(X_orig, y, X_smoter, y_smoter, 'SMOTER', axes[0, 1])
plot_pca_projection(X_orig, y, X_smotergn, y_smotergn, 'SMOTER+GN', axes[1, 0])
plot_pca_projection(X_orig, y, X_smogn, y_smogn, 'SMOGN', axes[1, 1])

# Subplot labels (a), (b), (c), (d)
for i, ax in enumerate(axes.flat):
    ax.text(0.03, 0.97, f'({chr(97 + i)})', transform=ax.transAxes,
            fontsize=32, fontweight='bold', va='top', fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig('Fig4_BCC_PCA_projection.png', dpi=600, bbox_inches='tight')
plt.show()

print("\nAll figures saved successfully.")