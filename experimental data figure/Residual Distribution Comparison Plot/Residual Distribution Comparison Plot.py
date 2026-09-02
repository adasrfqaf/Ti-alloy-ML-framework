"""
Residual scatter plot for three phases (C14 Alloy Screening, TiFe, C14).
High-resolution PNG output only.
Connecting lines between corresponding samples are retained.
Grid lines are removed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ====================== Global font settings ======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ====================== High-resolution output settings ======================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05


def load_data(split_file, ours_file, phase_label):
    """
    Load two CSV files, unify column names, and add model labels.

    Parameters
    ----------
    split_file : str
        Path to the "Split training" CSV file.
    ours_file : str
        Path to the "Proposed" CSV file.
    phase_label : str
        Phase name (used only for printing).

    Returns
    -------
    combined : pd.DataFrame
        Combined DataFrame with columns 'true_capacity', 'pred', 'model'.
    """
    split_df = pd.read_csv(split_file)
    ours_df = pd.read_csv(ours_file)

    # Rename columns: first column is true_capacity, second is prediction
    split_df = split_df.rename(columns={split_df.columns[0]: 'true_capacity',
                                        split_df.columns[1]: 'pred'})
    ours_df = ours_df.rename(columns={ours_df.columns[0]: 'true_capacity',
                                      ours_df.columns[1]: 'pred'})

    split_df['model'] = 'Split training'
    ours_df['model'] = 'Proposed'

    combined = pd.concat([split_df, ours_df], ignore_index=True)
    print(f"{phase_label} total samples: {len(combined)} (Split={len(split_df)}, Proposed={len(ours_df)})")
    return combined


def plot_combined_residual(combined_df, phase_label, output_base):
    """
    Plot residual scatter plot with two models.

    Connecting lines between corresponding samples are shown in gray dashes.
    Statistical box is always in the upper-right corner.
    Legend placement: C14 -> lower right, C14 Alloy Screening and TiFe -> upper left.
    """
    combined_df['residual'] = combined_df['true_capacity'] - combined_df['pred']
    combined_df = combined_df.sort_values('true_capacity').reset_index(drop=True)

    split_data = combined_df[combined_df['model'] == 'Split training']
    ours_data = combined_df[combined_df['model'] == 'Proposed']

    split_median = split_data['residual'].median()
    ours_median = ours_data['residual'].median()
    split_iqr = split_data['residual'].quantile(0.75) - split_data['residual'].quantile(0.25)
    ours_iqr = ours_data['residual'].quantile(0.75) - ours_data['residual'].quantile(0.25)

    print(f"  Split training: median={split_median:.4f}, IQR={split_iqr:.4f}")
    print(f"  Proposed:       median={ours_median:.4f}, IQR={ours_iqr:.4f}")

    # ===== Large figure =====
    fig, ax = plt.subplots(figsize=(15, 9))

    x_split = split_data.index.values
    x_ours = ours_data.index.values

    # ----- Scatter points (large size) with specified colors -----
    ax.scatter(x_split, split_data['residual'],
               color='#4EA8C0', s=120, marker='o',
               label='Split training', alpha=0.85, edgecolors='none')

    ax.scatter(x_ours, ours_data['residual'],
               color='#FFA453', s=120, marker='o',
               label='Proposed', alpha=0.85, edgecolors='none')

    # ----- Connecting lines for common true capacities (gray dashed) -----
    common_caps = set(split_data['true_capacity']) & set(ours_data['true_capacity'])
    for cap in common_caps:
        split_vals = split_data[split_data['true_capacity'] == cap]['residual'].values
        ours_vals = ours_data[ours_data['true_capacity'] == cap]['residual'].values
        if len(split_vals) > 0 and len(ours_vals) > 0:
            x1 = split_data[split_data['true_capacity'] == cap].index[0]
            x2 = ours_data[ours_data['true_capacity'] == cap].index[0]
            ax.plot([x1, x2], [split_vals[0], ours_vals[0]],
                    color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

    # ----- Zero line (red dashed) -----
    ax.axhline(y=0, color='red', linestyle='--', linewidth=3.0, alpha=0.8)

    # ===== Legend location based on phase =====
    if phase_label == 'C14':
        legend_loc = 'lower right'
    else:
        legend_loc = 'upper left'

    # ----- Statistical information box (always upper right) -----
    stats_text = (
        f"Split training: Median={split_median:.3f} wt.%, IQR={split_iqr:.3f}\n"
        f"Proposed:        Median={ours_median:.3f} wt.%, IQR={ours_iqr:.3f}"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=24, verticalalignment='top', horizontalalignment='right',
            fontfamily='Times New Roman', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      alpha=0.95, edgecolor='black', linewidth=2))

    # ----- Axis labels (large font) -----
    ax.set_xlabel(f'{phase_label} phase test samples (sorted by capacity)',
                  fontsize=32, fontfamily='Times New Roman', fontweight='bold')
    ax.set_ylabel('Residual (wt.%)',
                  fontsize=32, fontfamily='Times New Roman', fontweight='bold')

    # ----- Legend -----
    legend = ax.legend(loc=legend_loc, fontsize=24, frameon=True,
                       edgecolor='black', fancybox=False,
                       handletextpad=0.8, markerscale=1.5)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')
    legend.get_frame().set_linewidth(2)

    # ----- Ticks (large font) -----
    ax.tick_params(axis='both', labelsize=24, width=2.5, length=10)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily('Times New Roman')
        tick_label.set_fontweight('bold')

    # Axis spines width
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # Grid lines are removed (no ax.grid() call)

    plt.tight_layout(pad=3.0)

    # ===== Save only PNG (no PDF, no SVG) =====
    png_file = f'{output_base}.png'
    plt.savefig(png_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ PNG saved: {png_file}")

    plt.show()
    plt.close()


# ============================================================
# Main program
# ============================================================
# Define file mappings for each phase
FILE_MAPPING = {
    'C14 Alloy Screening': {
        'split': 'test_predictions_raw_BCC.csv',
        'ours': 'test_predictions_SMOGN_BCC.csv'
    },
    'TiFe': {
        'split': 'test_predictions_raw_TiFe.csv',
        'ours': 'test_predictions_SMOGN_TiFe.csv'
    },
    'C14': {
        'split': 'test_predictions_raw_C14.csv',
        'ours': 'test_predictions_GN_C14.csv'
    }
}

OUTPUTS = {
    'C14 Alloy Screening': 'Figure6_BCC_Residual',
    'TiFe': 'FigureS3a_TiFe_Residual',
    'C14': 'FigureS3b_C14_Residual'
}

print("=" * 60)
print("Residual scatter plot (high-resolution PNG output)")
print("Comparing 'Split training' (raw) vs 'Proposed' (custom)")
print("Connecting lines between same samples are shown.")
print("=" * 60)

for phase in ['C14 Alloy Screening', 'TiFe', 'C14']:
    print(f"\nProcessing {phase}...")
    try:
        split_file = FILE_MAPPING[phase]['split']
        ours_file = FILE_MAPPING[phase]['ours']
        combined_df = load_data(split_file, ours_file, phase)
        plot_combined_residual(combined_df, phase, OUTPUTS[phase])
    except FileNotFoundError as e:
        print(f"❌ File not found: {e.filename}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\nDone! Only PNG files are saved.")