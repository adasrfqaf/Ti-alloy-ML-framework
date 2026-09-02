"""
C14 Laves Phase Candidate Screening Script
GBDT Model with Gaussian Noise Augmentation

This script loads the trained optimal model and screens for promising
C14 Laves phase compositions within the high-capacity composition window.
"""

import pandas as pd
import numpy as np
import joblib

# ====================== Configuration ======================
SEED = 49
np.random.seed(SEED)

print("=" * 70)
print("Loading trained model and artifacts...")
print("=" * 70)

# ====================== Load Model and Artifacts ======================
model_path = "c14_gbdt_gn_best.pkl"
best_model = joblib.load(model_path)
print(f"Model loaded: {type(best_model).__name__}")

scaler_path = "c14_scaler.pkl"
scaler = joblib.load(scaler_path)
print("Scaler loaded")

features_path = "c14_selected_features.pkl"
selected_features = joblib.load(features_path)
print(f"Selected features loaded: {len(selected_features)} features")

stats_path = "c14_train_stats.pkl"
train_stats = joblib.load(stats_path)
print("Training stats loaded")

X_stats = train_stats['X_train_stats']

# ====================== Load Raw Data for High-Capacity Analysis ======================
raw_data = pd.read_csv("C14_data.csv")
print(f"\nRaw data shape: {raw_data.shape}")

# High-capacity samples (>2.5 wt%)
high_cap = raw_data[raw_data['Max_H2_Uptake_wt_pct'] > 2.5].copy()
print(f"High-capacity samples (>2.5 wt%): {len(high_cap)}")

# ====================== Define Composition Window ======================
# Based on analysis of high-capacity C14 samples (>2.5 wt%)
# Key findings:
#   - High capacity C14: Ti 38%, V 23-29%, Mn 8%, Zr 3-9%, Cr 10%, Fe 12%
#   - Also: Ti 30.5%, V 15.25%, Mn 50.85%, Zr 3.39%
#   - Window covers both sub-systems

ELEMENT_WINDOW = {
    'Element_at_pct_Ti': (28, 42),
    'Element_at_pct_V': (12, 32),
    'Element_at_pct_Mn': (6, 15),
    'Element_at_pct_Zr': (3, 12),
    'Element_at_pct_Cr': (0, 14),
    'Element_at_pct_Fe': (8, 16),
    'Element_at_pct_Nb': (0, 5),
    'Element_at_pct_Cu': (0, 5),
    'Element_at_pct_Al': (0, 5),
    'Element_at_pct_Ni': (0, 5),
    'Element_at_pct_Mo': (0, 5),
}

# Elements to exclude (based on high-capacity sample analysis)
EXCLUDED_ELEMENTS = ['Element_at_pct_Ce', 'Element_at_pct_Y']

print("\nComposition window (based on high-capacity C14 sample analysis):")
for elem, (min_val, max_val) in ELEMENT_WINDOW.items():
    print(f"  {elem}: {min_val:.1f} - {max_val:.1f}")
print(f"  Excluded: {EXCLUDED_ELEMENTS}")

# ====================== Constraint Checking ======================
def check_constraints(row):
    """Check if candidate falls within the composition window."""
    for elem, (min_val, max_val) in ELEMENT_WINDOW.items():
        val = row.get(elem, 0)
        if val < min_val or val > max_val:
            return False

    # Excluded elements must be zero
    for elem in EXCLUDED_ELEMENTS:
        if row.get(elem, 0) > 0.5:
            return False

    # Ti + V should be at least 40% (C14 phase stability)
    ti = row.get('Element_at_pct_Ti', 0)
    v = row.get('Element_at_pct_V', 0)
    if ti + v < 40:
        return False

    return True

# ====================== Fixed Experimental Parameters ======================
FIXED_PARAMS = {
    'Unit_Cell_Volume_Å3': 175.0,      # C14 typical value
    'Test_Temperature_K': 298,
    'Initial_Hydrogen_Pressure_MPa': 5.0,
    'Hydrogen_Absorption_Cycles': 1,
}

print("\nFixed experimental parameters:")
for param, value in FIXED_PARAMS.items():
    print(f"  {param}: {value}")

# ====================== Candidate Generation ======================
def generate_candidates(element_window, excluded_elements, n_candidates=50000, step=2):
    """Generate candidate compositions within the composition window."""
    elements = list(element_window.keys())

    ranges = {}
    for elem, (min_val, max_val) in element_window.items():
        ranges[elem] = np.arange(min_val, max_val + step, step)

    candidates = []
    iteration = 0
    max_iterations = n_candidates * 3

    print(f"\nGenerating candidates (step={step})...")

    while len(candidates) < n_candidates and iteration < max_iterations:
        iteration += 1

        combo = []
        for elem in elements:
            val = np.random.choice(ranges[elem])
            combo.append(val)

        total = sum(combo)

        if 95 <= total <= 105:
            normalized = {}
            for i, elem in enumerate(elements):
                normalized[elem] = round(combo[i] / total * 100, 1)

            for elem in excluded_elements:
                normalized[elem] = 0.0

            if check_constraints(normalized):
                candidates.append(normalized)

        if iteration % 100000 == 0:
            print(f"  Generated {len(candidates)} candidates...")

    print(f"Generated {len(candidates)} valid candidates")
    return pd.DataFrame(candidates)

def assemble_features(candidate_df, fixed_params, selected_features):
    """Assemble complete feature vectors."""
    rows = []
    for _, row in candidate_df.iterrows():
        feature_dict = {}
        for feat in selected_features:
            feature_dict[feat] = 0
        for col in candidate_df.columns:
            if col in selected_features:
                feature_dict[col] = row[col]
        for param, value in fixed_params.items():
            if param in selected_features:
                feature_dict[param] = value
        rows.append(feature_dict)
    df = pd.DataFrame(rows)
    df = df[selected_features]
    return df

def format_candidate(row, decimals=0):
    """Format composition into a readable string."""
    element_order = ['Element_at_pct_Ti', 'Element_at_pct_V', 'Element_at_pct_Mn',
                     'Element_at_pct_Zr', 'Element_at_pct_Cr', 'Element_at_pct_Fe',
                     'Element_at_pct_Nb', 'Element_at_pct_Cu', 'Element_at_pct_Al',
                     'Element_at_pct_Ni', 'Element_at_pct_Mo']
    comp_parts = []
    for col in element_order:
        if row.get(col, 0) > 0.5:
            elem = col.replace('Element_at_pct_', '')
            if decimals == 0:
                val = int(round(row[col]))
            else:
                val = row[col]
            if val > 0:
                if decimals == 0:
                    comp_parts.append(f"{elem}{val}")
                else:
                    comp_parts.append(f"{elem}{val:.1f}")
    return ''.join(comp_parts)

def is_distinct(candidate, existing_candidates, tol=3.0):
    """Check if candidate is distinct from existing ones."""
    if not existing_candidates:
        return True
    key_elements = ['Element_at_pct_Ti', 'Element_at_pct_V', 'Element_at_pct_Mn',
                    'Element_at_pct_Zr', 'Element_at_pct_Cr', 'Element_at_pct_Fe']
    for existing in existing_candidates:
        diff = sum(abs(candidate.get(k, 0) - existing.get(k, 0))
                   for k in key_elements if k in candidate and k in existing)
        if diff < tol:
            return False
    return True

# ====================== Main ======================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("C14 Laves Phase Candidate Screening")
    print("=" * 70)

    # Generate candidates
    print("\n[1] Generating candidate compositions...")
    candidate_df = generate_candidates(
        element_window=ELEMENT_WINDOW,
        excluded_elements=EXCLUDED_ELEMENTS,
        n_candidates=50000,
        step=1
    )

    if len(candidate_df) == 0:
        print("ERROR: No candidates generated.")
        exit(1)

    print(f"\nGenerated {len(candidate_df)} candidates")

    # Display statistics
    print("\nCandidate composition statistics:")
    for col in candidate_df.columns:
        if col.startswith('Element_at_pct_'):
            print(f"  {col}: min={candidate_df[col].min():.1f}, "
                  f"max={candidate_df[col].max():.1f}, "
                  f"mean={candidate_df[col].mean():.1f}")

    # Predict
    print("\n[2] Predicting with GBDT model...")
    X = assemble_features(candidate_df, FIXED_PARAMS, selected_features)
    X_scaled = scaler.transform(X)
    y_pred = best_model.predict(X_scaled)

    results = candidate_df.copy()
    results['Predicted_Capacity_wt'] = y_pred
    results = results.sort_values('Predicted_Capacity_wt', ascending=False)

    # Select top 3 distinct candidates
    print("\n[3] Selecting top 3 distinct candidates...")
    distinct_results = []
    for _, row in results.iterrows():
        if is_distinct(row, distinct_results, tol=3.0):
            distinct_results.append(row)
        if len(distinct_results) >= 3:
            break

    print(f"Found {len(distinct_results)} distinct candidates")

    # Display results
    print("\n" + "=" * 70)
    print("Top 3 Distinct C14 Candidates")
    print("=" * 70)

    for idx, row in enumerate(distinct_results, 1):
        comp_str = format_candidate(row, decimals=1)
        details = []
        for col in ['Element_at_pct_Ti', 'Element_at_pct_V', 'Element_at_pct_Mn',
                   'Element_at_pct_Zr', 'Element_at_pct_Cr', 'Element_at_pct_Fe']:
            if col in row and row[col] > 0.5:
                details.append(f"{col.replace('Element_at_pct_','')}={row[col]:.1f}")
        cap = row['Predicted_Capacity_wt']
        print(f"{idx}. {comp_str}")
        print(f"   Composition: {', '.join(details)}")
        print(f"   Predicted Capacity: {cap:.3f} wt%")
        print()

    # Table 5 format
    print("=" * 70)
    print("Table 5 Format: Top 3 Distinct C14 Candidates")
    print("=" * 70)
    print(f"{'Rank':<6} {'Composition':<35} {'Capacity (wt%)':<15} {'Key Features'}")
    print("-" * 70)

    for idx, row in enumerate(distinct_results, 1):
        comp_str = format_candidate(row, decimals=0)
        cap = row['Predicted_Capacity_wt']
        ti = row.get('Element_at_pct_Ti', 0)
        v = row.get('Element_at_pct_V', 0)
        mn = row.get('Element_at_pct_Mn', 0)
        features = []
        if ti > 0:
            features.append(f"Ti~{ti:.0f}%")
        if v > 0:
            features.append(f"V~{v:.0f}%")
        if mn > 0:
            features.append(f"Mn~{mn:.0f}%")
        print(f"{idx:<6} {comp_str:<35} {cap:.2f}            {', '.join(features)}")

    # Save
    results.to_csv("c14_candidates.csv", index=False)
    print("\nResults saved to: c14_candidates.csv")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total candidates generated: {len(candidate_df)}")
    if distinct_results:
        print(f"Top distinct candidate: {format_candidate(distinct_results[0], decimals=1)}")
        print(f"Top distinct capacity: {distinct_results[0]['Predicted_Capacity_wt']:.3f} wt%")

    print("\n✅ Screening completed!")