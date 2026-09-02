"""
BCC Phase Candidate Screening - Data-Driven Composition Window

Modified to produce 3 distinct candidate compositions with decimal values.
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
model_path = "bcc_xgboost_smogn_best.pkl"
best_model = joblib.load(model_path)
print(f"Model loaded: {type(best_model).__name__}")

features_path = "bcc_selected_features.pkl"
selected_features = joblib.load(features_path)
print(f"Selected features loaded: {len(selected_features)} features")

stats_path = "bcc_train_stats.pkl"
train_stats = joblib.load(stats_path)
print("Training stats loaded")

# ====================== Expanded Composition Window ======================
# Based on high-capacity samples (>3.0 wt%) but with wider ranges to allow diversity
ELEMENT_WINDOW = {
    'Element_at_pct_Ti': (32, 42),
    'Element_at_pct_V': (28, 36),
    'Element_at_pct_Cr': (7, 13),
    'Element_at_pct_Fe': (4, 16),
    'Element_at_pct_Mn': (4, 16),
}

# Elements to exclude
EXCLUDED_ELEMENTS = ['Element_at_pct_Zr', 'Element_at_pct_Al', 'Element_at_pct_Ni']

print("\nComposition window (expanded for diversity):")
for elem, (min_val, max_val) in ELEMENT_WINDOW.items():
    print(f"  {elem}: {min_val:.1f} - {max_val:.1f}")
print(f"  Fe + Mn: <= 22%")
print(f"  Excluded: {EXCLUDED_ELEMENTS}")

# ====================== Fixed Experimental Parameters ======================
FIXED_PARAMS = {
    'Unit_Cell_Volume_Å3': 30.0,
    'Lattice_Constant_Å': 3.1,
    'Test_Temperature_K': 298,
    'Initial_Hydrogen_Pressure_MPa': 3.0,
    'Hydrogen_Absorption_Cycles': 1,
}

print("\nFixed experimental parameters:")
for param, value in FIXED_PARAMS.items():
    print(f"  {param}: {value}")

# ====================== Constraint Checking ======================
def check_constraints(row):
    """Check if candidate falls within the composition window."""
    ti = row.get('Element_at_pct_Ti', 0)
    v = row.get('Element_at_pct_V', 0)
    cr = row.get('Element_at_pct_Cr', 0)
    fe = row.get('Element_at_pct_Fe', 0)
    mn = row.get('Element_at_pct_Mn', 0)

    if not (32 <= ti <= 42):
        return False
    if not (28 <= v <= 36):
        return False
    if not (7 <= cr <= 13):
        return False
    if not (4 <= fe <= 16):
        return False
    if not (4 <= mn <= 16):
        return False
    if fe + mn > 22:
        return False
    if ti + v < 60:
        return False

    # Excluded elements must be zero
    for elem in EXCLUDED_ELEMENTS:
        if row.get(elem, 0) > 0.5:
            return False

    return True

# ====================== Candidate Generation ======================
def generate_candidates(element_window, excluded_elements, n_candidates=50000, step=2):
    """Generate candidate compositions with expanded window for diversity."""
    elements = list(element_window.keys())

    # Build ranges
    ranges = {}
    for elem, (min_val, max_val) in element_window.items():
        ranges[elem] = np.arange(min_val, max_val + step, step)

    candidates = []
    iteration = 0
    max_iterations = n_candidates * 3

    print(f"\nGenerating candidates (step={step})...")

    while len(candidates) < n_candidates and iteration < max_iterations:
        iteration += 1

        ti = np.random.choice(ranges['Element_at_pct_Ti'])
        v = np.random.choice(ranges['Element_at_pct_V'])
        cr = np.random.choice(ranges['Element_at_pct_Cr'])
        fe = np.random.choice(ranges['Element_at_pct_Fe'])
        mn = np.random.choice(ranges['Element_at_pct_Mn'])

        if fe + mn > 22:
            continue
        if ti + v < 60:
            continue

        total = ti + v + cr + fe + mn

        if 95 <= total <= 105:
            normalized = {
                'Element_at_pct_Ti': round(ti / total * 100, 1),
                'Element_at_pct_V': round(v / total * 100, 1),
                'Element_at_pct_Cr': round(cr / total * 100, 1),
                'Element_at_pct_Fe': round(fe / total * 100, 1),
                'Element_at_pct_Mn': round(mn / total * 100, 1),
            }

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
    """Format composition with optional decimals."""
    element_order = ['Element_at_pct_Ti', 'Element_at_pct_V', 'Element_at_pct_Cr',
                     'Element_at_pct_Fe', 'Element_at_pct_Mn']
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

def get_composition_string(row):
    """Get detailed composition string for display."""
    details = []
    for col in ['Element_at_pct_Ti', 'Element_at_pct_V', 'Element_at_pct_Cr',
                'Element_at_pct_Fe', 'Element_at_pct_Mn']:
        if col in row and row[col] > 0.5:
            details.append(f"{col.replace('Element_at_pct_','')}={row[col]:.1f}")
    return ', '.join(details)

def is_distinct(candidate, existing_candidates, tol=1.0):
    """Check if candidate is distinct from existing ones."""
    if not existing_candidates:
        return True
    for existing in existing_candidates:
        diff = sum(abs(candidate.get(k, 0) - existing.get(k, 0))
                   for k in ['Element_at_pct_Ti', 'Element_at_pct_V',
                            'Element_at_pct_Cr', 'Element_at_pct_Fe',
                            'Element_at_pct_Mn'])
        if diff < tol:
            return False
    return True

# ====================== Main ======================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BCC Phase Candidate Screening (Expanded Window)")
    print("=" * 70)

    # Generate candidates
    print("\n[1] Generating candidate compositions...")
    candidate_df = generate_candidates(
        element_window=ELEMENT_WINDOW,
        excluded_elements=EXCLUDED_ELEMENTS,
        n_candidates=50000,
        step=3
    )

    if len(candidate_df) == 0:
        print("ERROR: No candidates generated.")
        exit(1)

    print(f"\nGenerated {len(candidate_df)} candidates")

    # Predict
    print("\n[2] Predicting with XGBoost model...")
    X = assemble_features(candidate_df, FIXED_PARAMS, selected_features)
    y_pred = best_model.predict(X)

    results = candidate_df.copy()
    results['Predicted_Capacity_wt'] = y_pred
    results = results.sort_values('Predicted_Capacity_wt', ascending=False)

    # ===== Select top 3 distinct candidates =====
    distinct_results = []
    for _, row in results.iterrows():
        if is_distinct(row, distinct_results, tol=1.0):
            distinct_results.append(row)
        if len(distinct_results) >= 3:
            break

    print(f"\nFound {len(distinct_results)} distinct candidates")

    # Display results
    print("\n" + "=" * 70)
    print("Top 3 Distinct BCC Candidates")
    print("=" * 70)

    for idx, row in enumerate(distinct_results, 1):
        comp_str = format_candidate(row, decimals=1)
        details = get_composition_string(row)
        cap = row['Predicted_Capacity_wt']
        print(f"{idx}. {comp_str}")
        print(f"   Composition: {details}")
        print(f"   Predicted Capacity: {cap:.3f} wt%")
        print()

    # Table 5 format
    print("=" * 70)
    print("Table 5 Format: Top 3 Distinct BCC Candidates")
    print("=" * 70)
    print(f"{'Rank':<6} {'Composition':<35} {'Capacity (wt%)':<15} {'Key Features'}")
    print("-" * 70)

    for idx, row in enumerate(distinct_results, 1):
        comp_str = format_candidate(row, decimals=0)
        cap = row['Predicted_Capacity_wt']
        v = row.get('Element_at_pct_V', 0)
        fe = row.get('Element_at_pct_Fe', 0)
        mn = row.get('Element_at_pct_Mn', 0)
        print(f"{idx:<6} {comp_str:<35} {cap:.2f}            V~{v:.0f}%, Fe+Mn={fe+mn:.0f}%")

    # Save
    results.to_csv("bcc_candidates_window.csv", index=False)
    print("\nResults saved to: bcc_candidates_window.csv")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total candidates generated: {len(candidate_df)}")
    if distinct_results:
        print(f"Top distinct candidate: {format_candidate(distinct_results[0], decimals=1)}")
        print(f"Top distinct capacity: {distinct_results[0]['Predicted_Capacity_wt']:.3f} wt%")

    print("\n✅ Screening completed!")