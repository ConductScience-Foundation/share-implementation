"""
Consistency Validation: Do "consistent sharers" get more reuse than "spike sharers"
at the same S-Index level?

Key question: At the same S-Index, does a researcher with high avg SHARE score
(consistent) get more citations/derivatives than one with high max but low avg (spiky)?

Uses Zenodo DuckDB (1.3M records).
"""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import os
import warnings
warnings.filterwarnings('ignore')

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.environ.get('SHARE_DATA_DIR', os.path.join(_PROJECT_DIR, 'data'))
DB_PATH = os.path.join(_DATA_DIR, 'zenodo_tiered_full.db')
OUTPUT_DIR = os.environ.get('SHARE_OUTPUT_DIR', _DATA_DIR)

print("=" * 70)
print("  CONSISTENCY VALIDATION ANALYSIS")
print("  Do consistent sharers get more reuse than spike sharers?")
print("=" * 70)

conn = duckdb.connect(DB_PATH, read_only=True)

# Step 1: Build researcher-level dataset with outcomes
print("\n[1/5] Building researcher profiles with outcomes...")

df = conn.execute("""
WITH author_exploded AS (
    SELECT
        unnest(authors) as author_raw,
        share_score,
        stewardship_score,
        harmonization_score,
        access_score,
        reuse_score,
        engagement_score,
        citation_count,
        resource_type,
        doi,
        CASE WHEN related_identifiers THEN 1 ELSE 0 END as has_related,
        CASE WHEN supplement_tie THEN 1 ELSE 0 END as has_derivative
    FROM records
    WHERE authors IS NOT NULL
      AND share_score IS NOT NULL
      AND resource_type = 'dataset'
),
researcher_stats AS (
    SELECT
        author_raw,
        COUNT(*) as n_datasets,
        AVG(share_score) as avg_share,
        MAX(share_score) as max_share,
        MIN(share_score) as min_share,
        STDDEV_POP(share_score) as std_share,
        MEDIAN(share_score) as median_share,
        -- Outcomes
        SUM(COALESCE(citation_count, 0)) as total_citations,
        MAX(COALESCE(citation_count, 0)) as max_citations,
        AVG(COALESCE(citation_count, 0)) as avg_citations,
        SUM(has_related) as n_related,
        SUM(has_derivative) as n_derivatives,
        CASE WHEN SUM(COALESCE(citation_count, 0)) > 0 THEN 1 ELSE 0 END as has_any_citation,
        CASE WHEN SUM(has_derivative) > 0 THEN 1 ELSE 0 END as has_any_derivative,
        -- Component averages
        AVG(stewardship_score) as avg_S,
        AVG(harmonization_score) as avg_H,
        AVG(access_score) as avg_A,
        AVG(reuse_score) as avg_R,
        AVG(engagement_score) as avg_E
    FROM author_exploded
    GROUP BY author_raw
    HAVING COUNT(*) >= 3
)
SELECT * FROM researcher_stats
""").fetchdf()

print(f"   Researchers with >= 3 datasets: {len(df):,}")

# Step 2: Compute S-Index and consistency metrics
print("\n[2/5] Computing S-Index and consistency metrics...")

def compute_s_index_from_row(avg, max_val, n):
    """Approximate S-Index. For exact, we'd need per-dataset scores."""
    # Use the H-index style: approximate from avg and n
    # S-Index = k where k datasets score >= k
    # Upper bound: min(n, max_val)
    # With avg_share and n, approximate
    k = 0
    for i in range(1, int(min(n, max_val)) + 1):
        # Estimate how many datasets score >= i
        # Rough: if avg >= i, likely most do; if avg < i, fewer do
        if avg >= i:
            k = i
        else:
            break
    return k

# Consistency ratio: avg/max (1.0 = perfectly consistent, low = spiky)
df['consistency_ratio'] = df['avg_share'] / df['max_share'].clip(lower=1)
df['score_range'] = df['max_share'] - df['min_share']
df['cv'] = df['std_share'] / df['avg_share'].clip(lower=1)  # coefficient of variation

# Approximate S-Index
df['s_index_approx'] = df.apply(
    lambda r: compute_s_index_from_row(r['avg_share'], r['max_share'], r['n_datasets']), axis=1
)

# Classify into consistency groups
df['consistency_group'] = pd.cut(
    df['consistency_ratio'],
    bins=[0, 0.5, 0.7, 0.85, 1.01],
    labels=['Very Spiky (<0.5)', 'Spiky (0.5-0.7)', 'Moderate (0.7-0.85)', 'Consistent (>0.85)']
)

print(f"   Consistency ratio: mean={df['consistency_ratio'].mean():.3f}, median={df['consistency_ratio'].median():.3f}")
print(f"   Score range: mean={df['score_range'].mean():.1f}, median={df['score_range'].median():.1f}")

# Step 3: Compare outcomes by consistency group
print("\n[3/5] Comparing outcomes by consistency group...")

# Filter to researchers with similar S-Index ranges for fair comparison
# Use S-Index bins to control for volume/quality level
df['s_index_bin'] = pd.cut(df['s_index_approx'], bins=[0, 10, 20, 30, 50, 200], labels=['1-10', '11-20', '21-30', '31-50', '51+'])

print("\n--- OVERALL: Outcomes by Consistency Group ---")
print(f"{'Group':<25} {'n':>8} {'AvgCit':>8} {'%Cited':>8} {'%Deriv':>8} {'AvgSHARE':>9} {'AvgMax':>8}")
print("-" * 80)

for group in ['Very Spiky (<0.5)', 'Spiky (0.5-0.7)', 'Moderate (0.7-0.85)', 'Consistent (>0.85)']:
    subset = df[df['consistency_group'] == group]
    if len(subset) == 0:
        continue
    n = len(subset)
    avg_cit = subset['avg_citations'].mean()
    pct_cited = subset['has_any_citation'].mean() * 100
    pct_deriv = subset['has_any_derivative'].mean() * 100
    avg_share = subset['avg_share'].mean()
    avg_max = subset['max_share'].mean()
    print(f"{group:<25} {n:>8,} {avg_cit:>8.2f} {pct_cited:>7.1f}% {pct_deriv:>7.1f}% {avg_share:>9.1f} {avg_max:>8.1f}")

# Step 4: Control for S-Index level (the key analysis)
print("\n--- CONTROLLED: Outcomes by Consistency, Within S-Index Bins ---")
print("(This controls for researcher productivity/quality level)")

results_rows = []
for s_bin in ['1-10', '11-20', '21-30', '31-50', '51+']:
    bin_df = df[df['s_index_bin'] == s_bin]
    if len(bin_df) < 20:
        continue

    print(f"\n  S-Index {s_bin} (n={len(bin_df):,}):")
    print(f"  {'Group':<25} {'n':>7} {'AvgCit':>8} {'%Cited':>8} {'%Deriv':>8} {'TotCit':>8}")
    print(f"  {'-'*68}")

    for group in ['Very Spiky (<0.5)', 'Spiky (0.5-0.7)', 'Moderate (0.7-0.85)', 'Consistent (>0.85)']:
        subset = bin_df[bin_df['consistency_group'] == group]
        if len(subset) < 5:
            continue
        n = len(subset)
        avg_cit = subset['avg_citations'].mean()
        pct_cited = subset['has_any_citation'].mean() * 100
        pct_deriv = subset['has_any_derivative'].mean() * 100
        tot_cit = subset['total_citations'].mean()
        print(f"  {group:<25} {n:>7,} {avg_cit:>8.2f} {pct_cited:>7.1f}% {pct_deriv:>7.1f}% {tot_cit:>8.1f}")

        results_rows.append({
            's_index_bin': s_bin,
            'consistency_group': group,
            'n': n,
            'avg_citations': avg_cit,
            'pct_cited': pct_cited,
            'pct_derivative': pct_deriv,
            'total_citations': tot_cit,
        })

# Step 5: Statistical tests
print("\n\n[4/5] Statistical tests...")

# Test 1: Correlation between consistency ratio and citations (controlling for S-Index)
print("\n--- Partial correlation: consistency_ratio -> citations, controlling for s_index ---")

# Simple correlations first
r_cons_cit, p_cons_cit = sp_stats.pearsonr(df['consistency_ratio'], df['total_citations'])
r_cons_deriv, p_cons_deriv = sp_stats.pointbiserialr(df['has_any_derivative'], df['consistency_ratio'])
r_avg_cit, p_avg_cit = sp_stats.pearsonr(df['avg_share'], df['total_citations'])

print(f"  consistency_ratio vs total_citations:  r={r_cons_cit:.4f}, p={p_cons_cit:.2e}")
print(f"  consistency_ratio vs has_derivative:   r={r_cons_deriv:.4f}, p={p_cons_deriv:.2e}")
print(f"  avg_share vs total_citations:          r={r_avg_cit:.4f}, p={p_avg_cit:.2e}")

# Partial correlation (residualize on S-Index)
from numpy.linalg import lstsq

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    # Residualize x on z
    A = np.column_stack([z, np.ones(len(z))])
    coef_x, _, _, _ = lstsq(A, x, rcond=None)
    resid_x = x - A @ coef_x
    coef_y, _, _, _ = lstsq(A, y, rcond=None)
    resid_y = y - A @ coef_y
    return sp_stats.pearsonr(resid_x, resid_y)

valid = df.dropna(subset=['consistency_ratio', 'total_citations', 's_index_approx'])
x = valid['consistency_ratio'].values
y = valid['total_citations'].values
z = valid['s_index_approx'].values

r_partial, p_partial = partial_corr(x, y, z)
print(f"\n  Partial corr (consistency -> citations | S-Index): r={r_partial:.4f}, p={p_partial:.2e}")

y_deriv = valid['has_any_derivative'].values.astype(float)
r_partial_d, p_partial_d = partial_corr(x, y_deriv, z)
print(f"  Partial corr (consistency -> derivative | S-Index): r={r_partial_d:.4f}, p={p_partial_d:.2e}")

# Test 2: Logistic regression - does consistency predict citation after controlling for S-Index?
print("\n--- Logistic regression: has_citation ~ consistency_ratio + s_index ---")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    valid2 = df.dropna(subset=['consistency_ratio', 'has_any_citation', 's_index_approx', 'n_datasets'])
    X = valid2[['consistency_ratio', 's_index_approx', 'n_datasets']].values
    y_cit = valid2['has_any_citation'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, y_cit)

    feature_names = ['consistency_ratio', 's_index_approx', 'n_datasets']
    print(f"  {'Feature':<25} {'Coef':>8} {'Odds Ratio':>12}")
    print(f"  {'-'*50}")
    for name, coef in zip(feature_names, lr.coef_[0]):
        or_val = np.exp(coef)
        print(f"  {name:<25} {coef:>8.4f} {or_val:>12.4f}")
    print(f"  Accuracy: {lr.score(X_scaled, y_cit):.4f}")

    # Same for derivatives
    print("\n--- Logistic regression: has_derivative ~ consistency_ratio + s_index ---")
    y_der = valid2['has_any_derivative'].values
    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(X_scaled, y_der)

    print(f"  {'Feature':<25} {'Coef':>8} {'Odds Ratio':>12}")
    print(f"  {'-'*50}")
    for name, coef in zip(feature_names, lr2.coef_[0]):
        or_val = np.exp(coef)
        print(f"  {name:<25} {coef:>8.4f} {or_val:>12.4f}")
    print(f"  Accuracy: {lr2.score(X_scaled, y_der):.4f}")

except ImportError:
    print("  sklearn not available, skipping logistic regression")

# Test 3: Within S-Index bin, t-test between top and bottom consistency quartiles
print("\n--- Within-bin t-tests: Top vs Bottom consistency quartile ---")
for s_bin in ['11-20', '21-30', '31-50']:
    bin_df = df[df['s_index_bin'] == s_bin].dropna(subset=['consistency_ratio', 'total_citations'])
    if len(bin_df) < 40:
        continue

    q25 = bin_df['consistency_ratio'].quantile(0.25)
    q75 = bin_df['consistency_ratio'].quantile(0.75)

    low_cons = bin_df[bin_df['consistency_ratio'] <= q25]['total_citations']
    high_cons = bin_df[bin_df['consistency_ratio'] >= q75]['total_citations']

    if len(low_cons) < 5 or len(high_cons) < 5:
        continue

    t, p = sp_stats.mannwhitneyu(high_cons, low_cons, alternative='greater')

    print(f"\n  S-Index {s_bin}:")
    print(f"    Low consistency  (n={len(low_cons):,}): median citations = {low_cons.median():.1f}, mean = {low_cons.mean():.1f}")
    print(f"    High consistency (n={len(high_cons):,}): median citations = {high_cons.median():.1f}, mean = {high_cons.mean():.1f}")
    print(f"    Mann-Whitney U (high > low): U={t:.0f}, p={p:.4f}")

    # Same for derivatives
    low_d = bin_df[bin_df['consistency_ratio'] <= q25]['has_any_derivative']
    high_d = bin_df[bin_df['consistency_ratio'] >= q75]['has_any_derivative']

    pct_low = low_d.mean() * 100
    pct_high = high_d.mean() * 100

    if low_d.sum() + high_d.sum() > 0:
        chi2_table = np.array([
            [high_d.sum(), len(high_d) - high_d.sum()],
            [low_d.sum(), len(low_d) - low_d.sum()]
        ])
        chi2, p_chi = sp_stats.chi2_contingency(chi2_table)[:2]
        print(f"    Derivative creation: high={pct_high:.1f}% vs low={pct_low:.1f}%, chi2={chi2:.2f}, p={p_chi:.4f}")

# Summary
print("\n\n[5/5] Summary...")
print("=" * 70)

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'consistency_validation_results.csv'), index=False)
print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'consistency_validation_results.csv')}")

conn.close()
print("\nDone.")
