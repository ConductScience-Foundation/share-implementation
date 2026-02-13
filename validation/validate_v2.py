"""
SHARE Score v2 Validation Script
=================================
Recomputes key validation statistics using the NEW v2 scoring methodology
on the Zenodo 2016 cohort. Produces:

  1. Reuse prediction (logistic regression): OR, AUC, Cohen's d
  2. Citation correlation: Pearson r with log(citations+1)
  3. Per-signal citation boost: median citations present vs absent
  4. Model comparison: Full SHARE vs simple field count (AUC)
  5. Cross-repo validation: known-groups validity check

Usage:
  py -3.14 scripts/validate_v2.py

Output:
  data/validation_v2.json
"""

import os
import sys
import json
import math
import statistics
from collections import defaultdict

import duckdb
import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.environ.get('SHARE_DATA_DIR',
    os.path.join(PROJECT_DIR, 'data'))

# ---------------------------------------------------------------------------
# Import pledge definitions and condition helpers from rescore_unified.py
# (Copied here verbatim so script is self-contained and reproducible)
# ---------------------------------------------------------------------------

def _bool(v):
    """Check if value is truthy boolean."""
    return v is True or v == 1

def _nonempty(v):
    """Check if value is non-null and non-empty string."""
    if v is None:
        return False
    s = str(v).strip()
    return len(s) > 0 and s.lower() not in ('none', 'null', '')

def _array_nonempty(v):
    """Check if value is a non-empty list/array."""
    if v is None:
        return False
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    s = str(v).strip()
    return len(s) > 0 and s not in ('[]', 'None', 'null', '')

def _gt(threshold):
    """Return a condition function: value > threshold."""
    def check(v):
        if v is None:
            return False
        try:
            return float(v) > threshold
        except (ValueError, TypeError):
            return False
    return check

def _strlen_gt(threshold):
    """Return a condition function: string length > threshold."""
    def check(v):
        if v is None:
            return False
        return len(str(v)) > threshold
    return check

def _eq(target):
    """Return a condition function: value == target."""
    def check(v):
        if v is None:
            return False
        return str(v).strip().lower() == str(target).lower()
    return check

def _contains(target):
    """Return a condition function: target substring in value."""
    def check(v):
        if v is None:
            return False
        return target.lower() in str(v).lower()
    return check


# --- ZENODO PLEDGE (21 signals) — from rescore_unified.py ---
ZENODO_PLEDGE = [
    # S (6)
    ('Keywords',              'keywords',              _array_nonempty,   'S'),
    ('Contributors',          'contributors',          _bool,             'S'),
    ('Language',              'language',               _nonempty,         'S'),
    ('Subjects',              'subjects',              _bool,             'S'),
    ('Dates',                 'dates',                 _bool,             'S'),
    ('Locations',             'locations',             _bool,             'S'),
    # H (3)
    ('Description quality',   'description',           _strlen_gt(100),   'H'),
    ('References',            'reference',             _array_nonempty,   'H'),
    ('Methods',               'method',                _nonempty,         'H'),
    # A (2)
    ('Access right',          'access_right',          _eq('open'),       'A'),
    ('License',               'license_id',            _nonempty,         'A'),
    # R (3)
    ('Discovery (views)',     'views',                 _gt(0),            'R'),
    ('Access (downloads)',    'download_count',        _gt(0),            'R'),
    ('Formal citations',      'citation_count',        _gt(0),            'R'),
    # E (7)
    ('Related identifiers',   'related_identifiers',   _bool,             'E'),
    ('Journal',               'journal',               _bool,             'E'),
    ('Version',               'version',               _nonempty,         'E'),
    ('Imprint',               'imprint',               _bool,             'E'),
    ('Alternate identifiers', 'alternate_identifiers', _bool,             'E'),
    ('Grants/Funding',        'grants',                _bool,             'E'),
    ('Meeting/Conference',    'meeting',               _bool,             'E'),
]

# Identify SHAE signals (non-R) and R signals
SHAE_SIGNALS = [(n, c, fn, b) for n, c, fn, b in ZENODO_PLEDGE if b != 'R']
R_SIGNALS = [(n, c, fn, b) for n, c, fn, b in ZENODO_PLEDGE if b == 'R']


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_record_full(record):
    """Score a record using the full 21-signal Zenodo pledge.
    Returns SHARE score (0-100)."""
    present = 0
    available = len(ZENODO_PLEDGE)
    for name, col, condition_fn, bucket in ZENODO_PLEDGE:
        val = record.get(col)
        if condition_fn is not None and condition_fn(val):
            present += 1
    return (present / available) * 100 if available > 0 else 0.0


def score_record_shae(record):
    """Score a record using only SHAE signals (no R bucket).
    This is the 'SHARE-minus-R' score."""
    present = 0
    available = len(SHAE_SIGNALS)
    for name, col, condition_fn, bucket in SHAE_SIGNALS:
        val = record.get(col)
        if condition_fn is not None and condition_fn(val):
            present += 1
    return (present / available) * 100 if available > 0 else 0.0


def count_fields(record):
    """Simple field count: how many of the 21 Zenodo columns are non-null/non-empty.
    This is the 'naive' baseline model."""
    count = 0
    for name, col, condition_fn, bucket in ZENODO_PLEDGE:
        val = record.get(col)
        if val is not None:
            if isinstance(val, bool):
                if val:
                    count += 1
            elif isinstance(val, (int, float)):
                if val != 0:
                    count += 1
            elif isinstance(val, str):
                if val.strip() and val.strip().lower() not in ('none', 'null'):
                    count += 1
            elif isinstance(val, (list, tuple)):
                if len(val) > 0:
                    count += 1
            else:
                count += 1
    return count


def score_record_per_signal(record):
    """Return a dict of signal_name -> bool (present/absent) for each of the 21 signals."""
    results = {}
    for name, col, condition_fn, bucket in ZENODO_PLEDGE:
        val = record.get(col)
        results[name] = bool(condition_fn is not None and condition_fn(val))
    return results


def score_record_per_bucket(record):
    """Return per-bucket percentage scores (0-100 each)."""
    bucket_present = defaultdict(int)
    bucket_available = defaultdict(int)
    for name, col, condition_fn, bucket in ZENODO_PLEDGE:
        bucket_available[bucket] += 1
        val = record.get(col)
        if condition_fn is not None and condition_fn(val):
            bucket_present[bucket] += 1
    result = {}
    for b in ['S', 'H', 'A', 'R', 'E']:
        avail = bucket_available.get(b, 0)
        pres = bucket_present.get(b, 0)
        result[b] = (pres / avail * 100) if avail > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zenodo_2016():
    """Load Zenodo records from 2016."""
    db_path = os.path.join(DATA_DIR, 'zenodo_tiered_full.db')
    conn = duckdb.connect(db_path, read_only=True)

    cols = [
        'doi', 'keywords', 'contributors', 'language', 'subjects', 'dates',
        'locations', 'description', 'reference', 'method', 'access_right',
        'license_id', 'views', 'download_count', 'citation_count',
        'related_identifiers', 'journal', 'version', 'imprint',
        'alternate_identifiers', 'grants', 'meeting',
        'supplement_tie', 'share_score', 'pub_date'
    ]

    query = f"""
        SELECT {', '.join(cols)}
        FROM records
        WHERE EXTRACT(YEAR FROM pub_date) = 2016
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    records = [dict(zip(cols, row)) for row in rows]
    return records


def load_zenodo_all():
    """Load ALL Zenodo records (for cross-checks)."""
    db_path = os.path.join(DATA_DIR, 'zenodo_tiered_full.db')
    conn = duckdb.connect(db_path, read_only=True)

    cols = [
        'doi', 'keywords', 'contributors', 'language', 'subjects', 'dates',
        'locations', 'description', 'reference', 'method', 'access_right',
        'license_id', 'views', 'download_count', 'citation_count',
        'related_identifiers', 'journal', 'version', 'imprint',
        'alternate_identifiers', 'grants', 'meeting',
        'supplement_tie', 'share_score'
    ]

    rows = conn.execute(f"SELECT {', '.join(cols)} FROM records").fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(group1, group0):
    """Compute Cohen's d effect size between two groups."""
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return 0.0
    m1, m0 = np.mean(group1), np.mean(group0)
    s1, s0 = np.std(group1, ddof=1), np.std(group0, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m0) / pooled_std


# ---------------------------------------------------------------------------
# Validation analyses
# ---------------------------------------------------------------------------

def _run_logistic(X_vals, y_vals, label):
    """Run logistic regression and return stats dict.

    Returns dict with OR, AUC, Cohen's d, etc.
    """
    X = np.array(X_vals).reshape(-1, 1)
    y = np.array(y_vals)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"\n  --- {label} ---")
    print(f"  N = {len(y):,} (positive: {n_pos:,}, negative: {n_neg:,})")
    print(f"  Prevalence: {n_pos / len(y) * 100:.2f}%")

    result = {
        'label': label,
        'n': len(y),
        'n_positive': n_pos,
        'n_negative': n_neg,
        'prevalence_pct': round(n_pos / len(y) * 100, 2),
    }

    # --- Statsmodels logistic regression for OR and CI ---
    X_sm = sm.add_constant(X)
    try:
        logit_model = sm.Logit(y, X_sm)
        logit_result = logit_model.fit(disp=0, maxiter=100)

        coef = logit_result.params[1]
        se = logit_result.bse[1]
        ci_lower, ci_upper = logit_result.conf_int()[1]

        or_10 = math.exp(coef * 10)
        or_10_lower = math.exp(ci_lower * 10)
        or_10_upper = math.exp(ci_upper * 10)
        p_value = logit_result.pvalues[1]

        print(f"  Logistic Regression:")
        print(f"    Coefficient:  {coef:.6f}")
        print(f"    SE:           {se:.6f}")
        print(f"    p-value:      {p_value:.2e}")
        print(f"    OR per 10-pt: {or_10:.3f} (95% CI: {or_10_lower:.3f} - {or_10_upper:.3f})")

        result.update({
            'or_per_10pt': round(or_10, 4),
            'or_ci_lower': round(or_10_lower, 4),
            'or_ci_upper': round(or_10_upper, 4),
            'p_value': float(f"{p_value:.2e}"),
        })
    except Exception as e:
        print(f"  Statsmodels failed: {e}")
        result.update({'or_per_10pt': None, 'or_ci_lower': None, 'or_ci_upper': None, 'p_value': None})

    # --- Sklearn logistic regression for AUC ---
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X, y)
    y_prob = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"  AUC (ROC): {auc:.4f}")
    result['auc'] = round(auc, 4)

    # --- Cohen's d ---
    scores_pos = [X_vals[i] for i in range(len(y)) if y[i] == 1]
    scores_neg = [X_vals[i] for i in range(len(y)) if y[i] == 0]
    d = cohens_d(scores_pos, scores_neg)
    print(f"  Cohen's d: {d:.4f}")
    print(f"    Mean SHAE (pos): {np.mean(scores_pos):.2f}")
    print(f"    Mean SHAE (neg): {np.mean(scores_neg):.2f}")

    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"
    print(f"    Interpretation: {interp}")

    result.update({
        'cohens_d': round(d, 4),
        'mean_shae_positive': round(float(np.mean(scores_pos)), 2),
        'mean_shae_negative': round(float(np.mean(scores_neg)), 2),
        'effect_interpretation': interp,
    })

    return result


def analysis_1_reuse_prediction(records):
    """Logistic regression: SHARE-minus-R (SHAE) score predicting reuse.

    Two reuse proxies:
      (a) PRIMARY: citation_count > 0 (actual scholarly reuse)
      (b) SECONDARY: supplement_tie (structural derivation link)

    Reports: OR per 10-point SHAE increase, AUC, Cohen's d
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Reuse Prediction (Logistic Regression)")
    print("=" * 70)
    print("  Predictor: SHAE score (S+H+A+E signals, excluding R bucket)")

    # Build common predictor array
    shae_scores = [score_record_shae(r) for r in records]

    # (a) Primary: citation_count > 0
    y_cited = []
    for r in records:
        cc = r.get('citation_count')
        y_cited.append(1 if cc is not None and cc > 0 else 0)

    result_cited = _run_logistic(shae_scores, y_cited,
                                  "Proxy A: Has citations (citation_count > 0)")

    # (b) Secondary: supplement_tie
    y_supp = []
    for r in records:
        y_supp.append(1 if r.get('supplement_tie') is True or r.get('supplement_tie') == 1 else 0)

    result_supp = _run_logistic(shae_scores, y_supp,
                                 "Proxy B: Supplement tie (structural derivative link)")

    return {
        'primary_citation_reuse': result_cited,
        'secondary_supplement_tie': result_supp,
    }


def analysis_2_citation_correlation(records):
    """Pearson r between SHARE-minus-R score and log(citations+1).
    Only includes records with citation_count > 0."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Citation Correlation")
    print("=" * 70)

    shae_scores = []
    log_citations = []
    for r in records:
        cc = r.get('citation_count')
        if cc is not None and cc > 0:
            shae = score_record_shae(r)
            shae_scores.append(shae)
            log_citations.append(math.log(cc + 1))

    n = len(shae_scores)
    print(f"\n  N (citation_count > 0): {n:,}")

    if n < 3:
        print("  Not enough records for correlation.")
        return {'n': n, 'pearson_r': None, 'p_value': None}

    r_val, p_val = sp_stats.pearsonr(shae_scores, log_citations)
    print(f"  Pearson r:  {r_val:.4f}")
    print(f"  p-value:    {p_val:.2e}")

    # Spearman as robustness check
    rho, rho_p = sp_stats.spearmanr(shae_scores, log_citations)
    print(f"  Spearman rho: {rho:.4f} (p={rho_p:.2e})")

    # Descriptive stats
    print(f"\n  SHAE score (cited records):  mean={np.mean(shae_scores):.2f}, median={np.median(shae_scores):.2f}")
    print(f"  log(citations+1):           mean={np.mean(log_citations):.2f}, median={np.median(log_citations):.2f}")

    return {
        'n': n,
        'pearson_r': round(r_val, 4),
        'pearson_p': float(f"{p_val:.2e}"),
        'spearman_rho': round(rho, 4),
        'spearman_p': float(f"{rho_p:.2e}"),
        'mean_shae': round(np.mean(shae_scores), 2),
        'median_shae': round(np.median(shae_scores), 2),
        'mean_log_cit': round(np.mean(log_citations), 2),
    }


def analysis_3_per_signal_citation_boost(records):
    """For each of the 21 Zenodo signals: mean & median citations when present vs absent.

    Two views:
      A) All records: mean citations (present vs absent), ratio
      B) Cited records only (citation_count > 0): median citations, ratio

    Ranked by mean ratio descending.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Per-Signal Citation Boost")
    print("=" * 70)

    # Collect citation_count for each signal present vs absent
    signal_data = {}
    for name, col, condition_fn, bucket in ZENODO_PLEDGE:
        signal_data[name] = {
            'present_all': [], 'absent_all': [],
            'present_cited': [], 'absent_cited': [],
            'bucket': bucket,
        }

    for r in records:
        cc = r.get('citation_count')
        if cc is None:
            cc = 0
        cited = cc > 0
        for name, col, condition_fn, bucket in ZENODO_PLEDGE:
            val = r.get(col)
            is_present = bool(condition_fn is not None and condition_fn(val))
            if is_present:
                signal_data[name]['present_all'].append(cc)
                if cited:
                    signal_data[name]['present_cited'].append(cc)
            else:
                signal_data[name]['absent_all'].append(cc)
                if cited:
                    signal_data[name]['absent_cited'].append(cc)

    results = []

    # --- View A: All records, mean citations ---
    print(f"\n  View A: All records — mean citations (present vs absent)")
    print(f"  {'Signal':<25} {'Bucket':>6} {'Mean(P)':>9} {'Mean(A)':>9} {'Ratio':>8} {'N(P)':>8} {'N(A)':>8}")
    print(f"  {'-' * 76}")

    for name in [n for n, _, _, _ in ZENODO_PLEDGE]:
        d = signal_data[name]
        mean_p = np.mean(d['present_all']) if d['present_all'] else 0
        mean_a = np.mean(d['absent_all']) if d['absent_all'] else 0
        ratio = (mean_p / mean_a) if mean_a > 0 else (float('inf') if mean_p > 0 else 1.0)

        med_cited_p = np.median(d['present_cited']) if d['present_cited'] else 0
        med_cited_a = np.median(d['absent_cited']) if d['absent_cited'] else 0
        cited_ratio = (med_cited_p / med_cited_a) if med_cited_a > 0 else (float('inf') if med_cited_p > 0 else 1.0)

        results.append({
            'signal': name,
            'bucket': d['bucket'],
            'mean_present': round(float(mean_p), 3),
            'mean_absent': round(float(mean_a), 3),
            'mean_ratio': round(ratio, 3) if ratio != float('inf') else 'inf',
            'n_present': len(d['present_all']),
            'n_absent': len(d['absent_all']),
            'median_cited_present': round(float(med_cited_p), 1),
            'median_cited_absent': round(float(med_cited_a), 1),
            'cited_ratio': round(cited_ratio, 3) if cited_ratio != float('inf') else 'inf',
            'n_cited_present': len(d['present_cited']),
            'n_cited_absent': len(d['absent_cited']),
        })

    # Sort by mean_ratio descending
    results.sort(key=lambda x: x['mean_ratio'] if isinstance(x['mean_ratio'], (int, float)) else 999, reverse=True)

    for row in results:
        ratio_str = f"{row['mean_ratio']:.3f}" if isinstance(row['mean_ratio'], (int, float)) else row['mean_ratio']
        print(f"  {row['signal']:<25} {row['bucket']:>6} {row['mean_present']:>9.3f} {row['mean_absent']:>9.3f} {ratio_str:>8} {row['n_present']:>8,} {row['n_absent']:>8,}")

    # --- View B: Cited records only, median citations ---
    print(f"\n  View B: Cited records only (citation_count > 0) — median citations")
    print(f"  {'Signal':<25} {'Bucket':>6} {'Med(P)':>8} {'Med(A)':>8} {'Ratio':>8} {'N(P)':>8} {'N(A)':>8}")
    print(f"  {'-' * 73}")

    results_b = sorted(results, key=lambda x: x['cited_ratio'] if isinstance(x['cited_ratio'], (int, float)) else 999, reverse=True)
    for row in results_b:
        ratio_str = f"{row['cited_ratio']:.3f}" if isinstance(row['cited_ratio'], (int, float)) else row['cited_ratio']
        print(f"  {row['signal']:<25} {row['bucket']:>6} {row['median_cited_present']:>8.1f} {row['median_cited_absent']:>8.1f} {ratio_str:>8} {row['n_cited_present']:>8,} {row['n_cited_absent']:>8,}")

    return results


def analysis_4_model_comparison(records):
    """Model comparison for predicting reuse (citation_count > 0):
      - Model A: Full 5-bucket SHARE v2 score (AUC)
      - Model B: Simple field count (AUC)
      - Model C: SHAE-only score (no R signals) (AUC)
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Model Comparison (Full SHARE vs Field Count vs SHAE)")
    print("=" * 70)
    print("  Outcome: citation_count > 0 (has been cited)")

    y_vals = []
    full_scores = []
    field_counts = []
    shae_scores = []

    for r in records:
        cc = r.get('citation_count')
        y_vals.append(1 if cc is not None and cc > 0 else 0)
        full_scores.append(score_record_full(r))
        field_counts.append(count_fields(r))
        shae_scores.append(score_record_shae(r))

    y = np.array(y_vals)
    n_pos = int(y.sum())
    print(f"\n  N = {len(y):,} (cited: {n_pos:,}, uncited: {len(y) - n_pos:,})")

    if n_pos < 5 or (len(y) - n_pos) < 5:
        print("  Not enough events for meaningful AUC comparison.")
        return {'model_a_auc': None, 'model_b_auc': None, 'model_c_auc': None}

    # Model A: Full SHARE score (all 21 signals including R)
    X_a = np.array(full_scores).reshape(-1, 1)
    lr_a = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_a.fit(X_a, y)
    auc_a = roc_auc_score(y, lr_a.predict_proba(X_a)[:, 1])

    # Model B: Simple field count
    X_b = np.array(field_counts).reshape(-1, 1)
    lr_b = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_b.fit(X_b, y)
    auc_b = roc_auc_score(y, lr_b.predict_proba(X_b)[:, 1])

    # Model C: SHAE-only score (deposit-time signals only, no R)
    X_c = np.array(shae_scores).reshape(-1, 1)
    lr_c = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_c.fit(X_c, y)
    auc_c = roc_auc_score(y, lr_c.predict_proba(X_c)[:, 1])

    print(f"\n  Model A (Full SHARE v2, 21 signals):     AUC = {auc_a:.4f}")
    print(f"  Model B (Simple field count):             AUC = {auc_b:.4f}")
    print(f"  Model C (SHAE-only, 18 deposit signals):  AUC = {auc_c:.4f}")
    print(f"\n  Delta A-B: {auc_a - auc_b:+.4f}")
    print(f"  Delta A-C: {auc_a - auc_c:+.4f}")
    print(f"  Delta C-B: {auc_c - auc_b:+.4f}")

    # Note on Model A: it includes R signals (views, downloads, citations)
    # which are correlated with the outcome, so AUC(A) > AUC(C) is expected.
    # The key comparison is C vs B: does structured SHAE scoring beat naive field count?
    print(f"\n  Note: Model A includes R signals (views/downloads/citations) which")
    print(f"  overlap with the outcome. Model C (SHAE) is the fair comparison vs B.")

    # Descriptive
    print(f"\n  Full SHARE scores: mean={np.mean(full_scores):.2f}, sd={np.std(full_scores):.2f}")
    print(f"  SHAE scores:       mean={np.mean(shae_scores):.2f}, sd={np.std(shae_scores):.2f}")
    print(f"  Field counts:      mean={np.mean(field_counts):.2f}, sd={np.std(field_counts):.2f}")

    winner = "SHARE" if auc_c > auc_b else "field count"
    print(f"\n  -> {winner} wins for deposit-time prediction (SHAE={auc_c:.4f} vs count={auc_b:.4f})")

    return {
        'n': len(y),
        'n_cited': n_pos,
        'model_a_full_share_auc': round(auc_a, 4),
        'model_b_field_count_auc': round(auc_b, 4),
        'model_c_shae_only_auc': round(auc_c, 4),
        'delta_a_b': round(auc_a - auc_b, 4),
        'delta_a_c': round(auc_a - auc_c, 4),
        'delta_c_b': round(auc_c - auc_b, 4),
        'mean_full_share': round(float(np.mean(full_scores)), 2),
        'sd_full_share': round(float(np.std(full_scores)), 2),
        'mean_shae': round(float(np.mean(shae_scores)), 2),
        'sd_shae': round(float(np.std(shae_scores)), 2),
        'mean_field_count': round(float(np.mean(field_counts)), 2),
        'sd_field_count': round(float(np.std(field_counts)), 2),
    }


def analysis_5_cross_repo_validation():
    """Cross-repo validation using rescore_unified.py engine.
    Score all repos and check known-groups validity:
    strict repos (SRA) should score higher than general (Zenodo)."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: Cross-Repository Validation (Known-Groups)")
    print("=" * 70)

    # Import the scoring engine from rescore_unified.py
    sys.path.insert(0, SCRIPT_DIR)
    import rescore_unified as ru

    repo_results = {}
    repos_to_score = ['zenodo', 'dryad', 'openneuro', 'sra']

    # Conditionally add repos that have databases
    for extra in ['openaire', 'ctgov', 'geo']:
        cfg = ru.REPO_CONFIG.get(extra)
        if cfg:
            repos_to_score.append(extra)

    for repo_name in repos_to_score:
        cfg = ru.REPO_CONFIG.get(repo_name)
        if cfg is None:
            continue

        try:
            print(f"\n  Loading {repo_name}...", flush=True)
            records = cfg['loader']()
            pledge = cfg['pledge']
            special = cfg['special']

            scores = []
            for r in records:
                result = ru.score_record(r, pledge, special)
                scores.append(result['share_score'])

            n = len(scores)
            if n == 0:
                continue

            mean_score = np.mean(scores)
            median_score = np.median(scores)
            sd_score = np.std(scores)

            repo_results[repo_name] = {
                'n': n,
                'mean': round(float(mean_score), 2),
                'median': round(float(median_score), 2),
                'sd': round(float(sd_score), 2),
                'min': round(float(min(scores)), 2),
                'max': round(float(max(scores)), 2),
                'signals_available': len(pledge),
            }

            print(f"    {repo_name}: n={n:,}, mean={mean_score:.1f}, median={median_score:.1f}, sd={sd_score:.1f}")

        except Exception as e:
            print(f"    {repo_name}: FAILED - {e}")
            continue

    # Sort by mean score (descending)
    sorted_repos = sorted(repo_results.items(), key=lambda x: x[1]['mean'], reverse=True)

    print(f"\n  {'Repository':<20} {'N':>12} {'Signals':>8} {'Mean':>8} {'Median':>8} {'SD':>8}")
    print(f"  {'-' * 64}")
    for repo_name, stats in sorted_repos:
        print(f"  {repo_name:<20} {stats['n']:>12,} {stats['signals_available']:>8} {stats['mean']:>8.1f} {stats['median']:>8.1f} {stats['sd']:>8.1f}")

    # Known-groups validity check
    print(f"\n  Known-Groups Validity Check:")
    print(f"  (Expectation: strict/curated repos > general repos)")

    # Define expected ordering
    expected_order = {
        'sra': 'strict (structured submission)',
        'openneuro': 'strict (BIDS standard)',
        'dryad': 'curated (peer-reviewed)',
        'zenodo': 'general (self-deposit)',
        'openaire': 'aggregated (mixed quality)',
    }

    for repo_name, description in expected_order.items():
        if repo_name in repo_results:
            print(f"    {repo_name:<15} ({description}): {repo_results[repo_name]['mean']:.1f}")

    # Check if SRA > Zenodo
    if 'sra' in repo_results and 'zenodo' in repo_results:
        sra_mean = repo_results['sra']['mean']
        zen_mean = repo_results['zenodo']['mean']
        if sra_mean > zen_mean:
            print(f"\n  PASS: SRA ({sra_mean:.1f}) > Zenodo ({zen_mean:.1f})")
        else:
            print(f"\n  NOTE: SRA ({sra_mean:.1f}) <= Zenodo ({zen_mean:.1f}) — may reflect different pledge sizes")

    return {
        'repos': repo_results,
        'ranking': [r[0] for r in sorted_repos],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  SHARE Score v2 — Validation Script")
    print("  Zenodo 2016 Cohort Analysis")
    print("=" * 70)

    # Load data
    print("\n  Loading Zenodo 2016 cohort...", flush=True)
    records_2016 = load_zenodo_2016()
    print(f"  Loaded {len(records_2016):,} records from 2016.\n")

    # Quick descriptive stats
    full_scores = [score_record_full(r) for r in records_2016]
    shae_scores = [score_record_shae(r) for r in records_2016]
    print(f"  Full SHARE v2:  mean={np.mean(full_scores):.2f}, median={np.median(full_scores):.2f}, sd={np.std(full_scores):.2f}")
    print(f"  SHAE (no R):    mean={np.mean(shae_scores):.2f}, median={np.median(shae_scores):.2f}, sd={np.std(shae_scores):.2f}")
    print(f"  Signals: {len(ZENODO_PLEDGE)} total ({len(SHAE_SIGNALS)} SHAE + {len(R_SIGNALS)} R)")

    results = {}

    # Analysis 1: Reuse prediction
    results['reuse_prediction'] = analysis_1_reuse_prediction(records_2016)

    # Analysis 2: Citation correlation
    results['citation_correlation'] = analysis_2_citation_correlation(records_2016)

    # Analysis 3: Per-signal citation boost
    results['per_signal_citation_boost'] = analysis_3_per_signal_citation_boost(records_2016)

    # Analysis 4: Model comparison
    results['model_comparison'] = analysis_4_model_comparison(records_2016)

    # Analysis 5: Cross-repo validation
    results['cross_repo_validation'] = analysis_5_cross_repo_validation()

    # Add metadata
    results['metadata'] = {
        'cohort': 'Zenodo 2016',
        'n_records': len(records_2016),
        'scoring_version': 'v2 (pledge-based, binary signals)',
        'zenodo_pledge_signals': len(ZENODO_PLEDGE),
        'shae_signals': len(SHAE_SIGNALS),
        'r_signals': len(R_SIGNALS),
        'full_share_mean': round(np.mean(full_scores), 2),
        'full_share_median': round(np.median(full_scores), 2),
        'shae_mean': round(np.mean(shae_scores), 2),
        'shae_median': round(np.median(shae_scores), 2),
    }

    # Save results
    output_path = os.path.join(DATA_DIR, 'validation_v2.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
