"""
SHARE Score Unified Scoring Engine — Methodology v2
====================================================
Implements: SHARE = (signals present / 25) × 100

All signals are binary (present/absent). Equal weight per signal.
Fixed denominator of 25 (universal signal count) ensures cross-repository comparability.

Usage:
  py -3.14 scripts/rescore_unified.py                 # All repos
  py -3.14 scripts/rescore_unified.py zenodo           # Zenodo only
  py -3.14 scripts/rescore_unified.py zenodo dryad     # Multiple repos
  py -3.14 scripts/rescore_unified.py --fillrates      # Fill rates only (fast)

Repositories: zenodo, dryad, openaire, openneuro, sra, ctgov, geo
"""

import duckdb
import json
import sys
import os
import glob
import statistics
import math
from collections import defaultdict

DATA_DIR = os.environ.get('SHARE_DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
OUTPUT_DIR = os.environ.get('SHARE_OUTPUT_DIR', DATA_DIR)

# ============================================================================
# PLEDGE DEFINITIONS — from METHODOLOGY_v2.html Section 4
# Each signal: (name, column_or_key, condition_fn, bucket)
# condition_fn takes a record dict and returns bool
# ============================================================================

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
    # DuckDB sometimes returns arrays as strings
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


# --- ZENODO (21 signals) ---
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

# --- DRYAD (13 signals) ---
DRYAD_PLEDGE = [
    # S (2)
    ('Geographic locations',  'locations',             _bool,             'S'),
    ('Human subjects stmt',   'hsi_statement',         _nonempty,         'S'),
    # H (4)
    ('Methods',               'methods',               _nonempty,         'H'),
    ('ORCID (any author)',    'atleast_one_orcid',     _bool,             'H'),
    ('ROR (any affiliation)', 'atleast_one_ror',       _bool,             'H'),
    ('ISNI (any affiliation)','atleast_one_isni',      _bool,             'H'),
    # A (1)
    ('Visibility',            'visibility',            _eq('public'),     'A'),
    # R (3)
    ('Discovery (views)',     'views',                 _gt(0),            'R'),
    ('Access (downloads)',    'download_count',        _gt(0),            'R'),
    ('Formal citations',      'citations',             _gt(0),            'R'),
    # E (3)
    ('Related works',         'related_works',         _bool,             'E'),
    ('Funders',               'funders',               _bool,             'E'),
    ('Related pub ISSN',      'relatedPublicationISSN',_nonempty,         'E'),
]

# --- OPENAIRE (14 signals) ---
OPENAIRE_PLEDGE = [
    # S (5)
    ('Geographic locations',  'has_geoLocations',      _bool,             'S'),
    ('Contributors',          'has_contributors',      _bool,             'S'),
    ('Subjects/Keywords',     'has_subjects',          _bool,             'S'),
    ('Size declared',         'has_size',              _bool,             'S'),
    ('Language',              'has_language',           _bool,             'S'),
    # H (2)
    ('Description',           'has_descriptions',      _bool,             'H'),
    ('File formats',          'formats',               _nonempty,         'H'),
    # A (2)
    ('Access rights',         'best_access_right_label', _contains('open'), 'A'),
    ('License',               'license',               _nonempty,         'A'),
    # R (1)
    ('Formal citations',      'citation_count',        _gt(0),            'R'),
    # E (4)
    ('Instance URLs',         'has_instances_urls',    _bool,             'E'),
    ('Version',               'has_version',           _bool,             'E'),
    ('Subtitle',              'has_subTitle',          _bool,             'E'),
    ('Alternate identifiers', 'has_instances_alternateIdentifiers_scheme', _bool, 'E'),
]

# --- OPENNEURO (14 signals) ---
OPENNEURO_PLEDGE = [
    # S (4)
    ('Consent attestation',   'affirmedConsent',       _bool,             'S'),
    ('Deidentification',      'affirmedDefaced',       _bool,             'S'),
    ('Species',               'species',               _nonempty,         'S'),
    ('Ethics approvals',      'EthicsApprovals',       _array_nonempty,   'S'),
    # H (4)
    ('Study design',          'studyDesign',           _nonempty,         'H'),
    ('Study domain',          'studyDomain',           _nonempty,         'H'),
    ('Acknowledgements',      'Acknowledgements',      _nonempty,         'H'),
    ('References & links',    'ReferencesAndLinks',    _array_nonempty,   'H'),
    # A (1)
    ('License',               'License',               _nonempty,         'A'),
    # R (1)
    ('Formal citations',      'citationCount',         _gt(0),            'R'),
    # E (4)
    ('Associated paper DOI',  '_assoc_paper',          None,              'E'),  # special
    ('Funding',               '_funding',              None,              'E'),  # special
    ('Modalities',            'modalities',            _array_nonempty,   'E'),
    ('How to acknowledge',    'HowToAcknowledge',      _nonempty,         'E'),
]

def _openneuro_special(r, signal_name):
    """Handle OpenNeuro's compound signals."""
    if signal_name == '_assoc_paper':
        return _nonempty(r.get('associatedPaperDOI')) or _nonempty(r.get('openneuroPaperDOI'))
    elif signal_name == '_funding':
        return _nonempty(r.get('grantFunderName')) or _array_nonempty(r.get('Funding'))
    return False

# --- SRA (11 signals) ---
SRA_PLEDGE = [
    # S (2)
    ('Organism specified',    'has_organism',          _bool,             'S'),
    ('Geographic coverage',   'geo_coverage',          _gt(0),            'S'),
    # H (4)
    ('Platform specified',    'has_platform',          _bool,             'H'),
    ('Assay type',            'has_assay_type',        _bool,             'H'),
    ('Library layout',        'has_librarylayout',     _bool,             'H'),
    ('Library selection',     'has_libraryselection',  _bool,             'H'),
    # A (2)
    ('Datastore provider',    'has_datastore',         _bool,             'A'),
    ('All runs public',       'all_public',            _bool,             'A'),
    # E (3)
    ('Multiple centers',      'center_count',          _gt(1),            'E'),
    ('Organism diversity',    'organism_count',        _gt(1),            'E'),
    ('Multiple assay types',  'assay_type_count',      _gt(1),            'E'),
]

# --- CLINICALTRIALS.GOV (19 signals) — scored from generic batch format ---
# Native CT.gov fields are embedded in generic format. We extract what we can.
CTGOV_PLEDGE = [
    # S (5) — approximated from generic fields
    ('Location countries',    '_ct_countries',         None,              'S'),
    ('Eligibility criteria',  '_ct_eligibility',       None,              'S'),
    ('Responsible party',     '_ct_responsible',       None,              'S'),
    ('Conditions',            '_ct_conditions',        None,              'S'),
    ('Keywords',              'keywords',              _array_nonempty,   'S'),
    # H (5)
    ('Study design',          '_ct_design',            None,              'H'),
    ('Investigator affil.',   '_ct_affiliation',       None,              'H'),
    ('Organization',          '_ct_organization',      None,              'H'),
    ('PubMed references',     'citation_count',        _gt(0),            'H'),
    ('Description quality',   'description',           _strlen_gt(300),   'H'),
    # A (4)
    ('IPD sharing intent',    '_ct_ipd_yes',           None,              'A'),
    ('IPD access criteria',   '_ct_ipd_criteria',      None,              'A'),
    ('IPD sharing info',      '_ct_ipd_info',          None,              'A'),
    ('IPD timeframe',         '_ct_ipd_timeframe',     None,              'A'),
    # R (1)
    ('Formal citations',      'citation_count',        _gt(0),            'R'),
    # E (4)
    ('Collaborators',         '_ct_collaborators',     None,              'E'),
    ('Lead sponsor',          'funder_id',             _bool,             'E'),
    ('Study dates',           '_ct_dates',             None,              'E'),
    ('Has results posted',    '_ct_has_results',       None,              'E'),
]

def _ctgov_special(r, signal_name):
    """Handle ClinicalTrials.gov signals parsed from generic format."""
    desc = str(r.get('description', '') or '')
    license_tag = str(r.get('license_tag', '') or '')

    if signal_name == '_ct_countries':
        # Parse "Countries: X" from description
        if 'Countries:' in desc:
            try:
                after = desc.split('Countries:')[1].strip()
                n = int(after.split('.')[0].split()[0])
                return n > 0
            except (ValueError, IndexError):
                pass
        return False
    elif signal_name == '_ct_eligibility':
        # Most CT.gov studies have eligibility criteria
        return 'Eligibility' in desc or r.get('complete_core_metadata', False)
    elif signal_name == '_ct_responsible':
        # Approximated from authors or core metadata
        authors = r.get('authors', [])
        return (isinstance(authors, list) and len(authors) > 0) or r.get('complete_core_metadata', False)
    elif signal_name == '_ct_conditions':
        # Conditions are usually in keywords for CT.gov
        kw = r.get('keywords', [])
        return isinstance(kw, list) and any(k not in ('clinical trial', 'NIH', 'ClinicalTrials.gov') for k in kw)
    elif signal_name == '_ct_design':
        # Design info embedded in description
        return any(x in desc.lower() for x in ['randomized', 'allocation:', 'masking:', 'parallel', 'crossover', 'single group'])
    elif signal_name == '_ct_affiliation':
        return r.get('complete_core_metadata', False)
    elif signal_name == '_ct_organization':
        return r.get('complete_core_metadata', False)
    elif signal_name == '_ct_ipd_yes':
        return 'IPD-YES' in license_tag or 'IPD Sharing: Yes' in desc
    elif signal_name == '_ct_ipd_criteria':
        return 'IPD-YES' in license_tag  # If IPD=YES, usually has criteria
    elif signal_name == '_ct_ipd_info':
        return 'IPD-YES' in license_tag
    elif signal_name == '_ct_ipd_timeframe':
        return 'IPD-YES' in license_tag
    elif signal_name == '_ct_collaborators':
        return False  # Can't extract from generic format
    elif signal_name == '_ct_dates':
        return True  # CT.gov always has dates
    elif signal_name == '_ct_has_results':
        return 'Has Results' in desc or 'results posted' in desc.lower()
    return False

# --- GEO (12 signals) — scored from generic batch format ---
GEO_PLEDGE = [
    # S (3)
    ('Organism/Species',      '_geo_taxon',            None,              'S'),
    ('Sample documentation',  '_geo_samples',          None,              'S'),
    ('BioProject linkage',    '_geo_bioproject',       None,              'S'),
    # H (4)
    ('Summary quality',       'description',           _strlen_gt(200),   'H'),
    ('Platform (GPL)',        '_geo_gpl',              None,              'H'),
    ('Data type',             '_geo_type',             None,              'H'),
    ('Supplementary files',   '_geo_supp',             None,              'H'),
    # A (2)
    ('FTP download',          'download_available',    _bool,             'A'),
    ('Interactive analysis',  '_geo_geo2r',            None,              'A'),
    # R (1)
    ('Formal citations',      'citation_count',        _gt(0),            'R'),
    # E (2)
    ('PubMed linkage',        'citation_count',        _gt(0),            'E'),
    ('External relations',    '_geo_relations',        None,              'E'),
]

def _geo_special(r, signal_name):
    """Handle GEO signals parsed from generic format."""
    desc = str(r.get('description', '') or '')
    kw = r.get('keywords', []) or []

    if signal_name == '_geo_taxon':
        # First keyword is typically species name
        if isinstance(kw, list) and len(kw) > 0:
            return kw[0] not in ('GEO', 'geo', '')
        return False
    elif signal_name == '_geo_samples':
        # Parse "X samples" from description
        if 'samples' in desc.lower():
            try:
                parts = desc.lower().split('samples')[0].strip().split()
                n = int(parts[-1].replace(',', ''))
                return n > 0
            except (ValueError, IndexError):
                pass
        return False
    elif signal_name == '_geo_bioproject':
        # BioProject linkage — look for PRJNA in description
        return 'PRJNA' in desc or 'bioproject' in desc.lower()
    elif signal_name == '_geo_gpl':
        # Platform (GPL) — look for GPL in description or keywords
        return 'GPL' in desc or any('GPL' in str(k) for k in kw)
    elif signal_name == '_geo_type':
        # Data type — profiling type in keywords or description
        return any(t in str(kw) for t in ['Expression profiling', 'Genome binding', 'Methylation profiling', 'Non-coding RNA'])
    elif signal_name == '_geo_supp':
        # Supplementary files — most GEO series have them
        return 'supp' in desc.lower() or r.get('download_available', False)
    elif signal_name == '_geo_geo2r':
        # GEO2R availability — can't determine from generic format
        return False
    elif signal_name == '_geo_relations':
        return False  # Can't determine from generic format
    return False


# ============================================================================
# UNIFIED SCORING ENGINE
# ============================================================================

def score_record(record, pledge, special_fn=None):
    """Score a single record against a pledge.

    Returns dict with:
      - share_score: float 0-100
      - signals_present: int
      - signals_available: int (always 25 — fixed universal denominator)
      - per_bucket: dict of bucket -> (present, available)
      - signal_results: list of (name, bucket, present)
    """
    signals_available = 25  # Fixed denominator per SHARE methodology
    signals_present = 0
    per_bucket = defaultdict(lambda: [0, 0])  # bucket -> [present, available]
    signal_results = []

    for name, col, condition_fn, bucket in pledge:
        per_bucket[bucket][1] += 1  # available

        if col.startswith('_') and special_fn:
            # Use special handler
            present = special_fn(record, col)
        elif condition_fn is not None:
            val = record.get(col)
            present = condition_fn(val)
        else:
            present = False

        if present:
            signals_present += 1
            per_bucket[bucket][0] += 1

        signal_results.append((name, bucket, present))

    share_score = (signals_present / signals_available * 100) if signals_available > 0 else 0.0

    return {
        'share_score': round(share_score, 2),
        'signals_present': signals_present,
        'signals_available': signals_available,
        'per_bucket': dict(per_bucket),
        'signal_results': signal_results,
    }


def compute_stats(values):
    """Compute basic statistics for a list of numbers."""
    if not values:
        return {'n': 0, 'mean': 0, 'median': 0, 'sd': 0, 'min': 0, 'max': 0}
    n = len(values)
    mean_val = sum(values) / n
    median_val = statistics.median(values)
    sd_val = statistics.stdev(values) if n > 1 else 0
    return {
        'n': n,
        'mean': round(mean_val, 2),
        'median': round(median_val, 2),
        'sd': round(sd_val, 2),
        'min': round(min(values), 2),
        'max': round(max(values), 2),
    }


def compute_distribution(values, bins=10):
    """Compute histogram bins (0-10, 10-20, ..., 90-100)."""
    counts = [0] * bins
    for v in values:
        idx = min(int(v / (100 / bins)), bins - 1)
        counts[idx] += 1
    labels = [f"{i*10}-{(i+1)*10}" for i in range(bins)]
    return dict(zip(labels, counts))


def compute_fill_rates(records, pledge, special_fn=None):
    """Compute fill rate for each signal across all records."""
    signal_counts = [0] * len(pledge)
    n = len(records)

    for r in records:
        for i, (name, col, condition_fn, bucket) in enumerate(pledge):
            if col.startswith('_') and special_fn:
                present = special_fn(r, col)
            elif condition_fn is not None:
                val = r.get(col)
                present = condition_fn(val)
            else:
                present = False
            if present:
                signal_counts[i] += 1

    results = []
    for i, (name, col, condition_fn, bucket) in enumerate(pledge):
        rate = signal_counts[i] / n * 100 if n > 0 else 0
        results.append({
            'signal': name,
            'bucket': bucket,
            'fill_rate': round(rate, 2),
            'count': signal_counts[i],
        })
    return results


# ============================================================================
# REPOSITORY LOADERS
# ============================================================================

def load_zenodo():
    """Load all Zenodo records from DuckDB."""
    db_path = os.path.join(DATA_DIR, 'zenodo_tiered_full.db')
    conn = duckdb.connect(db_path, read_only=True)
    print("  Loading Zenodo records...", flush=True)

    cols = ['doi', 'keywords', 'contributors', 'language', 'subjects', 'dates',
            'locations', 'description', 'reference', 'method', 'access_right',
            'license_id', 'views', 'download_count', 'citation_count',
            'related_identifiers', 'journal', 'version', 'imprint',
            'alternate_identifiers', 'grants', 'meeting',
            'share_score']  # old score for comparison

    rows = conn.execute(f"SELECT {', '.join(cols)} FROM records").fetchall()
    conn.close()

    records = [dict(zip(cols, row)) for row in rows]
    print(f"  Loaded {len(records):,} Zenodo records.", flush=True)
    return records


def load_dryad():
    """Load all Dryad records from DuckDB."""
    db_path = os.path.join(DATA_DIR, 'dryad.db')
    conn = duckdb.connect(db_path, read_only=True)
    print("  Loading Dryad records...", flush=True)

    cols = ['doi', 'locations', 'hsi_statement', 'methods', 'atleast_one_orcid',
            'atleast_one_ror', 'atleast_one_isni', 'visibility', 'views',
            'download_count', 'citations', 'related_works', 'funders',
            'relatedPublicationISSN', 'share_score']

    rows = conn.execute(f"SELECT {', '.join(cols)} FROM records").fetchall()
    conn.close()

    records = [dict(zip(cols, row)) for row in rows]
    print(f"  Loaded {len(records):,} Dryad records.", flush=True)
    return records


def load_openaire(sample_size=100000):
    """Load OpenAIRE records from DuckDB (sampled)."""
    db_path = os.path.join(DATA_DIR, 'openaire.db')
    conn = duckdb.connect(db_path, read_only=True)
    print(f"  Loading OpenAIRE sample ({sample_size:,} records)...", flush=True)

    cols = ['id', 'has_geoLocations', 'has_contributors', 'has_subjects', 'has_size',
            'has_language', 'has_descriptions', 'formats', 'best_access_right_label',
            'license', 'citation_count', 'has_instances_urls', 'has_version',
            'has_subTitle', 'has_instances_alternateIdentifiers_scheme',
            'share_score']

    rows = conn.execute(f"SELECT {', '.join(cols)} FROM records USING SAMPLE {sample_size}").fetchall()
    conn.close()

    records = [dict(zip(cols, row)) for row in rows]
    print(f"  Loaded {len(records):,} OpenAIRE records.", flush=True)
    return records


def load_openneuro():
    """Load all OpenNeuro records from DuckDB."""
    db_path = os.path.join(DATA_DIR, 'openneuro_full.db')
    conn = duckdb.connect(db_path, read_only=True)
    print("  Loading OpenNeuro records...", flush=True)

    cols = ['doi', 'affirmedConsent', 'affirmedDefaced', 'species', 'EthicsApprovals',
            'studyDesign', 'studyDomain', 'Acknowledgements', 'ReferencesAndLinks',
            'License', 'citationCount', 'associatedPaperDOI', 'openneuroPaperDOI',
            'grantFunderName', 'Funding', 'modalities', 'HowToAcknowledge']

    rows = conn.execute(f"SELECT {', '.join(cols)} FROM records").fetchall()
    conn.close()

    records = [dict(zip(cols, row)) for row in rows]
    print(f"  Loaded {len(records):,} OpenNeuro records.", flush=True)
    return records


def load_sra():
    """Load SRA BioProject records from DuckDB."""
    # Try full DB first, fall back to sample
    full_path = os.path.join(DATA_DIR, 'sra_full.db')
    sample_path = os.path.join(DATA_DIR, 'sra.db')

    db_path = full_path if os.path.exists(full_path) else sample_path
    conn = duckdb.connect(db_path, read_only=True)
    print(f"  Loading SRA records from {os.path.basename(db_path)}...", flush=True)

    cols = ['bioproject', 'has_organism', 'geo_coverage', 'has_platform',
            'has_assay_type', 'has_librarylayout', 'has_libraryselection',
            'has_datastore', 'all_public', 'center_count', 'organism_count',
            'assay_type_count']

    rows = conn.execute(f"SELECT {', '.join(cols)} FROM projects").fetchall()
    conn.close()

    records = [dict(zip(cols, row)) for row in rows]
    print(f"  Loaded {len(records):,} SRA BioProjects.", flush=True)
    return records


def load_ctgov():
    """Load ClinicalTrials.gov records from JSON batches."""
    print("  Loading ClinicalTrials.gov records...", flush=True)

    batch_files = sorted(glob.glob(os.path.join(DATA_DIR, 'clinicaltrials_bulk', 'batches', '*.json')))
    records = []
    for fp in batch_files:
        with open(fp) as f:
            data = json.load(f)
        batch = data.get('records', data) if isinstance(data, dict) else data
        records.extend(batch)

    print(f"  Loaded {len(records):,} ClinicalTrials.gov records.", flush=True)
    return records


def load_geo():
    """Load GEO records from JSON batches."""
    print("  Loading GEO records...", flush=True)

    batch_files = sorted(glob.glob(os.path.join(DATA_DIR, 'geo_bulk', 'batches', '*.json')))
    records = []
    for fp in batch_files:
        with open(fp) as f:
            data = json.load(f)
        batch = data.get('records', data) if isinstance(data, dict) else data
        records.extend(batch)

    print(f"  Loaded {len(records):,} GEO records.", flush=True)
    return records


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_repo(repo_name, records, pledge, special_fn=None):
    """Score all records and generate comprehensive report."""
    print(f"\n  Scoring {repo_name}...", flush=True)

    scored = []
    scores = []
    bucket_scores = defaultdict(list)

    for r in records:
        result = score_record(r, pledge, special_fn)
        scored.append(result)
        scores.append(result['share_score'])

        for bucket, (present, available) in result['per_bucket'].items():
            if available > 0:
                bucket_pct = present / available * 100
                bucket_scores[bucket].append(bucket_pct)

    # Overall stats
    stats = compute_stats(scores)
    dist = compute_distribution(scores)

    # Fill rates
    fill_rates = compute_fill_rates(records, pledge, special_fn)

    # Per-bucket stats
    bucket_stats = {}
    for b in ['S', 'H', 'A', 'R', 'E']:
        if b in bucket_scores:
            bucket_stats[b] = compute_stats(bucket_scores[b])
        else:
            bucket_stats[b] = {'n': 0, 'mean': 0, 'median': 0, 'sd': 0}

    # Old score comparison (where available)
    old_scores = [r.get('share_score') for r in records if r.get('share_score') is not None]
    old_stats = compute_stats(old_scores) if old_scores else None

    # Print report
    print(f"\n{'='*60}")
    print(f"  {repo_name.upper()} — SHARE Score v2")
    print(f"{'='*60}")
    print(f"  Records: {len(records):,}")
    print(f"  Signals available: {len(pledge)} ({sum(1 for _,_,_,b in pledge if b!='R')} SHAE + {sum(1 for _,_,_,b in pledge if b=='R')} R)")
    print(f"  Score per signal: {100/len(pledge):.2f} pts")

    print(f"\n  Score Statistics:")
    print(f"    Mean:   {stats['mean']:.1f} (±{stats['sd']:.1f})")
    print(f"    Median: {stats['median']:.1f}")
    print(f"    Range:  [{stats['min']:.1f}, {stats['max']:.1f}]")

    if old_stats:
        print(f"\n  Comparison to Old Scores:")
        print(f"    Old mean:  {old_stats['mean']:.1f}")
        print(f"    New mean:  {stats['mean']:.1f}")
        print(f"    Delta:     {stats['mean'] - old_stats['mean']:+.1f}")

    print(f"\n  Per-Bucket Averages (% filled):")
    for b in ['S', 'H', 'A', 'R', 'E']:
        if b in bucket_stats and bucket_stats[b]['n'] > 0:
            bs = bucket_stats[b]
            n_signals = sum(1 for _,_,_,bk in pledge if bk == b)
            print(f"    {b}: {bs['mean']:.1f}% ({n_signals} signals)")

    print(f"\n  Fill Rates:")
    for fr in fill_rates:
        filled = int(fr['fill_rate'] / 5)
        bar = '#' * filled + '.' * (20 - filled)
        print(f"    [{fr['bucket']}] {fr['signal']:<25} {bar} {fr['fill_rate']:6.1f}%")

    print(f"\n  Score Distribution:")
    n = len(scores)
    for bin_label, count in dist.items():
        bar = '#' * int(count / n * 50) if n > 0 else ''
        print(f"    {bin_label:<8} {count:>8,} ({count/n*100:5.1f}%) {bar}")

    return {
        'repository': repo_name,
        'record_count': len(records),
        'signals_available': 25,
        'score_per_signal': round(100 / 25, 2),
        'stats': stats,
        'old_stats': old_stats,
        'bucket_stats': bucket_stats,
        'fill_rates': fill_rates,
        'distribution': dist,
    }


# ============================================================================
# MAIN
# ============================================================================

REPO_CONFIG = {
    'zenodo':   {'loader': load_zenodo,   'pledge': ZENODO_PLEDGE,   'special': None},
    'dryad':    {'loader': load_dryad,    'pledge': DRYAD_PLEDGE,    'special': None},
    'openaire': {'loader': load_openaire, 'pledge': OPENAIRE_PLEDGE, 'special': None},
    'openneuro':{'loader': load_openneuro,'pledge': OPENNEURO_PLEDGE,'special': _openneuro_special},
    'sra':      {'loader': load_sra,      'pledge': SRA_PLEDGE,      'special': None},
    'ctgov':    {'loader': load_ctgov,    'pledge': CTGOV_PLEDGE,    'special': _ctgov_special},
    'geo':      {'loader': load_geo,      'pledge': GEO_PLEDGE,      'special': _geo_special},
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    fill_only = '--fillrates' in sys.argv

    repos_to_run = args if args else list(REPO_CONFIG.keys())

    all_stats = {}
    total_records = 0

    for repo in repos_to_run:
        repo = repo.lower()
        if repo not in REPO_CONFIG:
            print(f"Unknown repo: {repo}. Options: {', '.join(REPO_CONFIG.keys())}")
            continue

        cfg = REPO_CONFIG[repo]
        records = cfg['loader']()

        if fill_only:
            print(f"\n{'='*60}")
            print(f"  {repo.upper()} — Fill Rates")
            print(f"{'='*60}")
            fill_rates = compute_fill_rates(records, cfg['pledge'], cfg['special'])
            for fr in fill_rates:
                filled = int(fr['fill_rate'] / 5)
                bar = '#' * filled + '.' * (20 - filled)
                print(f"  [{fr['bucket']}] {fr['signal']:<25} {bar} {fr['fill_rate']:6.1f}%")
            continue

        stats = analyze_repo(repo.title() if repo != 'ctgov' else 'ClinicalTrials.gov',
                            records, cfg['pledge'], cfg['special'])
        all_stats[repo] = stats
        total_records += len(records)

    if fill_only:
        return

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY — SHARE Score v2 Methodology")
    print(f"{'='*70}")
    print(f"  {'Repository':<22} {'Records':>12} {'Signals':>8} {'Mean':>8} {'±SD':>8} {'Median':>8}")
    print(f"  {'-'*66}")

    weighted_sum = 0
    total_n = 0
    for repo, s in all_stats.items():
        name = s['repository']
        print(f"  {name:<22} {s['record_count']:>12,} {s['signals_available']:>8} {s['stats']['mean']:>8.1f} {s['stats']['sd']:>8.1f} {s['stats']['median']:>8.1f}")
        weighted_sum += s['stats']['mean'] * s['record_count']
        total_n += s['record_count']

    if total_n > 0:
        weighted_avg = weighted_sum / total_n
        print(f"  {'-'*66}")
        print(f"  {'TOTAL':<22} {total_n:>12,} {'':>8} {weighted_avg:>8.1f}")

    # Save output JSON
    output_path = os.path.join(OUTPUT_DIR, 'unified_scores_v2.json')
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
