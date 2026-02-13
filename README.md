# SHARE Implementation

Reference implementation of the [SHARE Framework](https://github.com/ConductScience-Foundation/share-framework) scoring and validation pipeline, as deployed on [sharescore.org](https://sharescore.org).

This repository contains the scoring engine that produced 76.4M+ scored datasets across 9 repositories, and the validation analyses demonstrating the framework's predictive validity.

## Repository Ecosystem

| Repository | Purpose |
|------------|---------|
| [share-framework](https://github.com/ConductScience-Foundation/share-framework) | Core algorithm (pip-installable) |
| [share-pledges](https://github.com/ConductScience-Foundation/share-pledges) | Repository signal mappings (9 JSON files) |
| **share-implementation** (this repo) | Scoring engine + validation scripts |
| [sharescore.org](https://sharescore.org) | Live platform (76.4M datasets, public API) |

## Structure

```
scoring/
  rescore_unified.py     # Main scoring engine (all 9 repositories)
validation/
  validate_v2.py         # Primary validation (OR, AUC, Cohen's d)
  consistency_validation.py  # Consistency vs spike sharers
  analyze_share_value.py     # Component discrimination analysis
  analyze_share_value_cohort.py  # Temporal cohort analysis
results/
  validation_v2.json     # Pre-computed validation output
data/
  README.md              # Data access instructions
```

## Quick Start

### Install dependencies

```bash
pip install duckdb numpy pandas scipy scikit-learn statsmodels share-framework
```

### Score datasets

The scoring engine reads from DuckDB databases. Set `SHARE_DATA_DIR` to point to your data directory:

```bash
export SHARE_DATA_DIR=/path/to/your/data

# Score all repositories
python scoring/rescore_unified.py

# Score a specific repository
python scoring/rescore_unified.py zenodo

# View fill rates only (fast)
python scoring/rescore_unified.py --fillrates
```

### Run validation

```bash
export SHARE_DATA_DIR=/path/to/your/data

# Full validation suite (Zenodo 2016 cohort)
python validation/validate_v2.py

# Consistency analysis
python validation/consistency_validation.py
```

### Query the live API (no local data needed)

```bash
# Top datasets by SHARE score
curl "https://api.sharescore.org/records?limit=10&sortBy=shareScore&sortOrder=desc"

# Top researchers by S-Index
curl "https://api.sharescore.org/authors?limit=10&sortBy=sIndex&sortOrder=desc"

# Platform metrics
curl "https://api.sharescore.org/metrics/live"
```

API docs: https://api.sharescore.org/docs

## Scoring Methodology

SHARE = (signals present / 25) x 100

Each dataset is scored against 25 universal signals across 5 buckets:

| Bucket | Name | Signals | Max |
|--------|------|---------|-----|
| S | Stewardship | 5 | 20 |
| H | Harmonization | 5 | 20 |
| A | Access | 4 (value-weighted) | 20 |
| R | Reuse | 1 (log-scaled) | 20 |
| E | Engagement | 5 | 20 |

The denominator is always 25 (not the number of signals a repository supports). This prevents gaming via selective signal adoption.

Repository-specific signal mappings are defined in [share-pledges](https://github.com/ConductScience-Foundation/share-pledges).

## Validation Results

Using the Zenodo 2016 cohort (n=48,771):

- **Citation prediction OR**: 3.0x per 10-point SHAE increase (95% CI: 2.87-3.18, p<0.001)
- **Derivative prediction OR**: 5.73x (95% CI: 4.97-6.61)
- **AUC (SHAE vs field count)**: Structured scoring outperforms naive field count
- **Known-groups validity**: Strict repositories (SRA) score higher than general (Zenodo)

See `results/validation_v2.json` for full output.

## Data Sources

All upstream data repositories are publicly accessible:

| Repository | Records | API |
|-----------|---------|-----|
| OpenAIRE | 73.4M | https://api.openaire.eu |
| Zenodo | 1.3M | https://zenodo.org/api |
| SRA | 644K | BigQuery: `nih-sra-datastore.sra.metadata` |
| ClinicalTrials.gov | 571K | https://clinicaltrials.gov/api/v2 |
| GEO | 274K | NCBI E-utilities + FTP |
| Dryad | 109K | https://datadryad.org/api/v2 |
| EDI | 47K | https://pasta.lternet.edu |
| NASA | 27K | https://data.nasa.gov/api/3 |
| OpenNeuro | 1,873 | https://openneuro.org/crn/graphql |

## License

Apache 2.0

## Citation

He S, He YH, Corscadden L. SHARE Framework: A Universal Data Sharing Quality Score.
ConductScience Foundation, 2026. https://sharescore.org
