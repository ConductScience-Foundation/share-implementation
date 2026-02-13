# Reproducing SHARE Scores from Scratch

This guide walks through reproducing the SHARE scoring pipeline end-to-end.

## Prerequisites

- Python 3.9+
- `pip install duckdb numpy pandas scipy scikit-learn statsmodels share-framework`

## Step 1: Obtain Raw Data

All source data is from public APIs. No credentials are required except for BigQuery (free tier).

### Zenodo (1.3M datasets)

```python
import requests

# Paginate through Zenodo API
params = {'size': 1000, 'page': 1, 'type': 'dataset'}
r = requests.get('https://zenodo.org/api/records', params=params)
records = r.json()['hits']['hits']
```

### SRA (644K BioProjects)

```sql
-- Google BigQuery (public dataset, no auth needed)
SELECT bioproject, organism, assay_type, platform, librarylayout,
       libraryselection, consent, center_name
FROM `nih-sra-datastore.sra.metadata`
GROUP BY bioproject
```

### ClinicalTrials.gov (571K studies)

```bash
# Bulk download
curl "https://clinicaltrials.gov/api/v2/studies?pageSize=1000" > batch_1.json
```

### Other repositories

See `data/README.md` for API endpoints and bulk download instructions for GEO, Dryad, EDI, NASA, and OpenNeuro.

## Step 2: Load Data into DuckDB

```python
import duckdb

conn = duckdb.connect('data/zenodo.db')
conn.execute('''
    CREATE TABLE records AS
    SELECT * FROM read_json_auto('zenodo_records/*.json')
''')
```

## Step 3: Apply Pledge Mappings

Each repository's signal mapping is defined in [share-pledges](https://github.com/ConductScience-Foundation/share-pledges). The scoring engine (`scoring/rescore_unified.py`) implements these mappings as Python condition functions.

```python
from scoring.rescore_unified import score_record, ZENODO_PLEDGE

record = {'keywords': ['biology'], 'access_right': 'open', ...}
result = score_record(record, ZENODO_PLEDGE)
print(result['share_score'])  # 0-100
```

## Step 4: Score All Records

```bash
export SHARE_DATA_DIR=/path/to/your/duckdb/files
python scoring/rescore_unified.py
```

Output: `data/unified_scores_v2.json` with per-repository statistics.

## Step 5: Validate Against Live Platform

Compare your locally computed scores against the sharescore.org API:

```python
import requests

# Get a scored dataset from the API
r = requests.get('https://api.sharescore.org/records?search=10.5281/zenodo.1234567')
api_score = r.json()['data'][0]['shareScore']

# Compare with local score
local_score = score_record(your_record, ZENODO_PLEDGE)['share_score']
assert abs(api_score - local_score) < 0.01
```

## Step 6: Run Validation Analyses

```bash
# Primary validation (logistic regression, AUC, Cohen's d)
python validation/validate_v2.py

# Consistency analysis
python validation/consistency_validation.py
```

Pre-computed results are available in `results/validation_v2.json`.

## Verification Checklist

- [ ] Raw data downloaded from public APIs
- [ ] DuckDB databases populated
- [ ] Scoring engine produces scores matching `results/validation_v2.json`
- [ ] Spot-check scores against sharescore.org API
- [ ] Validation analyses reproduce OR, AUC, Cohen's d values
