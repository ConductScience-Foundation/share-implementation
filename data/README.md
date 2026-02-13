# Data Access

The scoring engine reads from DuckDB databases stored in this directory. These databases are built from publicly accessible APIs.

## Required Databases

| File | Source | Records | Build Method |
|------|--------|---------|-------------|
| `zenodo_tiered_full.db` | [Zenodo API](https://zenodo.org/api/records) | 1.3M | Paginate `?type=dataset`, store in DuckDB |
| `dryad.db` | [Dryad API](https://datadryad.org/api/v2/datasets) | 109K | Paginate datasets endpoint |
| `openaire.db` | [OpenAIRE API](https://api.openaire.eu/search/datasets) | 73.4M | Harvest via OAI-PMH or Graph dumps |
| `openneuro_full.db` | [OpenNeuro GraphQL](https://openneuro.org/crn/graphql) | 1,873 | Query all datasets via GraphQL |
| `sra_full.db` | [BigQuery](https://console.cloud.google.com/bigquery?p=nih-sra-datastore&d=sra) | 644K | `nih-sra-datastore.sra.metadata` (public) |
| `nasa.db` | [NASA CKAN API](https://data.nasa.gov/api/3/action/package_search) | 27K | Paginate package_search |
| `edi.db` | [PASTA API](https://pasta.lternet.edu/package/eml) | 47K | Crawl EML documents |

## Batch Data (JSON format)

ClinicalTrials.gov and GEO data are stored as JSON batches:

```
clinicaltrials_bulk/batches/*.json    # 571K studies
geo_bulk/batches/*.json               # 274K series
```

## Environment Variable

Set `SHARE_DATA_DIR` to point to this directory:

```bash
export SHARE_DATA_DIR=/path/to/share-implementation/data
```

## Validation Data

| File | Description |
|------|------------|
| `derivative_analysis_dataset.csv` | 183K Zenodo records with derivative flags |
| `validation_v2.json` | Pre-computed validation statistics |

## Alternative: Use the API

If you don't need to re-score from scratch, query the live API:

```bash
curl "https://api.sharescore.org/records?limit=100&sortBy=shareScore&sortOrder=desc"
curl "https://api.sharescore.org/metrics/live"
```

API docs: https://api.sharescore.org/docs
