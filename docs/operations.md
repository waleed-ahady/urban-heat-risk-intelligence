# Operations Guide

## First-time setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make demo
```

## Live initialization

1. Download versioned demographic, environmental, and coordinate extracts.
2. Map them to the templates under `data/reference/templates/`.
3. Build and validate the canonical profile with `scripts/build_district_profiles.py` and `scripts/validate_district_profiles.py`.
4. Set historical dates in `configs/settings.json` or environment variables.
5. Run `python scripts/train_models.py --mode live`.
6. Review `artifacts/model_metrics.json`.
7. Run `python scripts/refresh_forecast.py --mode live`.
8. Review `data/processed/data_health.json`.
9. Start Streamlit.

## Failure handling

The pipeline fails closed for:

- Missing required weather columns
- Invalid timestamps
- Duplicate district/timestamp rows
- Humidity outside 0–100%
- Negative wind or precipitation
- Daily coverage below 20 hours
- Missing district profile fields
- Missing model artifacts

GitHub Actions will retain the last committed dashboard output when a refresh fails. Configure repository notifications so failures are visible.

## Retraining

Retrain when:

- The historical window changes materially
- District definitions change
- Feature definitions change
- Model disagreement increases persistently
- A new official vulnerability dataset is adopted
- A real operational target becomes available

## Reproducibility

The demo generator uses fixed random seeds. Model random state, contamination, dates, and risk weights are versioned in `configs/settings.json`.
