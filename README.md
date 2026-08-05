# Urban Heat Risk Intelligence

[![CI](https://github.com/YOUR_USERNAME/urban-heat-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/urban-heat-risk-intelligence/actions/workflows/ci.yml)
[![Streamlit](https://img.shields.io/badge/Streamlit-deploy-red)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end urban heat decision-support product for Berlin's 12 districts. It retrieves weather data, validates and aggregates it, derives local-climate features, compares two anomaly-detection models, and combines forecast hazard with population exposure and district vulnerability to produce an explainable 0–100 risk score in Streamlit.

> **Important:** This is a portfolio and research prototype. It is not an official warning system, a medical device, or a substitute for government emergency guidance.

## What the product does

```text
Open-Meteo historical + forecast weather
                  │
                  ▼
        Validation and daily features
                  │
                  ▼
       District/month climatology
                  │
          ┌───────┴────────┐
          ▼                ▼
  Isolation Forest   Robust Covariance
          └───────┬────────┘
                  ▼
       Ensemble anomaly percentile
                  │
                  ▼
 Hazard × exposure × vulnerability engine
                  │
                  ▼
       Streamlit decision dashboard
```

The weather provider supplies the forecast. This repository adds the decision layer:

- How unusual are the forecast conditions for this district and season?
- Which districts should be prioritized first?
- Is risk driven by heat severity, warm nights, event duration, population exposure, or local vulnerability?
- How much do the two anomaly models agree?
- What changes under a planning scenario?

## Modeling strategy

This project deliberately avoids fabricating health-outcome labels.

### Model 1: Isolation Forest

Captures globally unusual combinations of apparent temperature, warm nights, humidity, radiation, heat hours, and heatwave duration.

### Model 2: Robust Covariance

Uses a robust covariance envelope to identify multivariate conditions with unusually large robust distance.

### Ensemble

Each raw model score is converted to an empirical training percentile. The final anomaly score is:

```text
anomaly_score = 100 × (0.70 × isolation_forest_percentile
                     + 0.30 × robust_covariance_percentile)
```

The risk engine is transparent rather than learned from unavailable outcomes:

```text
risk = 0.55 × hazard + 0.20 × exposure + 0.25 × vulnerability
```

See [`docs/model_card.md`](docs/model_card.md) for assumptions, evaluation, and limitations.

## Repository features

- Live and deterministic offline data modes
- Open-Meteo historical and forecast clients
- Schema, range, duplicate, missingness, and coverage checks
- Hourly-to-daily feature pipeline
- District/month robust climatology using median and MAD
- Two unsupervised anomaly models and percentile ensemble
- Final-year time holdout evaluation
- Weak-label benchmark based on historical 98th-percentile extremes
- Explainable hazard–exposure–vulnerability risk engine
- Multi-page Streamlit dashboard
- Scenario simulator
- Model disagreement and data-health pages
- pytest unit tests
- Ruff lint configuration and typed module interfaces
- GitHub Actions CI and scheduled refresh
- Docker support
- Model card, architecture, data dictionary, and operating guide

## Quick start: fully offline demo

The demo is deterministic and does not require an API connection.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
make demo
make app
```

Open the local URL printed by Streamlit.

`make demo` generates synthetic historical and forecast weather, trains both models, evaluates them, and creates dashboard artifacts. The demo profile is visibly labeled and must not be interpreted as official district evidence.

## Live weather mode

Train on real Open-Meteo historical weather:

```bash
python scripts/train_models.py --mode live
```

Refresh current and upcoming risk outputs:

```bash
python scripts/refresh_forecast.py --mode live
streamlit run app/Home.py
```

Open-Meteo provides historical reanalysis and forecast APIs without requiring an API key for non-commercial use. Review its current terms and attribution requirements before deployment:

- Historical API: https://open-meteo.com/en/docs/historical-weather-api
- Forecast API: https://open-meteo.com/en/docs
- About and licensing: https://open-meteo.com/en/about

## District vulnerability data

`data/reference/district_profiles.csv` is an **offline demo seed**, with an explicit `source_quality` field. A credible real deployment must rebuild these fields from versioned public sources:

- Population and age: Amt für Statistik Berlin-Brandenburg / Berlin Open Data
- Population density, green volume, land use, and impervious surface: Berlin Environmental Atlas WFS or downloads

Relevant catalogues:

- https://daten.berlin.de/datensaetze
- https://www.statistik-berlin-brandenburg.de/
- https://www.berlin.de/umweltatlas/

The intended production contract is:

| Field | Meaning |
|---|---|
| `population_density_per_km2` | Exposure proxy |
| `share_age_65_plus_pct` | Heat-sensitivity proxy |
| `green_space_pct` | Cooling and adaptive-capacity proxy |
| `impervious_surface_pct` | Urban heat-retention proxy |
| `source_quality` | Provenance and release status |

Use the canonical builder after downloading and mapping official extracts to the provided templates:

```bash
python scripts/build_district_profiles.py \
  --demographics data/raw/demographics.csv \
  --environment data/raw/environment.csv \
  --coordinates data/raw/coordinates.csv \
  --source-quality "official_release_YYYY-MM-DD"
python scripts/validate_district_profiles.py
```

Input templates are under `data/reference/templates/`. The builder calculates population density and the share aged 65+, performs one-to-one district joins, validates ranges, and writes the canonical profile.

## Dashboard pages

1. **Executive overview** — map, district ranking, KPIs, seven-day outlook.
2. **District Forecast** — trajectory, risk decomposition, scenario simulator, detailed output.
3. **Anomaly Lab** — Isolation Forest versus robust covariance, ensemble score, disagreement analysis.
4. **Model Quality** — time-holdout metrics and proxy-label limitations.
5. **Data Health** — freshness, validation results, provenance status, output sample.

## Project layout

```text
urban-heat-risk-intelligence/
├── .github/workflows/          # CI and scheduled forecast refresh
├── app/                        # Streamlit application
├── artifacts/                  # Serialized models and metrics
├── configs/                    # Runtime settings
├── data/
│   ├── processed/              # Generated model and dashboard tables
│   └── reference/              # District profiles and provenance notes
├── docs/                       # Architecture, model card, data dictionary
├── scripts/                    # CLI entry points, including profile building
├── src/urban_heat_risk/
│   ├── data/                   # API clients, demo generation, validation
│   ├── features/               # Aggregation and climatology
│   ├── models/                 # Two anomaly detectors and evaluation
│   ├── pipelines/              # Train and refresh orchestration
│   └── risk/                   # Explainable risk engine
└── tests/                      # Unit tests
```

## Commands

```bash
make install   # install package and development dependencies
make demo      # generate offline data, train models, refresh outputs
make train     # train in demo mode
make refresh   # refresh with live Open-Meteo data
make app       # run Streamlit
make test      # run pytest
make lint      # Ruff
```

## Evaluation

The repository uses a final-year time holdout. Because no hospital or emergency-call labels are included, it evaluates retrieval of district/month extreme days, defined from training-period 98th-percentile apparent-temperature or warm-night thresholds.

This benchmark is useful for engineering validation, but it does **not** establish that the system predicts illness, mortality, or emergency demand. The exact generated metrics are stored in `artifacts/model_metrics.json` and displayed in the dashboard.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. Confirm generated artifacts are committed, or run the live training pipeline and commit them.
3. In Streamlit Community Cloud, select the repository and set the entry point to `app/Home.py`.
4. Deploy.

Community Cloud installs dependencies from `pyproject.toml` and updates the application after repository changes.

## Scheduled refresh

`.github/workflows/refresh.yml` runs daily and commits refreshed forecast artifacts. Before enabling it:

- Train and commit a **live** model bundle.
- Replace demo vulnerability profiles.
- Confirm Open-Meteo usage and attribution requirements.
- Protect the main branch appropriately or change the workflow to publish artifacts elsewhere.

GitHub can disable scheduled workflows in public repositories after extended inactivity. Keep a manual `workflow_dispatch` option, which is already included.

## Responsible use

Do not use this prototype to issue official public warnings, make clinical decisions, or deprioritize communities. The risk score is a relative prioritization mechanism, not a probability of harm. Validate all weights with domain experts, conduct subgroup and geographic sensitivity analyses, and replace proxy outcomes with legitimate operational targets when available.

## Roadmap

- Aggregate official Berlin LOR or district vulnerability datasets automatically.
- Add district polygons rather than centroid markers.
- Backtest archived issued forecasts against realized weather.
- Add uncertainty from ensemble weather forecasts.
- Integrate operational labels such as emergency calls or grid load where permitted.
- Add model and data drift history rather than only latest-run health checks.

## License

Code is MIT licensed. Upstream datasets retain their own licences and attribution requirements.
