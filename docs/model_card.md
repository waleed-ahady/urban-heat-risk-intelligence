# Model Card

## Model summary

The project contains two unsupervised anomaly-detection models:

1. Isolation Forest
2. Robust Covariance using EllipticEnvelope

They identify forecast daily heat conditions that are unusual relative to historical district/month climate patterns.

## Intended use

- Portfolio demonstration of production-oriented data science
- Relative prioritization of districts for further review
- Exploration of heat hazard and vulnerability drivers
- Internal planning scenarios

## Out-of-scope use

- Official emergency warnings
- Medical diagnosis or individual risk assessment
- Prediction of deaths, admissions, or ambulance calls
- Automated allocation that denies resources to a community
- Causal claims about green space or demographics

## Input features

- Robust apparent-temperature anomaly
- Robust maximum-temperature anomaly
- Robust warm-night anomaly
- Robust humidity anomaly
- Robust solar-radiation anomaly
- Hours with apparent temperature at or above 35°C
- Consecutive hot-day streak

All anomaly features are normalized against district/month historical climatology before model fitting.

## Training data

Two modes are supported:

- `demo`: deterministic synthetic weather for offline reproducibility
- `live`: Open-Meteo historical reanalysis for the configured period

The bundled serialized model was trained in demo mode so the repository runs immediately. It must be replaced with a live-trained artifact for real weather analysis.

## Evaluation

A strict final-year time holdout is used. Proxy extreme labels are defined as days exceeding the training-period district/month 98th percentile for either maximum apparent temperature or night minimum temperature.

Metrics:

- ROC AUC
- Average precision
- Precision among the top 3% anomaly scores
- Correlation between model scores

These metrics evaluate extreme-day retrieval only. They do not validate health or operational impact prediction.

## Risk score

The anomaly ensemble contributes to a deterministic hazard score. Total risk is:

```text
0.55 × hazard + 0.20 × exposure + 0.25 × vulnerability
```

The weights are product assumptions, not learned parameters. They require expert review for any operational use.

## Limitations

- District centroids do not capture within-district microclimates.
- The demo vulnerability profile is not an official release.
- Unsupervised anomalies can flag benign but unusual patterns.
- A high risk score is not a calibrated probability of harm.
- Historical reanalysis differs from an archived issued forecast.
- Static vulnerability features can become stale.
- Proxy metrics can favor the model that most closely reproduces the proxy definition.

## Monitoring

The current implementation records:

- Data freshness
- Row and district counts
- Missing columns and missingness
- Duplicate district/time rows
- Physical range violations
- Hourly coverage
- Model disagreement
- Reference-data source quality

A production extension should retain run history, monitor distribution drift, and evaluate delayed real outcomes.
