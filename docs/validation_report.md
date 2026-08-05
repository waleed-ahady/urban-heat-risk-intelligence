# Validation Report

Generated from the bundled deterministic demo pipeline.

## Automated checks

- Unit tests: 8 passed
- Python compile check: passed
- District profile validation: passed
- Hourly validation: True
- Daily validation: True
- Districts scored: 12
- Forecast rows: 84

## Time-holdout anomaly benchmark

| Model | ROC AUC | Average precision | Precision at top 3% |
|---|---:|---:|---:|
| Isolation Forest | 0.957 | 0.543 | 0.561 |
| Robust Covariance | 0.943 | 0.472 | 0.492 |
| Ensemble | 0.960 | 0.567 | 0.606 |

Model score correlation: 0.874.

## Demo output coverage

Risk-level counts: {'Moderate': 34, 'High': 33, 'Very High': 13, 'Low': 3, 'Extreme': 1}.

Peak demo risk: 83.2/100. Peak apparent temperature: 39.4 °C.

## Interpretation warning

Evaluation uses a climate-extreme proxy derived from historical 98th percentiles; it is not a health-outcome label. The reference vulnerability profile and weather inputs in the bundled artifacts are demo data.
