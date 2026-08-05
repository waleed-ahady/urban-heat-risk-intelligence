# Data Dictionary

## Hourly weather

| Column | Type | Description |
|---|---|---|
| `district` | string | Berlin district name |
| `timestamp` | datetime | Local hourly timestamp |
| `temperature_2m` | float | Air temperature at 2 m, °C |
| `relative_humidity_2m` | float | Relative humidity, % |
| `apparent_temperature` | float | Provider apparent temperature, °C |
| `precipitation` | float | Hourly precipitation, mm |
| `wind_speed_10m` | float | Wind speed at 10 m, km/h |
| `shortwave_radiation` | float | Shortwave radiation, W/m² |
| `source` | string | Data mode or provider marker |

## Daily engineered features

| Column | Description |
|---|---|
| `max_apparent_temp` | Daily maximum apparent temperature |
| `night_min_temperature` | Minimum temperature from 00:00–05:59 |
| `heat_hours_30` | Hours with air temperature at least 30°C |
| `heat_hours_apparent_35` | Hours with apparent temperature at least 35°C |
| `heatwave_streak` | Consecutive daily maximum temperatures at least 30°C |
| `*_anomaly_z` | Robust district/month anomaly based on median and MAD |
| `hourly_coverage` | Number of hourly records in the daily aggregate |

## Anomaly outputs

| Column | Description |
|---|---|
| `isolation_forest_raw` | Native oriented anomaly score; higher is more anomalous |
| `isolation_forest_percentile` | Empirical percentile versus model-fit scores |
| `robust_covariance_raw` | Native oriented robust-distance anomaly score |
| `robust_covariance_percentile` | Empirical percentile versus model-fit scores |
| `anomaly_score` | Weighted ensemble on a 0–100 scale |
| `model_disagreement` | Absolute model percentile difference × 100 |

## Risk outputs

| Column | Description |
|---|---|
| `hazard_score` | Weather severity and anomaly component |
| `exposure_score` | Relative population-density score |
| `vulnerability_score` | Age, impervious surface, and low-green-space score |
| `risk_score` | Weighted 0–100 prioritization index |
| `risk_level` | Low, Moderate, High, Very High, or Extreme |
| `dominant_driver` | Largest normalized driver |
| `recommendation` | Generic planning response text |
