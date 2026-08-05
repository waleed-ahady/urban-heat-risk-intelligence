from __future__ import annotations

import numpy as np
import pandas as pd

from urban_heat_risk.features.daily import (
    ANOMALY_FEATURES,
    fit_climatology,
    hourly_to_daily,
)


def _hourly_fixture(days: int = 70) -> pd.DataFrame:
    timestamps = pd.date_range("2024-05-01", periods=24 * days, freq="h")
    hour = timestamps.hour.to_numpy()
    day = np.arange(len(timestamps)) / 24
    temperature = 20 + 7 * np.sin(2 * np.pi * (hour - 8) / 24) + 0.03 * day
    return pd.DataFrame(
        {
            "district": "Test District",
            "timestamp": timestamps,
            "temperature_2m": temperature,
            "relative_humidity_2m": 55.0,
            "apparent_temperature": temperature + 1.5,
            "precipitation": 0.0,
            "wind_speed_10m": 4.0,
            "shortwave_radiation": np.clip(600 * np.sin(np.pi * (hour - 5) / 15), 0, None),
            "source": "test",
        }
    )


def test_hourly_to_daily_creates_expected_features() -> None:
    daily = hourly_to_daily(_hourly_fixture())
    assert len(daily) == 70
    assert daily["hourly_coverage"].eq(24).all()
    assert {"heatwave_streak", "night_min_temperature", "max_apparent_temp"}.issubset(daily)


def test_climatology_produces_finite_anomaly_features() -> None:
    daily = hourly_to_daily(_hourly_fixture())
    climatology = fit_climatology(daily)
    enriched = climatology.apply(daily)
    assert enriched[ANOMALY_FEATURES].notna().all().all()
    assert np.isfinite(enriched[ANOMALY_FEATURES].to_numpy()).all()
