from __future__ import annotations

import pandas as pd

from urban_heat_risk.data.open_meteo import validate_hourly_weather


def test_hourly_validation_detects_bad_humidity() -> None:
    frame = pd.DataFrame(
        {
            "district": ["A"],
            "timestamp": pd.to_datetime(["2026-07-01 00:00"]),
            "temperature_2m": [25.0],
            "relative_humidity_2m": [130.0],
            "apparent_temperature": [26.0],
            "precipitation": [0.0],
            "wind_speed_10m": [3.0],
            "shortwave_radiation": [0.0],
        }
    )
    health = validate_hourly_weather(frame)
    assert health["is_valid"] is False
    assert health["range_violations"]["relative_humidity_2m"] == 1


def test_hourly_validation_handles_missing_columns_without_crashing() -> None:
    frame = pd.DataFrame({"district": ["A"]})
    health = validate_hourly_weather(frame)
    assert health["is_valid"] is False
    assert "timestamp" in health["missing_columns"]
