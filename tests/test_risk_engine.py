from __future__ import annotations

import pandas as pd

from urban_heat_risk.risk.engine import calculate_risk


def test_risk_increases_for_hotter_more_vulnerable_district() -> None:
    forecast = pd.DataFrame(
        {
            "district": ["Cool", "Hot"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "max_apparent_temp": [27.0, 42.0],
            "night_min_temperature": [17.0, 27.0],
            "heatwave_streak": [0, 5],
            "heat_hours_apparent_35": [0, 14],
            "anomaly_score": [20.0, 99.0],
        }
    )
    profiles = pd.DataFrame(
        {
            "district": ["Cool", "Hot"],
            "latitude": [52.4, 52.5],
            "longitude": [13.2, 13.4],
            "population_density_per_km2": [1800, 12000],
            "share_age_65_plus_pct": [12.0, 25.0],
            "green_space_pct": [60.0, 15.0],
            "impervious_surface_pct": [25.0, 80.0],
            "source_quality": ["test", "test"],
        }
    )
    result = calculate_risk(forecast, profiles)
    cool_score = result.loc[result["district"] == "Cool", "risk_score"].iloc[0]
    hot_score = result.loc[result["district"] == "Hot", "risk_score"].iloc[0]
    assert hot_score > cool_score
    assert result["risk_score"].between(0, 100).all()
