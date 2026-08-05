from __future__ import annotations

import numpy as np
import pandas as pd

RISK_LEVELS = ["Low", "Moderate", "High", "Very High", "Extreme"]


def calculate_risk(
    forecast_scored: pd.DataFrame,
    district_profiles: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or {"hazard": 0.55, "exposure": 0.20, "vulnerability": 0.25}
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Risk weights must sum to 1.0")

    frame = forecast_scored.merge(district_profiles, on="district", how="left", validate="many_to_one")
    profile_columns = [
        "population_density_per_km2",
        "share_age_65_plus_pct",
        "green_space_pct",
        "impervious_surface_pct",
    ]
    if frame[profile_columns].isna().any().any():
        raise ValueError("District profile data is incomplete")

    frame["apparent_heat_score"] = _linear_score(frame["max_apparent_temp"], 24, 42)
    frame["night_heat_score"] = _linear_score(frame["night_min_temperature"], 16, 27)
    frame["duration_score"] = _linear_score(frame["heatwave_streak"], 0, 5)
    frame["heat_hours_score"] = _linear_score(frame["heat_hours_apparent_35"], 0, 12)
    frame["anomaly_component_score"] = frame["anomaly_score"].clip(0, 100)

    frame["hazard_score"] = (
        0.35 * frame["apparent_heat_score"]
        + 0.15 * frame["night_heat_score"]
        + 0.15 * frame["duration_score"]
        + 0.15 * frame["heat_hours_score"]
        + 0.20 * frame["anomaly_component_score"]
    )

    frame["exposure_score"] = _minmax_across_profiles(
        frame["population_density_per_km2"], district_profiles["population_density_per_km2"]
    )
    frame["age_vulnerability_score"] = _minmax_across_profiles(
        frame["share_age_65_plus_pct"], district_profiles["share_age_65_plus_pct"]
    )
    frame["impervious_score"] = _minmax_across_profiles(
        frame["impervious_surface_pct"], district_profiles["impervious_surface_pct"]
    )
    frame["low_green_score"] = 100 - _minmax_across_profiles(
        frame["green_space_pct"], district_profiles["green_space_pct"]
    )
    frame["vulnerability_score"] = (
        0.40 * frame["age_vulnerability_score"]
        + 0.35 * frame["impervious_score"]
        + 0.25 * frame["low_green_score"]
    )

    frame["hazard_contribution"] = weights["hazard"] * frame["hazard_score"]
    frame["exposure_contribution"] = weights["exposure"] * frame["exposure_score"]
    frame["vulnerability_contribution"] = (
        weights["vulnerability"] * frame["vulnerability_score"]
    )
    frame["risk_score"] = (
        frame["hazard_contribution"]
        + frame["exposure_contribution"]
        + frame["vulnerability_contribution"]
    ).clip(0, 100)
    frame["risk_level"] = pd.cut(
        frame["risk_score"],
        bins=[-np.inf, 25, 45, 65, 82, np.inf],
        labels=RISK_LEVELS,
        right=False,
    ).astype(str)
    frame["recommendation"] = frame["risk_level"].map(_recommendation)
    frame["dominant_driver"] = frame.apply(_dominant_driver, axis=1)
    return frame.sort_values(["date", "risk_score"], ascending=[True, False]).reset_index(drop=True)


def _linear_score(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return (100 * (values - lower) / (upper - lower)).clip(0, 100)


def _minmax_across_profiles(values: pd.Series, reference: pd.Series) -> pd.Series:
    low = float(reference.min())
    high = float(reference.max())
    if np.isclose(high, low):
        return pd.Series(50.0, index=values.index)
    return (100 * (values - low) / (high - low)).clip(0, 100)


def _recommendation(level: str) -> str:
    return {
        "Low": "Routine monitoring.",
        "Moderate": "Publish heat-safety guidance and verify cooling resources.",
        "High": "Target warnings; review outdoor work and event schedules.",
        "Very High": "Activate cooling measures and increase outreach to vulnerable groups.",
        "Extreme": "Escalate emergency heat plan; prioritize medical and cooling capacity.",
    }.get(level, "Review conditions.")


def _dominant_driver(row: pd.Series) -> str:
    components = {
        "apparent temperature": float(row["apparent_heat_score"]),
        "warm nights": float(row["night_heat_score"]),
        "heat duration": float(row["duration_score"]),
        "climate anomaly": float(row["anomaly_component_score"]),
        "population exposure": float(row["exposure_score"]),
        "district vulnerability": float(row["vulnerability_score"]),
    }
    return max(components, key=components.get)
