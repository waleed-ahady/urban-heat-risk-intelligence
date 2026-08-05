from __future__ import annotations

from pathlib import Path

import pandas as pd

PROFILE_COLUMNS = [
    "district",
    "latitude",
    "longitude",
    "population_density_per_km2",
    "share_age_65_plus_pct",
    "green_space_pct",
    "impervious_surface_pct",
    "source_quality",
]

DEMOGRAPHIC_COLUMNS = ["district", "population", "area_km2", "population_age_65_plus"]
ENVIRONMENT_COLUMNS = ["district", "green_space_pct", "impervious_surface_pct"]
COORDINATE_COLUMNS = ["district", "latitude", "longitude"]


def build_district_profiles(
    demographics: pd.DataFrame,
    environment: pd.DataFrame,
    coordinates: pd.DataFrame,
    source_quality: str,
) -> pd.DataFrame:
    """Build the canonical district profile from versioned public-data extracts."""
    _require_columns(demographics, DEMOGRAPHIC_COLUMNS, "demographics")
    _require_columns(environment, ENVIRONMENT_COLUMNS, "environment")
    _require_columns(coordinates, COORDINATE_COLUMNS, "coordinates")

    _require_unique_districts(demographics, "demographics")
    _require_unique_districts(environment, "environment")
    _require_unique_districts(coordinates, "coordinates")

    frame = demographics.merge(environment, on="district", how="inner", validate="one_to_one")
    frame = frame.merge(coordinates, on="district", how="inner", validate="one_to_one")
    expected_districts = set(demographics["district"])
    if set(frame["district"]) != expected_districts:
        missing = sorted(expected_districts - set(frame["district"]))
        raise ValueError(f"Districts missing after profile joins: {missing}")

    if (frame["population"] <= 0).any() or (frame["area_km2"] <= 0).any():
        raise ValueError("Population and area must be positive")
    if (frame["population_age_65_plus"] < 0).any():
        raise ValueError("population_age_65_plus cannot be negative")
    if (frame["population_age_65_plus"] > frame["population"]).any():
        raise ValueError("population_age_65_plus cannot exceed population")

    frame["population_density_per_km2"] = frame["population"] / frame["area_km2"]
    frame["share_age_65_plus_pct"] = 100 * frame["population_age_65_plus"] / frame["population"]
    frame["source_quality"] = source_quality
    result = frame[PROFILE_COLUMNS].sort_values("district").reset_index(drop=True)
    errors = validate_district_profiles(result)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def validate_district_profiles(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = sorted(set(PROFILE_COLUMNS) - set(frame.columns))
    if missing:
        return [f"Missing columns: {missing}"]
    if frame["district"].duplicated().any():
        errors.append("District names must be unique")
    checks = {
        "latitude": frame["latitude"].between(-90, 90),
        "longitude": frame["longitude"].between(-180, 180),
        "population_density_per_km2": frame["population_density_per_km2"] > 0,
        "share_age_65_plus_pct": frame["share_age_65_plus_pct"].between(0, 100),
        "green_space_pct": frame["green_space_pct"].between(0, 100),
        "impervious_surface_pct": frame["impervious_surface_pct"].between(0, 100),
    }
    for column, valid in checks.items():
        if not valid.all():
            errors.append(f"Invalid values in {column}: {int((~valid).sum())} rows")
    if frame[PROFILE_COLUMNS].isna().any().any():
        errors.append("Reference profile contains missing values")
    if frame["source_quality"].astype(str).str.strip().eq("").any():
        errors.append("source_quality must be populated")
    return errors


def read_and_build_profiles(
    demographics_path: str | Path,
    environment_path: str | Path,
    coordinates_path: str | Path,
    source_quality: str,
) -> pd.DataFrame:
    return build_district_profiles(
        demographics=pd.read_csv(demographics_path),
        environment=pd.read_csv(environment_path),
        coordinates=pd.read_csv(coordinates_path),
        source_quality=source_quality,
    )


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _require_unique_districts(frame: pd.DataFrame, name: str) -> None:
    if frame["district"].duplicated().any():
        raise ValueError(f"{name} contains duplicate districts")
