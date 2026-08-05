from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANOMALY_FEATURES = [
    "apparent_anomaly_z",
    "temperature_anomaly_z",
    "night_anomaly_z",
    "humidity_anomaly_z",
    "radiation_anomaly_z",
    "heat_hours_apparent_35_scaled",
    "heatwave_streak_scaled",
]

CLIMATOLOGY_METRICS = [
    "max_apparent_temp",
    "max_temperature",
    "night_min_temperature",
    "mean_humidity",
    "max_shortwave_radiation",
]


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Hourly weather contains invalid timestamps")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["night_temperature"] = frame["temperature_2m"].where(frame["hour"].between(0, 5))
    frame["heat_hour_30"] = (frame["temperature_2m"] >= 30).astype(int)
    frame["heat_hour_apparent_35"] = (frame["apparent_temperature"] >= 35).astype(int)

    daily = (
        frame.groupby(["district", "date"], as_index=False)
        .agg(
            max_apparent_temp=("apparent_temperature", "max"),
            mean_apparent_temp=("apparent_temperature", "mean"),
            max_temperature=("temperature_2m", "max"),
            min_temperature=("temperature_2m", "min"),
            night_min_temperature=("night_temperature", "min"),
            mean_humidity=("relative_humidity_2m", "mean"),
            mean_wind_speed=("wind_speed_10m", "mean"),
            max_shortwave_radiation=("shortwave_radiation", "max"),
            total_precipitation=("precipitation", "sum"),
            heat_hours_30=("heat_hour_30", "sum"),
            heat_hours_apparent_35=("heat_hour_apparent_35", "sum"),
            weather_source=("source", "first"),
            hourly_coverage=("timestamp", "count"),
        )
        .sort_values(["district", "date"])
        .reset_index(drop=True)
    )
    daily["night_min_temperature"] = daily["night_min_temperature"].fillna(
        daily["min_temperature"]
    )
    daily["hot_day"] = daily["max_temperature"] >= 30
    daily["heatwave_streak"] = (
        daily.groupby("district", group_keys=False)["hot_day"]
        .apply(_consecutive_true_count)
        .astype(int)
        .to_numpy()
    )
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    return daily


def _consecutive_true_count(values: pd.Series) -> pd.Series:
    groups = (~values).cumsum()
    return values.astype(int).groupby(groups).cumsum()


@dataclass(frozen=True)
class Climatology:
    table: pd.DataFrame

    def apply(self, daily: pd.DataFrame) -> pd.DataFrame:
        frame = daily.merge(self.table, on=["district", "month"], how="left", validate="many_to_one")
        missing_baseline = frame[[f"{metric}_median" for metric in CLIMATOLOGY_METRICS]].isna().any(axis=1)
        if missing_baseline.any():
            districts = sorted(frame.loc[missing_baseline, "district"].unique())
            raise ValueError(f"No climatology is available for districts: {districts}")

        frame["apparent_anomaly_z"] = _robust_z(
            frame["max_apparent_temp"],
            frame["max_apparent_temp_median"],
            frame["max_apparent_temp_mad"],
        ).clip(lower=0)
        frame["temperature_anomaly_z"] = _robust_z(
            frame["max_temperature"],
            frame["max_temperature_median"],
            frame["max_temperature_mad"],
        ).clip(lower=0)
        frame["night_anomaly_z"] = _robust_z(
            frame["night_min_temperature"],
            frame["night_min_temperature_median"],
            frame["night_min_temperature_mad"],
        ).clip(lower=0)
        frame["humidity_anomaly_z"] = _robust_z(
            frame["mean_humidity"],
            frame["mean_humidity_median"],
            frame["mean_humidity_mad"],
        ).clip(lower=0)
        frame["radiation_anomaly_z"] = _robust_z(
            frame["max_shortwave_radiation"],
            frame["max_shortwave_radiation_median"],
            frame["max_shortwave_radiation_mad"],
        ).clip(lower=0)
        frame["heat_hours_apparent_35_scaled"] = np.clip(
            frame["heat_hours_apparent_35"] / 12.0, 0, 2
        )
        frame["heatwave_streak_scaled"] = np.clip(frame["heatwave_streak"] / 5.0, 0, 2)
        return frame


def fit_climatology(historical_daily: pd.DataFrame) -> Climatology:
    rows: list[dict[str, object]] = []
    for (district, month), group in historical_daily.groupby(["district", "month"]):
        row: dict[str, object] = {"district": district, "month": int(month)}
        for metric in CLIMATOLOGY_METRICS:
            values = group[metric].dropna().astype(float)
            median = float(values.median())
            mad = float((values - median).abs().median())
            row[f"{metric}_median"] = median
            row[f"{metric}_mad"] = max(mad, 0.25)
            row[f"{metric}_p98"] = float(values.quantile(0.98))
        rows.append(row)
    return Climatology(pd.DataFrame(rows))


def create_proxy_extreme_label(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["max_apparent_temp"] >= frame["max_apparent_temp_p98"])
        | (frame["night_min_temperature"] >= frame["night_min_temperature_p98"])
    ).astype(int)


def validate_daily_weather(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "district",
        "date",
        "max_apparent_temp",
        "max_temperature",
        "night_min_temperature",
        "mean_humidity",
        "hourly_coverage",
    }
    missing = sorted(required - set(frame.columns))
    coverage_failures = int((frame["hourly_coverage"] < 20).sum()) if not missing else 0
    duplicate_rows = int(frame.duplicated(["district", "date"]).sum()) if not missing else 0
    missing_fraction = float(frame[list(required)].isna().mean().mean()) if not missing else 1.0
    return {
        "is_valid": (
            not missing
            and duplicate_rows == 0
            and coverage_failures == 0
            and missing_fraction == 0.0
        ),
        "row_count": int(len(frame)),
        "missing_columns": missing,
        "coverage_failures": coverage_failures,
        "duplicate_rows": duplicate_rows,
        "missing_fraction": missing_fraction,
    }


def _robust_z(values: pd.Series, medians: pd.Series, mads: pd.Series) -> pd.Series:
    return ((values - medians) / (1.4826 * mads.clip(lower=0.25))).clip(-8, 8)
