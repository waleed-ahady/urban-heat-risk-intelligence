from __future__ import annotations

import math

import numpy as np
import pandas as pd


def generate_demo_hourly(
    district_profiles: pd.DataFrame,
    start_date: str = "2018-01-01",
    end_date: str = "2025-07-31",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, realistic-enough hourly weather for offline demos and CI."""
    rng = np.random.default_rng(random_state)
    timestamps = pd.date_range(start=start_date, end=f"{end_date} 23:00:00", freq="h")
    day_of_year = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()
    year_fraction = 2 * np.pi * (day_of_year - 172) / 365.25
    diurnal = np.sin(2 * np.pi * (hour - 8) / 24)

    frames: list[pd.DataFrame] = []
    for index, row in enumerate(district_profiles.itertuples(index=False)):
        urban_heat = (row.impervious_surface_pct - row.green_space_pct) * 0.018
        district_offset = (index - 5.5) * 0.025
        seasonal_temp = 10.5 + 10.8 * np.cos(year_fraction)
        temperature = seasonal_temp + 4.2 * diurnal + urban_heat + district_offset
        temperature += rng.normal(0, 1.8, len(timestamps))

        # Inject a small number of coherent summer heatwave episodes.
        years = sorted(set(timestamps.year))
        for year in years:
            year_mask = timestamps.year == year
            summer_positions = np.flatnonzero(
                year_mask & (timestamps.month >= 6) & (timestamps.month <= 8)
            )
            if len(summer_positions) == 0:
                continue
            start = int(rng.choice(summer_positions[: -24 * 7]))
            duration_hours = int(rng.integers(72, 144))
            end = min(start + duration_hours, len(temperature))
            wave = np.sin(np.linspace(0, np.pi, end - start)) * rng.uniform(5.0, 9.0)
            temperature[start:end] += wave

        humidity = 69 - 0.85 * (temperature - 10) - 8 * diurnal
        humidity += rng.normal(0, 6, len(timestamps))
        humidity = np.clip(humidity, 22, 100)
        wind = np.clip(rng.gamma(2.2, 2.0, len(timestamps)), 0.1, 25)
        apparent = temperature + 0.035 * humidity - 0.12 * wind - 1.4
        daylight = np.clip(np.sin(np.pi * (hour - 5) / 15), 0, None)
        summer_factor = np.clip(0.35 + 0.65 * np.cos(year_fraction), 0.05, 1)
        radiation = 720 * daylight * summer_factor + rng.normal(0, 35, len(timestamps))
        radiation = np.clip(radiation, 0, None)
        rain_events = rng.random(len(timestamps)) < 0.045
        precipitation = np.where(rain_events, rng.gamma(1.4, 1.7, len(timestamps)), 0)

        frames.append(
            pd.DataFrame(
                {
                    "district": row.district,
                    "timestamp": timestamps,
                    "temperature_2m": temperature.round(2),
                    "relative_humidity_2m": humidity.round(2),
                    "apparent_temperature": apparent.round(2),
                    "precipitation": precipitation.round(2),
                    "wind_speed_10m": wind.round(2),
                    "shortwave_radiation": radiation.round(2),
                    "source": "synthetic_demo",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def generate_demo_forecast_hourly(
    historical_hourly: pd.DataFrame,
    forecast_days: int = 7,
    random_state: int = 43,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    last_timestamp = pd.Timestamp(historical_hourly["timestamp"].max())
    start = last_timestamp + pd.Timedelta(hours=1)
    end = start + pd.Timedelta(days=forecast_days) - pd.Timedelta(hours=1)
    timestamps = pd.date_range(start, end, freq="h")
    frames: list[pd.DataFrame] = []

    for district, district_history in historical_hourly.groupby("district", sort=False):
        recent = district_history.sort_values("timestamp").tail(24 * 30)
        base_by_hour = recent.groupby(recent["timestamp"].dt.hour).mean(numeric_only=True)
        rows = []
        for ts in timestamps:
            base = base_by_hour.loc[ts.hour]
            warming = 0.7 * math.sin((ts - start).total_seconds() / 86400 * math.pi / 3)
            # Deliberately create a forecast heat episode around days 3-5.
            day_index = (ts.date() - start.date()).days
            event = 15.0 * max(0.0, math.sin(math.pi * (day_index - 1) / 5))
            temp = float(base["temperature_2m"] + warming + event + rng.normal(0, 0.5))
            humidity = float(np.clip(base["relative_humidity_2m"] - event * 1.3, 25, 95))
            wind = float(max(0.2, base["wind_speed_10m"] + rng.normal(0, 0.5)))
            apparent = temp + 0.035 * humidity - 0.12 * wind - 1.4
            rows.append(
                {
                    "district": district,
                    "timestamp": ts,
                    "temperature_2m": temp,
                    "relative_humidity_2m": humidity,
                    "apparent_temperature": apparent,
                    "precipitation": max(0.0, float(base["precipitation"] * 0.5)),
                    "wind_speed_10m": wind,
                    "shortwave_radiation": max(0.0, float(base["shortwave_radiation"])),
                    "source": "synthetic_demo_forecast",
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)
