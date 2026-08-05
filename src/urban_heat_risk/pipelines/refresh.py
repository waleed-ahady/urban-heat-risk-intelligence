from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from urban_heat_risk.config import Paths, Settings, ensure_directories
from urban_heat_risk.data.demo import generate_demo_forecast_hourly
from urban_heat_risk.data.open_meteo import (
    OpenMeteoClient,
    require_valid_weather,
    validate_hourly_weather,
)
from urban_heat_risk.features.daily import hourly_to_daily, validate_daily_weather
from urban_heat_risk.risk.engine import calculate_risk

Mode = Literal["demo", "live"]


def run_refresh(
    mode: Mode = "demo",
    paths: Paths | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    paths = paths or Paths()
    settings = settings or Settings.load(paths)
    ensure_directories(paths)
    profiles = pd.read_csv(paths.district_profiles_file)
    if not paths.model_bundle_file.exists():
        raise FileNotFoundError("Model bundle is missing. Run the training pipeline first.")
    saved = joblib.load(paths.model_bundle_file)
    model_bundle = saved["model_bundle"]
    climatology = saved["climatology"]

    historical_daily = pd.read_csv(paths.historical_daily_file, parse_dates=["date"])
    context_start = historical_daily["date"].max() - pd.Timedelta(days=20)

    if mode == "demo":
        from urban_heat_risk.data.demo import generate_demo_hourly

        context_hourly = generate_demo_hourly(
            profiles,
            start_date=context_start.date().isoformat(),
            end_date=historical_daily["date"].max().date().isoformat(),
            random_state=settings.random_state,
        )
        forecast_hourly = generate_demo_forecast_hourly(
            context_hourly,
            forecast_days=settings.forecast_days,
            random_state=settings.random_state + 1,
        )
        combined_hourly = pd.concat([context_hourly, forecast_hourly], ignore_index=True)
        forecast_start = pd.Timestamp(forecast_hourly["timestamp"].min()).normalize()
    else:
        client = OpenMeteoClient(
            timeout_seconds=settings.request_timeout_seconds,
            timezone=settings.timezone,
        )
        combined_hourly = client.fetch_forecast_hourly(
            profiles,
            forecast_days=settings.forecast_days,
            past_days=20,
        )
        today_local = pd.Timestamp.now(tz=settings.timezone).tz_localize(None).normalize()
        forecast_start = today_local

    hourly_health = validate_hourly_weather(combined_hourly)
    require_valid_weather(hourly_health, f"{mode} forecast")
    combined_daily = hourly_to_daily(combined_hourly)
    daily_health = validate_daily_weather(combined_daily)
    if not daily_health["is_valid"]:
        raise ValueError(f"Forecast daily data failed validation: {daily_health}")

    forecast_daily = combined_daily[combined_daily["date"] >= forecast_start].copy()
    forecast_features = climatology.apply(forecast_daily)
    scored = model_bundle.score(forecast_features)
    risk = calculate_risk(scored, profiles, settings.risk_weights)
    risk["generated_at_utc"] = datetime.now(UTC).isoformat()
    risk["pipeline_mode"] = mode

    forecast_features.to_csv(paths.forecast_daily_file, index=False)
    risk.to_csv(paths.latest_risk_file, index=False)
    data_health: dict[str, object] = {
        "mode": mode,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "forecast_start": str(risk["date"].min()),
        "forecast_end": str(risk["date"].max()),
        "risk_rows": int(len(risk)),
        "districts": int(risk["district"].nunique()),
        "hourly": hourly_health,
        "daily": daily_health,
        "profile_source_quality": profiles["source_quality"].value_counts().to_dict(),
        "max_model_disagreement": float(risk["model_disagreement"].max()),
    }
    _write_json(paths.data_health_file, data_health)
    return data_health


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
