from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from urban_heat_risk.config import Paths, Settings, ensure_directories
from urban_heat_risk.data.demo import generate_demo_hourly
from urban_heat_risk.data.open_meteo import (
    OpenMeteoClient,
    require_valid_weather,
    validate_hourly_weather,
)
from urban_heat_risk.features.daily import (
    create_proxy_extreme_label,
    fit_climatology,
    hourly_to_daily,
    validate_daily_weather,
)
from urban_heat_risk.models.anomaly import evaluate_bundle, fit_model_bundle

Mode = Literal["demo", "live"]


def run_training(
    mode: Mode = "demo",
    paths: Paths | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    paths = paths or Paths()
    settings = settings or Settings.load(paths)
    ensure_directories(paths)
    profiles = pd.read_csv(paths.district_profiles_file)

    if mode == "demo":
        hourly = generate_demo_hourly(
            profiles,
            start_date=settings.history_start,
            end_date=settings.history_end,
            random_state=settings.random_state,
        )
    else:
        client = OpenMeteoClient(
            timeout_seconds=settings.request_timeout_seconds,
            timezone=settings.timezone,
        )
        hourly = client.fetch_historical_hourly(
            profiles, settings.history_start, settings.history_end
        )

    hourly_health = validate_hourly_weather(hourly)
    require_valid_weather(hourly_health, f"{mode} historical")
    daily = hourly_to_daily(hourly)
    daily_health = validate_daily_weather(daily)
    if not daily_health["is_valid"]:
        raise ValueError(f"Daily weather failed validation: {daily_health}")

    max_date = pd.Timestamp(daily["date"].max())
    evaluation_start = max_date - pd.Timedelta(days=365)
    training_raw = daily[daily["date"] < evaluation_start].copy()
    evaluation_raw = daily[daily["date"] >= evaluation_start].copy()
    if training_raw.empty or evaluation_raw.empty:
        raise ValueError("Not enough data for a time-based train/evaluation split")

    evaluation_climatology = fit_climatology(training_raw)
    training_features = evaluation_climatology.apply(training_raw)
    evaluation_features = evaluation_climatology.apply(evaluation_raw)
    evaluation_features["proxy_extreme"] = create_proxy_extreme_label(evaluation_features)

    # Bound robust covariance fitting cost while retaining all seasons and districts.
    fit_frame = _stratified_sample(training_features, max_rows=15000, random_state=settings.random_state)
    evaluation_bundle = fit_model_bundle(
        fit_frame,
        contamination=settings.contamination,
        random_state=settings.random_state,
    )
    evaluation_metrics = evaluate_bundle(evaluation_bundle, evaluation_features)

    final_climatology = fit_climatology(daily)
    full_features = final_climatology.apply(daily)
    final_fit_frame = _stratified_sample(
        full_features, max_rows=18000, random_state=settings.random_state
    )
    final_bundle = fit_model_bundle(
        final_fit_frame,
        contamination=settings.contamination,
        random_state=settings.random_state,
    )
    final_bundle.metadata.update(
        {
            "mode": mode,
            "history_start": settings.history_start,
            "history_end": settings.history_end,
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "evaluation_start": evaluation_start.date().isoformat(),
        }
    )

    joblib.dump(
        {"model_bundle": final_bundle, "climatology": final_climatology},
        paths.model_bundle_file,
        compress=3,
    )
    full_features.to_csv(paths.historical_daily_file, index=False)

    metrics: dict[str, object] = {
        "project": "Urban Heat Risk Intelligence",
        "mode": mode,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_type": "unsupervised anomaly detection",
        "models": ["Isolation Forest", "Robust Covariance"],
        "evaluation": evaluation_metrics,
        "training_rows_total": int(len(full_features)),
        "training_rows_model_fit": int(len(final_fit_frame)),
        "history_start": str(full_features["date"].min()),
        "history_end": str(full_features["date"].max()),
        "proxy_label_warning": (
            "Evaluation uses a climate-extreme proxy derived from historical 98th percentiles; "
            "it is not a health-outcome label."
        ),
        "hourly_data_health": hourly_health,
        "daily_data_health": daily_health,
    }
    _write_json(paths.model_metrics_file, metrics)
    return metrics


def _stratified_sample(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    fraction = max_rows / len(frame)
    sampled_indices: list[int] = []
    for _, group in frame.groupby(["district", "month"], sort=False):
        group_size = max(1, min(len(group), int(round(len(group) * fraction))))
        sampled_indices.extend(
            group.sample(n=group_size, random_state=random_state).index.tolist()
        )
    sampled = frame.loc[sampled_indices]
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
