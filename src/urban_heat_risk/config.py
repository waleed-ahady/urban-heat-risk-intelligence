from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    configured = os.getenv("UHR_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    working_directory = Path.cwd().resolve()
    if (working_directory / "configs" / "settings.json").exists():
        return working_directory
    return PACKAGE_DIR.parents[1]


PROJECT_ROOT = _resolve_project_root()


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def reference_dir(self) -> Path:
        return self.root / "data" / "reference"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def settings_file(self) -> Path:
        return self.root / "configs" / "settings.json"

    @property
    def district_profiles_file(self) -> Path:
        return self.reference_dir / "district_profiles.csv"

    @property
    def historical_daily_file(self) -> Path:
        return self.processed_dir / "historical_daily.csv"

    @property
    def forecast_daily_file(self) -> Path:
        return self.processed_dir / "forecast_daily.csv"

    @property
    def latest_risk_file(self) -> Path:
        return self.processed_dir / "latest_risk.csv"

    @property
    def data_health_file(self) -> Path:
        return self.processed_dir / "data_health.json"

    @property
    def model_bundle_file(self) -> Path:
        return self.artifacts_dir / "anomaly_models.joblib"

    @property
    def model_metrics_file(self) -> Path:
        return self.artifacts_dir / "model_metrics.json"


@dataclass(frozen=True)
class Settings:
    history_start: str
    history_end: str
    forecast_days: int
    timezone: str
    contamination: float
    random_state: int
    risk_weights: dict[str, float]
    request_timeout_seconds: float = 30.0

    @classmethod
    def load(cls, paths: Paths | None = None) -> Settings:
        paths = paths or Paths()
        with paths.settings_file.open(encoding="utf-8") as file:
            raw: dict[str, Any] = json.load(file)
        return cls(
            history_start=os.getenv("UHR_HISTORY_START", raw["history_start"]),
            history_end=os.getenv("UHR_HISTORY_END", raw["history_end"]),
            forecast_days=int(os.getenv("UHR_FORECAST_DAYS", raw["forecast_days"])),
            timezone=raw["timezone"],
            contamination=float(raw["contamination"]),
            random_state=int(raw["random_state"]),
            risk_weights={key: float(value) for key, value in raw["risk_weights"].items()},
            request_timeout_seconds=float(os.getenv("UHR_REQUEST_TIMEOUT_SECONDS", "30")),
        )


def ensure_directories(paths: Paths | None = None) -> None:
    paths = paths or Paths()
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
