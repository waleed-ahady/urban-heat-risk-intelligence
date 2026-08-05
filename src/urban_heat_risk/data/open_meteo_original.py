from __future__ import annotations

from dataclasses import dataclass

import httpx
import pandas as pd

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
]


class WeatherAPIError(RuntimeError):
    """Raised when weather data cannot be retrieved or parsed."""


@dataclass(frozen=True)
class OpenMeteoClient:
    timeout_seconds: float = 30.0
    timezone: str = "Europe/Berlin"

    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"

    def fetch_historical_hourly(
        self,
        districts: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        frames = []
        for row in districts.itertuples(index=False):
            params = {
                "latitude": row.latitude,
                "longitude": row.longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(HOURLY_VARIABLES),
                "timezone": self.timezone,
            }
            payload = self._get_json(self.archive_url, params)
            frames.append(self._hourly_payload_to_frame(payload, row.district, "observed"))
        return pd.concat(frames, ignore_index=True)

    def fetch_forecast_hourly(
        self,
        districts: pd.DataFrame,
        forecast_days: int = 7,
        past_days: int = 14,
    ) -> pd.DataFrame:
        frames = []
        for row in districts.itertuples(index=False):
            params = {
                "latitude": row.latitude,
                "longitude": row.longitude,
                "hourly": ",".join(HOURLY_VARIABLES),
                "timezone": self.timezone,
                "forecast_days": forecast_days,
                "past_days": past_days,
            }
            payload = self._get_json(self.forecast_url, params)
            frames.append(self._hourly_payload_to_frame(payload, row.district, "forecast"))
        return pd.concat(frames, ignore_index=True)

    def _get_json(self, url: str, params: dict[str, object]) -> dict[str, object]:
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherAPIError(f"Open-Meteo request failed: {exc}") from exc
        if "hourly" not in payload:
            reason = payload.get("reason", "missing hourly payload")
            raise WeatherAPIError(f"Open-Meteo response is invalid: {reason}")
        return payload

    @staticmethod
    def _hourly_payload_to_frame(
        payload: dict[str, object],
        district: str,
        source: str,
    ) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise WeatherAPIError("Hourly response is malformed")
        frame = pd.DataFrame(hourly)
        missing = set(HOURLY_VARIABLES + ["time"]) - set(frame.columns)
        if missing:
            raise WeatherAPIError(f"Hourly response is missing columns: {sorted(missing)}")
        frame = frame.rename(columns={"time": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["district"] = district
        frame["source"] = source
        return frame[["district", "timestamp", *HOURLY_VARIABLES, "source"]]


def validate_hourly_weather(frame: pd.DataFrame) -> dict[str, object]:
    required = {"district", "timestamp", *HOURLY_VARIABLES}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns or frame.empty:
        duplicate_rows = 0
        missing_fraction = 1.0
        range_violations: dict[str, int] = {}
    else:
        duplicate_rows = int(frame.duplicated(["district", "timestamp"]).sum())
        missing_fraction = float(frame[list(required)].isna().mean().mean())
        range_violations = {
            "temperature_2m": int((~frame["temperature_2m"].between(-60, 60)).sum()),
            "relative_humidity_2m": int(
                (~frame["relative_humidity_2m"].between(0, 100)).sum()
            ),
            "wind_speed_10m": int((frame["wind_speed_10m"] < 0).sum()),
            "precipitation": int((frame["precipitation"] < 0).sum()),
        }

    is_valid = (
        not missing_columns
        and duplicate_rows == 0
        and missing_fraction == 0.0
        and not any(range_violations.values())
    )
    return {
        "is_valid": bool(is_valid),
        "row_count": int(len(frame)),
        "district_count": int(frame["district"].nunique()) if "district" in frame else 0,
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_rows,
        "missing_fraction": missing_fraction,
        "range_violations": range_violations,
    }


def require_valid_weather(health: dict[str, object], context: str) -> None:
    if not health.get("is_valid", False):
        raise ValueError(f"{context} weather failed validation: {health}")
