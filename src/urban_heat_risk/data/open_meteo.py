from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    timeout_seconds: float = 60.0
    timezone: str = "Europe/Berlin"
    max_retries: int = 6
    retry_base_seconds: float = 10.0
    request_pause_seconds: float = 5.0
    cache_dir: Path = Path("data/cache/open_meteo")

    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"

    def fetch_historical_hourly(
        self,
        districts: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Download historical data district by district with persistent caching.

        Successful district downloads are saved under ``data/cache/open_meteo``.
        If a later request is rate-limited, rerunning the command resumes from
        the first district that is not already cached.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frames: list[pd.DataFrame] = []

        for index, row in enumerate(districts.itertuples(index=False)):
            params = {
                "latitude": row.latitude,
                "longitude": row.longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(HOURLY_VARIABLES),
                "timezone": self.timezone,
            }
            cache_file = self._historical_cache_file(
                district=str(row.district),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                start_date=start_date,
                end_date=end_date,
            )

            if cache_file.exists():
                frame = pd.read_csv(cache_file, parse_dates=["timestamp"])
            else:
                payload = self._get_json(self.archive_url, params)
                frame = self._hourly_payload_to_frame(
                    payload, str(row.district), "observed"
                )
                frame.to_csv(cache_file, index=False)

                # Avoid sending the next large archive request immediately.
                if index < len(districts) - 1:
                    time.sleep(self.request_pause_seconds)

            frames.append(frame)

        if not frames:
            raise WeatherAPIError("No historical weather frames were produced")
        return pd.concat(frames, ignore_index=True)

    def fetch_forecast_hourly(
        self,
        districts: pd.DataFrame,
        forecast_days: int = 7,
        past_days: int = 14,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for index, row in enumerate(districts.itertuples(index=False)):
            params = {
                "latitude": row.latitude,
                "longitude": row.longitude,
                "hourly": ",".join(HOURLY_VARIABLES),
                "timezone": self.timezone,
                "forecast_days": forecast_days,
                "past_days": past_days,
            }
            payload = self._get_json(self.forecast_url, params)
            frames.append(
                self._hourly_payload_to_frame(
                    payload, str(row.district), "forecast"
                )
            )
            if index < len(districts) - 1:
                time.sleep(1.0)

        if not frames:
            raise WeatherAPIError("No forecast weather frames were produced")
        return pd.concat(frames, ignore_index=True)

    def _get_json(self, url: str, params: dict[str, object]) -> dict[str, Any]:
        """Request JSON with Retry-After support and exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.get(
                        url,
                        params=params,
                        headers={"User-Agent": "urban-heat-risk-intelligence/1.0"},
                    )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = self._retry_delay(attempt, retry_after)
                    if attempt >= self.max_retries:
                        raise WeatherAPIError(
                            "Open-Meteo rate limit persisted after retries. "
                            "Rerun the command later; completed districts are cached."
                        )
                    time.sleep(delay)
                    continue

                if 500 <= response.status_code < 600:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    time.sleep(self._retry_delay(attempt, None))
                    continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise WeatherAPIError("Open-Meteo returned a non-object JSON response")

                if "hourly" not in payload:
                    reason = payload.get("reason", "missing hourly payload")
                    raise WeatherAPIError(f"Open-Meteo response is invalid: {reason}")
                return payload

            except WeatherAPIError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self._retry_delay(attempt, None))

        raise WeatherAPIError(f"Open-Meteo request failed: {last_error}") from last_error

    def _historical_cache_file(
        self,
        district: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> Path:
        safe_district = "".join(
            character if character.isalnum() else "_" for character in district
        ).strip("_")
        key_payload = {
            "district": district,
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": HOURLY_VARIABLES,
            "timezone": self.timezone,
        }
        digest = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return self.cache_dir / f"historical_{safe_district}_{digest}.csv"

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
        return self.retry_base_seconds * (2**attempt)

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