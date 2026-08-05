from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from urban_heat_risk.config import Paths

_PATHS = Paths()
RISK_FILE = _PATHS.latest_risk_file
HISTORY_FILE = _PATHS.historical_daily_file
METRICS_FILE = _PATHS.model_metrics_file
HEALTH_FILE = _PATHS.data_health_file
PROFILE_FILE = _PATHS.district_profiles_file


@st.cache_data(ttl=900)
def load_risk() -> pd.DataFrame:
    _require_file(RISK_FILE)
    frame = pd.read_csv(RISK_FILE, parse_dates=["date"])
    return frame.sort_values(["date", "risk_score"], ascending=[True, False])


@st.cache_data(ttl=3600)
def load_history() -> pd.DataFrame:
    _require_file(HISTORY_FILE)
    return pd.read_csv(HISTORY_FILE, parse_dates=["date"])


@st.cache_data(ttl=3600)
def load_profiles() -> pd.DataFrame:
    _require_file(PROFILE_FILE)
    return pd.read_csv(PROFILE_FILE)


@st.cache_data(ttl=900)
def load_json(path: Path) -> dict[str, object]:
    _require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def setup_page(title: str, icon: str = "🌡️") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        .small-note {font-size: 0.86rem; color: #5f6b76;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_data_mode_notice(risk: pd.DataFrame) -> None:
    mode = str(risk.get("pipeline_mode", pd.Series(["unknown"])).iloc[0])
    source_quality = str(risk.get("source_quality", pd.Series(["unknown"])).iloc[0])
    if mode == "demo" or "demo_seed" in source_quality:
        st.warning(
            "Demo mode is active. Weather is synthetic and the district vulnerability profile is "
            "starter data. Use the live pipeline and official Berlin aggregations for real analysis."
        )


def risk_level_order() -> list[str]:
    return ["Low", "Moderate", "High", "Very High", "Extreme"]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact is missing: {path}. Run `make demo` or the live pipelines first."
        )
