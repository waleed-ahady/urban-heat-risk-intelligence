from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from urban_heat_risk.dashboard import HEALTH_FILE, load_json, load_risk, setup_page

setup_page("System Status", "🩺")
st.title("System Status")
st.caption(
    "Operational monitoring for data freshness, validation, coverage, model agreement, "
    "and reference-data provenance."
)

health = load_json(HEALTH_FILE)
risk = load_risk()


def _as_utc(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _age_hours(value: Any) -> float | None:
    parsed = _as_utc(value)
    if parsed is None:
        return None
    return max(0.0, (pd.Timestamp.now(tz="UTC") - parsed).total_seconds() / 3600)


def _status_label(level: str) -> str:
    return {"healthy": "Healthy", "warning": "Warning", "critical": "Critical"}[level]


def _status_icon(level: str) -> str:
    return {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}[level]


def _validation_issues(layer: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not bool(layer.get("is_valid", False)):
        issues.append("validation failed")
    missing_columns = layer.get("missing_columns", []) or []
    if missing_columns:
        issues.append(f"missing columns: {', '.join(map(str, missing_columns))}")
    if int(layer.get("duplicate_rows", 0) or 0) > 0:
        issues.append(f"{int(layer['duplicate_rows'])} duplicate rows")
    if float(layer.get("missing_fraction", 0.0) or 0.0) > 0:
        issues.append(f"{float(layer['missing_fraction']):.2%} missing values")
    if int(layer.get("coverage_failures", 0) or 0) > 0:
        issues.append(f"{int(layer['coverage_failures'])} coverage failures")
    violations = layer.get("range_violations", {}) or {}
    bad_ranges = {key: value for key, value in violations.items() if int(value or 0) > 0}
    if bad_ranges:
        issues.append(
            "range violations: " + ", ".join(f"{key}={value}" for key, value in bad_ranges.items())
        )
    return issues


def _system_assessment() -> tuple[str, list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []

    hourly = health.get("hourly", {}) or {}
    daily = health.get("daily", {}) or {}

    hourly_issues = _validation_issues(hourly)
    daily_issues = _validation_issues(daily)
    if hourly_issues:
        critical.append("Hourly data: " + "; ".join(hourly_issues))
    if daily_issues:
        critical.append("Daily data: " + "; ".join(daily_issues))

    districts = int(health.get("districts", 0) or 0)
    if districts != 12:
        critical.append(f"Expected 12 districts, received {districts}.")

    risk_rows = int(health.get("risk_rows", 0) or 0)
    if risk_rows == 0:
        critical.append("No risk output rows were generated.")

    generated_age = _age_hours(health.get("generated_at_utc"))
    if generated_age is None:
        critical.append("Generation timestamp is missing or invalid.")
    elif generated_age > 48:
        critical.append(f"Pipeline output is {generated_age:.1f} hours old.")
    elif generated_age > 24:
        warnings.append(f"Pipeline output is {generated_age:.1f} hours old.")

    forecast_end = _as_utc(health.get("forecast_end"))
    if forecast_end is None:
        critical.append("Forecast end date is missing or invalid.")
    else:
        remaining_days = (forecast_end.normalize() - pd.Timestamp.now(tz="UTC").normalize()).days
        if remaining_days < 0:
            critical.append("Forecast coverage has expired.")
        elif remaining_days < 2:
            warnings.append(f"Only {remaining_days + 1} forecast day(s) remain.")

    mode = str(health.get("mode", "unknown")).lower()
    if mode == "demo":
        warnings.append("Synthetic demo weather is active.")

    source_quality = health.get("profile_source_quality", {}) or {}
    source_labels = " ".join(map(str, source_quality.keys())).lower()
    if "demo" in source_labels or "replace" in source_labels:
        warnings.append("District vulnerability inputs are demo seed data.")
    elif not source_quality:
        warnings.append("Reference-data provenance is not recorded.")

    disagreement = float(health.get("max_model_disagreement", 0.0) or 0.0)
    if disagreement >= 40:
        warnings.append(f"Maximum model disagreement is high ({disagreement:.1f}/100).")

    if critical:
        return "critical", critical, warnings
    if warnings:
        return "warning", critical, warnings
    return "healthy", critical, warnings


status, critical_issues, warning_issues = _system_assessment()
generated_at = _as_utc(health.get("generated_at_utc"))
generated_age = _age_hours(health.get("generated_at_utc"))
forecast_start = _as_utc(health.get("forecast_start"))
forecast_end = _as_utc(health.get("forecast_end"))

status_text = f"{_status_icon(status)} {_status_label(status)}"
if status == "healthy":
    st.success(f"Overall status: **{status_text}** — all monitored checks are within limits.")
elif status == "warning":
    st.warning(
        f"Overall status: **{status_text}** — outputs are usable, but review the warnings below."
    )
else:
    st.error(
        f"Overall status: **{status_text}** — do not rely on the current dashboard output until resolved."
    )

kpi_cols = st.columns(5)
kpi_cols[0].metric("Overall status", _status_label(status))
kpi_cols[1].metric("Pipeline mode", str(health.get("mode", "unknown")).title())
kpi_cols[2].metric("District coverage", f"{int(health.get('districts', 0) or 0)}/12")
kpi_cols[3].metric("Risk output rows", f"{int(health.get('risk_rows', 0) or 0):,}")
kpi_cols[4].metric(
    "Output age",
    "Unknown" if generated_age is None else f"{generated_age:.1f} h",
    delta=None if generated_age is None else ("Fresh" if generated_age <= 24 else "Review"),
    delta_color="off",
)

if critical_issues or warning_issues:
    st.subheader("Items requiring attention")
    for issue in critical_issues:
        st.error(issue)
    for issue in warning_issues:
        st.warning(issue)

st.subheader("Pipeline checks")
hourly = health.get("hourly", {}) or {}
daily = health.get("daily", {}) or {}
check_rows = []
for layer_name, layer in (("Hourly ingestion", hourly), ("Daily features", daily)):
    issues = _validation_issues(layer)
    check_rows.append(
        {
            "Layer": layer_name,
            "Status": "Pass" if not issues else "Fail",
            "Rows": int(layer.get("row_count", 0) or 0),
            "Missing values": f"{float(layer.get('missing_fraction', 0.0) or 0.0):.2%}",
            "Duplicates": int(layer.get("duplicate_rows", 0) or 0),
            "Coverage failures": int(layer.get("coverage_failures", 0) or 0),
            "Details": "No issues detected" if not issues else "; ".join(issues),
        }
    )
st.dataframe(pd.DataFrame(check_rows), hide_index=True, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Freshness and coverage")
    freshness_rows = [
        {
            "Metric": "Generated at",
            "Value": generated_at.strftime("%Y-%m-%d %H:%M UTC")
            if generated_at is not None
            else "Unknown",
        },
        {
            "Metric": "Forecast starts",
            "Value": forecast_start.strftime("%Y-%m-%d")
            if forecast_start is not None
            else "Unknown",
        },
        {
            "Metric": "Forecast ends",
            "Value": forecast_end.strftime("%Y-%m-%d") if forecast_end is not None else "Unknown",
        },
        {
            "Metric": "Forecast horizon",
            "Value": (
                f"{(forecast_end.normalize() - forecast_start.normalize()).days + 1} days"
                if forecast_start is not None and forecast_end is not None
                else "Unknown"
            ),
        },
    ]
    st.dataframe(pd.DataFrame(freshness_rows), hide_index=True, use_container_width=True)

with right:
    st.subheader("Model agreement")
    disagreement = float(health.get("max_model_disagreement", 0.0) or 0.0)
    st.metric("Maximum disagreement", f"{disagreement:.1f}/100")
    st.progress(min(max(disagreement / 100, 0.0), 1.0))
    if disagreement < 15:
        st.success("The two anomaly models are closely aligned.")
    elif disagreement < 40:
        st.info("Some model disagreement is present; inspect the Anomaly Lab for context.")
    else:
        st.warning("The models disagree substantially on at least one forecast record.")

st.subheader("Reference-data provenance")
source_quality = health.get("profile_source_quality", {}) or {}
if source_quality:
    provenance = pd.DataFrame(
        [
            {"Source label": str(label), "District rows": int(count)}
            for label, count in source_quality.items()
        ]
    )
    st.dataframe(provenance, hide_index=True, use_container_width=True)
else:
    st.warning("No reference-data source label was found.")

if any("demo" in str(label).lower() or "replace" in str(label).lower() for label in source_quality):
    st.warning(
        "The district profile still contains demo seed inputs. Replace it with documented official "
        "population, age, green/open-land, and impervious-surface data before presenting results as live analysis."
    )
else:
    st.success("Reference-data provenance is recorded and no demo marker was detected.")

st.subheader("Latest scored records")
preview_columns = [
    column
    for column in [
        "date",
        "district",
        "risk_score",
        "risk_level",
        "anomaly_score",
        "model_disagreement",
        "dominant_driver",
        "generated_at_utc",
    ]
    if column in risk.columns
]
preview = risk.sort_values(["date", "risk_score"], ascending=[True, False])[preview_columns].head(
    20
)
st.dataframe(preview, hide_index=True, use_container_width=True)

with st.expander("Raw health payload"):
    st.json(health)

st.caption(
    "Operational thresholds: output warning after 24 hours, critical after 48 hours; "
    "critical if district coverage is incomplete, validation fails, or forecast coverage expires."
)
