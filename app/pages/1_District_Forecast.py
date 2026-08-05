from __future__ import annotations

import plotly.express as px
import streamlit as st

from urban_heat_risk.dashboard import load_profiles, load_risk, setup_page, show_data_mode_notice
from urban_heat_risk.risk.engine import calculate_risk

setup_page("District Forecast", "📍")
st.title("District Forecast and Scenario Lab")
risk = load_risk()
profiles = load_profiles()
show_data_mode_notice(risk)

district = st.selectbox("District", sorted(risk["district"].unique()))
district_data = risk[risk["district"] == district].sort_values("date")

metrics = st.columns(4)
peak_row = district_data.loc[district_data["risk_score"].idxmax()]
metrics[0].metric("Peak risk", f"{peak_row['risk_score']:.1f}/100")
metrics[1].metric("Peak level", peak_row["risk_level"])
metrics[2].metric("Peak apparent temperature", f"{peak_row['max_apparent_temp']:.1f} °C")
metrics[3].metric("Peak anomaly", f"{peak_row['anomaly_score']:.1f}/100")

chart_data = district_data.melt(
    id_vars=["date"],
    value_vars=["risk_score", "hazard_score", "anomaly_score"],
    var_name="metric",
    value_name="score",
)
figure = px.line(chart_data, x="date", y="score", color="metric", markers=True)
figure.update_yaxes(range=[0, 100])
st.plotly_chart(figure, use_container_width=True)

st.subheader("Risk decomposition")
contributions = district_data[
    ["date", "hazard_contribution", "exposure_contribution", "vulnerability_contribution"]
].melt(id_vars="date", var_name="component", value_name="points")
bar = px.bar(contributions, x="date", y="points", color="component", barmode="stack")
st.plotly_chart(bar, use_container_width=True)

st.subheader("Scenario simulator")
st.caption(
    "Change one forecast day to test sensitivity. The climate anomaly score is held constant; "
    "this is a planning scenario, not a new weather forecast."
)
selected_date = st.selectbox(
    "Scenario date", district_data["date"].dt.date.tolist(), index=min(3, len(district_data) - 1)
)
base = district_data[district_data["date"].dt.date == selected_date].iloc[[0]].copy()
c1, c2, c3, c4 = st.columns(4)
temp_delta = c1.slider("Apparent temperature change (°C)", -5.0, 8.0, 0.0, 0.5)
night_delta = c2.slider("Night temperature change (°C)", -4.0, 6.0, 0.0, 0.5)
green_delta = c3.slider("Green-space change (points)", -15.0, 20.0, 0.0, 1.0)
density_delta = c4.slider("Population density change (%)", -20, 30, 0, 5)

scenario = base.copy()
scenario["max_apparent_temp"] += temp_delta
scenario["night_min_temperature"] += night_delta
scenario["heat_hours_apparent_35"] = (
    scenario["heat_hours_apparent_35"] + max(0, temp_delta) * 1.5
).clip(0, 24)
scenario_profiles = profiles.copy()
selected_mask = scenario_profiles["district"] == district
scenario_profiles.loc[selected_mask, "green_space_pct"] = (
    scenario_profiles.loc[selected_mask, "green_space_pct"] + green_delta
).clip(0, 100)
scenario_profiles.loc[selected_mask, "population_density_per_km2"] *= 1 + density_delta / 100

profile_columns_in_scenario = [
    column for column in profiles.columns if column in scenario.columns and column != "district"
]
result = calculate_risk(
    scenario.drop(columns=profile_columns_in_scenario),
    scenario_profiles,
)
result_row = result.iloc[0]
change = result_row["risk_score"] - base.iloc[0]["risk_score"]

m1, m2, m3 = st.columns(3)
m1.metric("Scenario risk", f"{result_row['risk_score']:.1f}/100", f"{change:+.1f}")
m2.metric("Scenario level", result_row["risk_level"])
m3.metric("Dominant driver", result_row["dominant_driver"])
st.info(result_row["recommendation"])

st.subheader("Forecast detail")
detail_columns = [
    "date",
    "risk_score",
    "risk_level",
    "max_apparent_temp",
    "night_min_temperature",
    "heatwave_streak",
    "anomaly_score",
    "model_disagreement",
    "dominant_driver",
]
st.dataframe(district_data[detail_columns], hide_index=True, use_container_width=True)
