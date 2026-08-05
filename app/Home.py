from __future__ import annotations

import plotly.express as px
import streamlit as st

from urban_heat_risk.dashboard import load_risk, setup_page, show_data_mode_notice

setup_page("Urban Heat Risk Intelligence")

st.title("Urban Heat Risk Intelligence")
st.caption(
    "A decision-support dashboard combining forecast heat hazard, climate anomaly, "
    "population exposure, and district vulnerability."
)

try:
    risk = load_risk()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

show_data_mode_notice(risk)
available_dates = sorted(risk["date"].dt.date.unique())
selected_date = st.select_slider("Forecast date", options=available_dates, value=available_dates[0])
day = risk[risk["date"].dt.date == selected_date].copy()

highest = day.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Highest district risk", f"{highest['risk_score']:.1f}/100")
col2.metric("Highest-risk district", highest["district"])
col3.metric("High or above", int(day["risk_level"].isin(["High", "Very High", "Extreme"]).sum()))
col4.metric("Peak apparent temperature", f"{day['max_apparent_temp'].max():.1f} °C")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("District risk map")
    map_figure = px.scatter_map(
        day,
        lat="latitude",
        lon="longitude",
        size="risk_score",
        color="risk_score",
        hover_name="district",
        hover_data={
            "risk_level": True,
            "risk_score": ":.1f",
            "max_apparent_temp": ":.1f",
            "anomaly_score": ":.1f",
            "latitude": False,
            "longitude": False,
        },
        size_max=32,
        zoom=8.5,
        height=520,
        map_style="open-street-map",
        color_continuous_scale="YlOrRd",
        range_color=(0, 100),
    )
    map_figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(map_figure, use_container_width=True)

with right:
    st.subheader("Priority ranking")
    ranking = day[
        ["district", "risk_score", "risk_level", "dominant_driver", "recommendation"]
    ].copy()
    ranking["risk_score"] = ranking["risk_score"].round(1)
    st.dataframe(ranking, hide_index=True, use_container_width=True, height=520)

st.subheader("Seven-day risk outlook")
outlook = risk.groupby("date", as_index=False).agg(
    maximum_risk=("risk_score", "max"),
    median_risk=("risk_score", "median"),
    maximum_apparent_temperature=("max_apparent_temp", "max"),
)
figure = px.line(
    outlook,
    x="date",
    y=["maximum_risk", "median_risk"],
    markers=True,
    labels={"value": "Risk score", "variable": "Series", "date": "Date"},
)
figure.update_yaxes(range=[0, 100])
st.plotly_chart(figure, use_container_width=True)

st.markdown(
    "<p class='small-note'>Risk is a prioritization index, not a medical diagnosis or an "
    "official emergency warning.</p>",
    unsafe_allow_html=True,
)
