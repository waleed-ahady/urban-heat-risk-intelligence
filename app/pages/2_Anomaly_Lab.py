from __future__ import annotations

import plotly.express as px
import streamlit as st

from urban_heat_risk.dashboard import load_risk, setup_page, show_data_mode_notice

setup_page("Anomaly Lab", "🔎")
st.title("Anomaly Detection Lab")
risk = load_risk()
show_data_mode_notice(risk)

st.markdown(
    "Isolation Forest captures globally unusual combinations, while Robust Covariance "
    "models the robust multivariate climate envelope and flags large robust-distance departures. The displayed "
    "anomaly score is a weighted percentile ensemble."
)

district = st.selectbox("District", ["All districts", *sorted(risk["district"].unique())])
view = risk if district == "All districts" else risk[risk["district"] == district]

scatter = px.scatter(
    view,
    x="isolation_forest_percentile",
    y="robust_covariance_percentile",
    color="risk_score",
    size="max_apparent_temp",
    hover_name="district",
    hover_data=["date", "anomaly_score", "model_disagreement"],
    labels={
        "isolation_forest_percentile": "Isolation Forest percentile",
        "robust_covariance_percentile": "Robust covariance percentile",
    },
    range_x=[0, 1],
    range_y=[0, 1],
)
st.plotly_chart(scatter, use_container_width=True)

st.subheader("Highest anomaly forecasts")
columns = [
    "date",
    "district",
    "max_apparent_temp",
    "apparent_anomaly_z",
    "isolation_forest_percentile",
    "robust_covariance_percentile",
    "anomaly_score",
    "model_disagreement",
]
st.dataframe(
    view.nlargest(20, "anomaly_score")[columns].round(3),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Model disagreement")
disagreement = view.groupby("date", as_index=False)["model_disagreement"].max()
figure = px.bar(disagreement, x="date", y="model_disagreement")
figure.update_yaxes(title="Maximum percentile-point disagreement")
st.plotly_chart(figure, use_container_width=True)
