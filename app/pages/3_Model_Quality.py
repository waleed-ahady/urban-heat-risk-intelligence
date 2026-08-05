from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from urban_heat_risk.dashboard import METRICS_FILE, load_json, setup_page

setup_page("Model Quality", "🧪")
st.title("Model Quality and Limitations")
metrics = load_json(METRICS_FILE)
evaluation = metrics["evaluation"]

st.warning(metrics["proxy_label_warning"])
st.markdown(
    "The models are evaluated on a strict final-year holdout. Because no health-impact labels "
    "are bundled, the benchmark asks whether the models recover days exceeding district-and-"
    "month historical 98th-percentile thresholds."
)

summary = st.columns(4)
summary[0].metric("Evaluation rows", f"{evaluation['rows']:,}")
summary[1].metric("Proxy-positive rate", f"{100 * evaluation['positive_rate']:.1f}%")
summary[2].metric("Model correlation", f"{evaluation['model_score_correlation']:.2f}")
summary[3].metric("Training rows", f"{metrics['training_rows_total']:,}")

rows = []
for name in ["isolation_forest", "robust_covariance", "ensemble"]:
    values = evaluation[name]
    rows.append(
        {
            "model": name.replace("_", " ").title(),
            "ROC AUC": values["roc_auc"],
            "Average precision": values["average_precision"],
            "Precision at top 3%": values["precision_at_top_3pct"],
        }
    )
quality = pd.DataFrame(rows)
st.dataframe(quality.round(3), hide_index=True, use_container_width=True)

plot = quality.melt(id_vars="model", var_name="metric", value_name="score")
figure = px.bar(plot, x="model", y="score", color="metric", barmode="group")
figure.update_yaxes(range=[0, 1])
st.plotly_chart(figure, use_container_width=True)

st.subheader("Interpretation")
st.markdown(
    """
- **Isolation Forest** is the stronger detector in the bundled demo benchmark.
- **Robust Covariance** provides a parametric, robust-distance view that complements the tree-based detector.
- The **ensemble** is designed for robustness, not guaranteed superiority on every proxy metric.
- A real deployment should replace the proxy benchmark with outcomes such as ambulance calls, hospital admissions, occupational incidents, or load stress where legally and ethically available.
    """
)
