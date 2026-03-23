import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES

st.set_page_config(page_title="Cluster Explorer", page_icon="🔍", layout="wide")

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    pca    = pd.read_csv("deployment/pca_sample.csv")
    prof   = pd.read_csv("deployment/cluster_profiles.csv", index_col=0)
    mean_p = pd.read_csv("deployment/mean_profile.csv", index_col=0)
    return pca, prof, mean_p

pca_df, profiles, mean_profile = load_data()

pca_df["Archetype"] = pca_df["macro_cluster"].map(
    lambda g: f"Group {g} — {NAMES[g]}"
)
color_map = {f"Group {g} — {NAMES[g]}": COLORS[g] for g in range(4)}

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🔍 Cluster Explorer")
st.markdown(
    "Explore how 10,000 sampled workloads distribute across the 4 discovered archetypes "
    "in the PCA projection space, then drill into any archetype to see its feature profile."
)

# ── PCA scatter ────────────────────────────────────────────────────────────────
st.subheader("PCA Projection — 10,000 Sampled Workloads")
st.caption(
    "PC1 captures memory scale (33% of variance). "
    "PC2 captures utilisation efficiency (21%). "
    "Toggle groups on/off by clicking the legend."
)

fig = px.scatter(
    pca_df, x="PC1", y="PC2",
    color="Archetype",
    color_discrete_map=color_map,
    opacity=0.35,
    height=520,
    labels={
        "PC1": "PC1 — Memory Scale (33% variance)",
        "PC2": "PC2 — Utilisation Efficiency (21% variance)",
    },
)
fig.update_traces(marker=dict(size=3))
fig.update_layout(
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="left", x=0, font=dict(size=11)
    ),
    margin=dict(t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Archetype drill-down ───────────────────────────────────────────────────────
st.subheader("Archetype Profile")
options = [f"Group {g} — {EMOJIS[g]} {NAMES[g]}" for g in range(4)]
selected = st.selectbox("Select an archetype:", options)
g = int(selected.split(" ")[1])

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Workloads",      f"{int(profiles.loc[g, 'n_workloads']):,}",
          f"{profiles.loc[g, 'pct_workloads']}% of total")
m2.metric("Failure Rate",   f"{profiles.loc[g, 'failure_rate_pct']}%")
m3.metric("Micro-Clusters", int(profiles.loc[g, 'n_micro_clusters']))
m4.metric("Noise Share",    f"{profiles.loc[g, 'noise_pct']}%")

st.markdown(f"**Feature profile for Group {g} — {NAMES[g]}**")
st.caption(
    "Bars show mean standardised feature value (z-score). "
    "Positive = above dataset average. Negative = below. "
    "CPU features are excluded (zero discriminative signal)."
)

# Drop zero-signal CPU features
group_series = mean_profile.loc[f"Group {g}"]
non_zero = group_series[group_series.abs() > 1e-6]

bar_colors = [COLORS[g] if v >= 0 else "#bdc3c7" for v in non_zero.values]

fig2 = go.Figure(go.Bar(
    x=non_zero.values,
    y=non_zero.index,
    orientation="h",
    marker_color=bar_colors,
    text=[f"{v:+.3f}" for v in non_zero.values],
    textposition="outside",
))
fig2.update_layout(
    height=420,
    xaxis_title="Mean Standardised Value (z-score)",
    yaxis_title="",
    title=f"Group {g} — {NAMES[g]}",
    margin=dict(l=180, r=80, t=40, b=30),
    xaxis=dict(zeroline=True, zerolinecolor="#555", zerolinewidth=1),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Side-by-side group comparison ─────────────────────────────────────────────
st.divider()
st.subheader("All Archetypes — Size and Failure Rate")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    fig3 = go.Figure(go.Bar(
        x=[f"G{g}" for g in range(4)],
        y=[profiles.loc[g, "pct_workloads"] for g in range(4)],
        marker_color=[COLORS[g] for g in range(4)],
        text=[f"{profiles.loc[g, 'pct_workloads']}%" for g in range(4)],
        textposition="outside",
    ))
    fig3.update_layout(
        title="Workload Share (%)", yaxis_title="%",
        height=320, margin=dict(t=40, b=20),
        yaxis=dict(range=[0, 55]),
    )
    st.plotly_chart(fig3, use_container_width=True)

with comp_col2:
    fig4 = go.Figure(go.Bar(
        x=[f"G{g}" for g in range(4)],
        y=[profiles.loc[g, "failure_rate_pct"] for g in range(4)],
        marker_color=[COLORS[g] for g in range(4)],
        text=[f"{profiles.loc[g, 'failure_rate_pct']}%" for g in range(4)],
        textposition="outside",
    ))
    fig4.update_layout(
        title="Failure Rate (%)", yaxis_title="%",
        height=320, margin=dict(t=40, b=20),
        yaxis=dict(range=[0, 42]),
    )
    st.plotly_chart(fig4, use_container_width=True)
