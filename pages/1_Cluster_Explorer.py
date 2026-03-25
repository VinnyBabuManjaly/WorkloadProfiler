import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    pca    = pd.read_csv("deployment/pca_sample.csv")
    prof   = pd.read_csv("deployment/cluster_profiles.csv", index_col=0)
    mean_p = pd.read_csv("deployment/mean_profile.csv", index_col=0)
    return pca, prof, mean_p

pca_df, profiles, mean_profile = load_data()

pca_df["Archetype"] = pca_df["macro_cluster"].map(
    lambda g: f"Group {g} - {NAMES[g]}"
)
color_map = {f"Group {g} - {NAMES[g]}": COLORS[g] for g in range(4)}

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🔍 Cluster Explorer")
st.markdown(
    "This page lets you **see** the 4 discovered workload archetypes as a visual map - "
    "and then drill into any single archetype to understand exactly how its memory usage, "
    "runtime, and efficiency metrics compare to the dataset average.\n\n"
    "The scatter plot uses **PCA (Principal Component Analysis)** to compress 17 features "
    "down to 2 dimensions for visualisation. The actual clustering was performed on all 17 "
    "features - PCA is used here purely for diagnosis and storytelling, not as a model input."
)

# ── PCA scatter ────────────────────────────────────────────────────────────────
st.subheader("PCA Projection - 10,000 Sampled Workloads")
st.caption(
    "Each dot is a workload. **PC1 (x-axis) = memory scale** - tasks further right use more memory. "
    "**PC2 (y-axis) = utilisation efficiency** - tasks higher up use their allocation more efficiently. "
    "PC1 explains 33% of total variance; PC2 explains 21%. "
    "Toggle archetypes on/off by clicking the legend."
)

fig = px.scatter(
    pca_df, x="PC1", y="PC2",
    color="Archetype",
    color_discrete_map=color_map,
    opacity=0.35,
    height=520,
    labels={
        "PC1": "PC1 - Memory Scale (33% variance)",
        "PC2": "PC2 - Utilisation Efficiency (21% variance)",
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
st.plotly_chart(fig, width='stretch')

st.divider()

# ── Archetype drill-down ───────────────────────────────────────────────────────
st.subheader("Archetype Deep-Dive")
st.caption(
    "Select any archetype to see its full feature profile - how it compares to the dataset average "
    "across all non-CPU features. Each bar is a mean z-score: positive means above average, "
    "negative means below. CPU features are excluded (zero signal confirmed across all groups)."
)
options = [f"Group {g} - {EMOJIS[g]} {NAMES[g]}" for g in range(4)]
selected = st.selectbox("Select an archetype to explore:", options)
g = int(selected.split(" ")[1])

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Workloads",      f"{int(profiles.loc[g, 'n_workloads']):,}",
          f"{profiles.loc[g, 'pct_workloads']}% of total")
m2.metric("Failure Rate",   f"{profiles.loc[g, 'failure_rate_pct']}%")
m3.metric("Micro-Clusters", int(profiles.loc[g, 'n_micro_clusters']),
          help="Number of HDBSCAN micro-clusters that map to this macro archetype.")
m4.metric("Noise Share",    f"{profiles.loc[g, 'noise_pct']}%",
          help="% of workloads in this archetype that were flagged as HDBSCAN noise points.")

st.markdown(f"**Standardised feature profile - Group {g}: {NAMES[g]}**")
st.caption(
    "Mean z-score per feature (standardised against the full dataset). "
    "Bars above zero = this group is above average on that feature. "
    "Bars below zero = below average. CPU features are omitted - confirmed zero signal."
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
    title=f"Group {g} - {NAMES[g]}",
    margin=dict(l=180, r=80, t=40, b=30),
    xaxis=dict(zeroline=True, zerolinecolor="#555", zerolinewidth=1),
)
st.plotly_chart(fig2, width='stretch')

# ── Side-by-side group comparison ─────────────────────────────────────────────
st.divider()
st.subheader("All Archetypes - Size and Failure Rate")
st.caption(
    "How large is each group, and how often do tasks in it fail? "
    "The gap between Group 0 (31.4%) and Group 3 (3.3%) is nearly 10x - "
    "invisible in any aggregate metric, only surfaced by the clustering."
)
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
    st.plotly_chart(fig3, width='stretch')

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
    st.plotly_chart(fig4, width='stretch')
