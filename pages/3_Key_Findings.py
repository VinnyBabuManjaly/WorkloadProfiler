import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES, RECOMMENDATIONS, SIZE_PCT, FAILURE_RATE, N_MICRO


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    profiles    = pd.read_csv("deployment/cluster_profiles.csv", index_col=0)
    mean_profile = pd.read_csv("deployment/mean_profile.csv", index_col=0)
    return profiles, mean_profile

profiles, mean_profile = load_data()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📊 Key Findings")
st.markdown(
    "Business-level insights from the 4 discovered workload archetypes - "
    "including novel findings that emerged beyond the original project objectives. "
    "All findings are derived from the HDBSCAN + KMeans pipeline applied to "
    "405,894 workload instances from the Google Borg 2019 cluster trace."
)

st.divider()

# ── Failure rate gradient ──────────────────────────────────────────────────────
st.subheader("Failure Rate Gradient - The 10x Spread")
st.caption(
    "The failure rate decreases monotonically from Group 0 (31.4%) to Group 3 (3.3%) - "
    "a nearly 10x range hidden inside a 23% overall average. "
    "This structure is completely invisible in any aggregate metric; it only emerges "
    "when workloads are grouped by memory behaviour via unsupervised clustering."
)

labels = [f"Group {g}\n{NAMES[g]}" for g in range(4)]
fig1 = go.Figure(go.Bar(
    x=labels,
    y=[FAILURE_RATE[g] for g in range(4)],
    marker_color=[COLORS[g] for g in range(4)],
    text=[f"{FAILURE_RATE[g]}%" for g in range(4)],
    textposition="outside",
    width=0.5,
))
fig1.update_layout(
    height=380, yaxis_title="Failure Rate (%)",
    yaxis=dict(range=[0, 42]),
    margin=dict(t=20, b=20),
    showlegend=False,
)
st.plotly_chart(fig1, width='stretch')

st.divider()

# ── Feature heatmap ────────────────────────────────────────────────────────────
st.subheader("Feature Profile Heatmap - What Makes Each Archetype Different")
st.caption(
    "Each cell shows the mean standardised value (z-score) for that feature in that group. "
    "**Green = above the dataset average. Red = below.** Grey means no signal. "
    "Notice the CPU rows are uniformly grey - confirmed zero discriminative signal. "
    "Memory features (top rows) drive all the variation."
)

# Exclude zero-signal CPU columns
non_cpu = mean_profile.loc[:, mean_profile.abs().sum() > 1e-6].columns.tolist()
heat_data = mean_profile[non_cpu]

fig2 = px.imshow(
    heat_data.T,
    color_continuous_scale="RdYlGn",
    aspect="auto",
    labels=dict(x="Macro-Group", y="Feature", color="Mean z-score"),
    x=[f"Group {g}" for g in range(4)],
    text_auto=".2f",
)
fig2.update_layout(height=480, margin=dict(t=20, b=20))
fig2.update_coloraxes(colorbar_title="Mean<br>z-score")
st.plotly_chart(fig2, width='stretch')

st.divider()

# ── Novel findings ─────────────────────────────────────────────────────────────
st.subheader("Novel Findings - Beyond the Original Objectives")

col1, col2 = st.columns(2)
with col1:
    st.error(
        "**🔇 CPU is not a workload differentiator in Google Borg 2019**\n\n"
        "All six CPU-related features show zero discriminative signal across all four archetypes. "
        "Memory is the sole primary resource dimension - a structurally important finding for any "
        "future CPU-based autoscaling or scheduling model on this dataset."
    )
    st.warning(
        "**📦 Over-provisioning does not reduce failures**\n\n"
        "Group 0 is simultaneously the most over-provisioned archetype (below-average memory "
        "utilisation despite long runtimes) and carries the highest failure rate (31.4%). "
        "Resource slack is not preventing failures - structural causes dominate. "
        "More memory will not fix this group."
    )
with col2:
    st.success(
        "**💎 The rarest workloads are the most reliable**\n\n"
        "Group 3 (only 1.9% of workloads) consumes the most memory (+5.8 SD above the dataset mean) "
        "yet has the lowest failure rate (3.3%). The most resource-expensive workloads are also the "
        "most carefully managed - a strong argument for dedicated infrastructure investment in their placement."
    )
    st.info(
        "**📐 Memory scale is the primary axis of workload variation**\n\n"
        "PCA confirms that PC1 (33% of variance) is driven by memory features. "
        "Workload typing in Google Borg is fundamentally a memory-scale problem, "
        "not a CPU or runtime problem. This aligns with Borg's known memory-first "
        "resource scheduling design."
    )

st.divider()

# ── Business recommendations ───────────────────────────────────────────────────
st.subheader("Infrastructure Recommendations - Per Archetype")
st.caption(
    "Concrete scheduling, autoscaling, and SLA actions for each workload type. "
    "Expand any archetype to see its three recommendation areas."
)

for g in range(4):
    with st.expander(
        f"{EMOJIS[g]} Group {g} - {NAMES[g]}  "
        f"({SIZE_PCT[g]}% of workloads · {FAILURE_RATE[g]}% failure rate · {N_MICRO[g]} micro-clusters)"
    ):
        rc1, rc2, rc3 = st.columns(3)
        recs = RECOMMENDATIONS[g]
        for col, (label, text) in zip([rc1, rc2, rc3], recs.items()):
            col.markdown(
                f"<div style='"
                f"background:{COLORS[g]}12;"
                f"border-left:4px solid {COLORS[g]};"
                f"padding:12px;"
                f"border-radius:4px;"
                f"'>"
                f"<b>{label}</b><br><br>"
                f"<span style='font-size:13px;'>{text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ── Model summary ──────────────────────────────────────────────────────────────
st.subheader("Model Comparison - All 6 Algorithms")
st.caption(
    "All models were evaluated on the same 30% random sample (121,535 workloads). "
    "Two baselines set the floor; four density/centroid models competed above it. "
    "HDBSCAN wins on every metric and is the only model with stable results across repeated runs."
)
summary_data = {
    "Model":         ["HDBSCAN mcs=50 ms=10", "DBSCAN eps=0.203 ms=10",
                      "KMeans k=4",           "GMM k=40",
                      "Runtime Quantile",     "Single Cluster"],
    "Clusters":      ["617", "687", "4", "40", "6", "1"],
    "Silhouette":    ["0.874", "0.493", "0.303", "0.142", "0.041", "-"],
    "DB Index":      ["0.488", "0.508", "1.127", "2.070", "93.09", "-"],
    "Role":          ["✅ Primary model", "Secondary (validation)",
                      "Macro labelling", "❌ Eliminated",
                      "Baseline", "Baseline"],
}
st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

st.caption(
    "✅ **HDBSCAN** is the approved model - leads on Silhouette Score (0.874), "
    "Davies-Bouldin Index (0.488), and Calinski-Harabasz Score. "
    "Macro ARI = 0.9975 ± 0.0006 across 5 sub-sample stability runs confirms "
    "production-grade reliability of the 4-archetype grouping. "
    "❌ GMM was eliminated due to unstable cluster assignments across runs."
)
