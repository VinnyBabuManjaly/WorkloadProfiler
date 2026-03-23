import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES, RECOMMENDATIONS, SIZE_PCT, FAILURE_RATE, N_MICRO

st.set_page_config(page_title="Key Findings", page_icon="📊", layout="wide")

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
    "Business-level insights derived from the 4 discovered workload archetypes — "
    "including novel findings beyond the original project objectives."
)

st.divider()

# ── Failure rate gradient ──────────────────────────────────────────────────────
st.subheader("Failure Rate Gradient Across Archetypes")
st.caption(
    "The failure rate decreases monotonically from Group 0 to Group 3 — "
    "a clear risk tier structure that maps directly to infrastructure priorities."
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
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Feature heatmap ────────────────────────────────────────────────────────────
st.subheader("Feature Profile Heatmap — All Archetypes")
st.caption(
    "Colour shows relative intensity per feature (green = high, red = low). "
    "CPU features are excluded — confirmed zero discriminative signal across all groups."
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
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Novel findings ─────────────────────────────────────────────────────────────
st.subheader("Novel Findings")

col1, col2 = st.columns(2)
with col1:
    st.error(
        "**🔇 CPU is not a workload differentiator in Google Borg 2019**\n\n"
        "All six CPU-related features show zero discriminative signal across all four archetypes. "
        "Memory is the sole primary resource dimension — a structurally important finding for any "
        "future CPU-based autoscaling or scheduling model on this dataset."
    )
    st.warning(
        "**📦 Over-provisioning does not reduce failures**\n\n"
        "Group 0 is simultaneously the most over-provisioned archetype (below-average memory "
        "utilisation despite long runtimes) and carries the highest failure rate (31.4%). "
        "Resource slack is not preventing failures — structural causes dominate. "
        "More memory will not fix this group."
    )
with col2:
    st.success(
        "**💎 The rarest workloads are the most reliable**\n\n"
        "Group 3 (only 1.9% of workloads) consumes the most memory (+5.8 SD above the dataset mean) "
        "yet has the lowest failure rate (3.3%). The most resource-expensive workloads are also the "
        "most carefully managed — a strong argument for dedicated infrastructure investment in their placement."
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
st.subheader("Infrastructure Recommendations per Archetype")

for g in range(4):
    with st.expander(
        f"{EMOJIS[g]} Group {g} — {NAMES[g]}  "
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
st.subheader("Model Summary")
summary_data = {
    "Model":         ["HDBSCAN mcs=50 ms=10", "DBSCAN eps=0.203 ms=10",
                      "KMeans k=4",           "GMM k=40",
                      "Runtime Quantile",     "Single Cluster"],
    "Clusters":      [617, 687, 4, 40, 6, 1],
    "Silhouette":    [0.874, 0.493, 0.303, 0.142, 0.041, "—"],
    "DB Index":      [0.488, 0.508, 1.127, 2.070, 93.09, "—"],
    "Role":          ["✅ Primary model", "Secondary (validation)",
                      "Macro labelling", "❌ Eliminated",
                      "Baseline", "Baseline"],
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.caption(
    "HDBSCAN is the approved model. It leads on Silhouette, DB Index, and CH Score. "
    "Macro-level ARI = 0.9975 confirms production-grade stability of the 4-archetype grouping."
)
