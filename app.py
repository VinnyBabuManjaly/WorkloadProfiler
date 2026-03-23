import streamlit as st
from utils import COLORS, EMOJIS, NAMES, DESCRIPTIONS, SIZE_PCT, FAILURE_RATE, N_MICRO

st.set_page_config(
    page_title="Workload Profiler",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("⚙️ Workload Profiler")
st.subheader("Unsupervised Workload Type Discovery in Google Borg Cluster Traces")

st.markdown(
    "Analysing **121,535 workload instances** from the Google Borg 2019 cluster trace, "
    "this project discovers four distinct workload archetypes using HDBSCAN density-based "
    "clustering — validated for production-level stability and mapped to actionable "
    "infrastructure recommendations."
)

st.divider()

# ── Model performance ──────────────────────────────────────────────────────────
st.subheader("Model Performance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Silhouette Score",    "0.874",   "Excellent  (target > 0.6)")
c2.metric("Davies-Bouldin Index","0.488",   "Excellent  (target < 1.0)")
c3.metric("HDBSCAN Clusters",   "617",     "Micro-level patterns")
c4.metric("Macro-Level ARI",    "0.9975",  "Production stable")
c5.metric("Noise Fraction",     "12.4%",   "Within 20% threshold")

st.divider()

# ── Archetype cards ────────────────────────────────────────────────────────────
st.subheader("Discovered Workload Archetypes")
cols = st.columns(4)
for g, col in enumerate(cols):
    with col:
        st.markdown(
            f"""
            <div style="
                background:{COLORS[g]}18;
                border-left:5px solid {COLORS[g]};
                padding:18px 16px;
                border-radius:6px;
                min-height:220px;
            ">
                <h4 style="color:{COLORS[g]};margin-top:0;">
                    {EMOJIS[g]} Group {g}
                </h4>
                <p style="font-weight:700;font-size:14px;margin-bottom:6px;">
                    {NAMES[g]}
                </p>
                <p style="font-size:13px;color:#555;margin-bottom:10px;">
                    {DESCRIPTIONS[g]}
                </p>
                <hr style="margin:8px 0;border-color:{COLORS[g]}44;">
                <p style="font-size:12px;margin:3px 0;">
                    📊 &nbsp;<b>{SIZE_PCT[g]}%</b> of workloads
                </p>
                <p style="font-size:12px;margin:3px 0;">
                    ❌ &nbsp;<b>{FAILURE_RATE[g]}%</b> failure rate
                </p>
                <p style="font-size:12px;margin:3px 0;">
                    🔍 &nbsp;<b>{N_MICRO[g]}</b> micro-clusters
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Navigation hint ────────────────────────────────────────────────────────────
st.markdown(
    "**Use the sidebar to navigate:**\n"
    "- 🔍 **Cluster Explorer** — visualise the PCA projection and drill into each archetype\n"
    "- 🎯 **Workload Classifier** — input workload features and receive archetype classification "
    "with infrastructure recommendations\n"
    "- 📊 **Key Findings** — business insights, novel findings, and per-archetype recommendations"
)

st.divider()
st.caption(
    "Dataset: Google Borg 2019 Cluster Trace · "
    "Model: HDBSCAN (mcs=50, ms=10) + KMeans k=4 macro-grouping · "
    "Macro ARI: 0.9975"
)
