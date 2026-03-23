import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES, DESCRIPTIONS, SIZE_PCT, FAILURE_RATE, N_MICRO

st.title("⚙️ Workload Profiler")
st.markdown(
    "An unsupervised clustering analysis of **405,894 workload instances** from the "
    "Google Borg 2019 cluster trace — profiling distinct resource-usage archetypes to "
    "inform scheduling, autoscaling, and infrastructure allocation strategies."
)

st.divider()

# ── Model headline metrics ──────────────────────────────────────────────────
st.subheader("Approved Model — HDBSCAN (mcs=50, ms=10)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Silhouette Score",  "0.874",  help="Target > 0.6 — Excellent")
m2.metric("Davies-Bouldin",    "0.488",  help="Target < 1.0 — Excellent")
m3.metric("Micro-Clusters",    "617",    help="Natural density-based patterns")
m4.metric("Macro ARI",         "0.9975", help="Stability across 5 × 80% subsets")
m5.metric("Noise Share",       "12.6%",  help="Genuine outliers, not mislabels")

st.divider()

# ── Archetype cards ─────────────────────────────────────────────────────────
st.subheader("4 Discovered Workload Archetypes")
st.caption("617 micro-clusters collapsed into 4 macro-level archetypes via KMeans k=4 nearest-centroid assignment.")

cols = st.columns(4)
for g, col in enumerate(cols):
    col.markdown(
        f"<div style='"
        f"background:{COLORS[g]}12;"
        f"border-left:5px solid {COLORS[g]};"
        f"padding:16px;"
        f"border-radius:6px;"
        f"height:220px;"
        f"'>"
        f"<b style='font-size:15px;'>{EMOJIS[g]} Group {g}</b><br>"
        f"<span style='font-size:13px;color:{COLORS[g]};'><b>{NAMES[g]}</b></span><br><br>"
        f"<span style='font-size:12px;'>{DESCRIPTIONS[g]}</span><br><br>"
        f"<span style='font-size:12px;'>"
        f"<b>{SIZE_PCT[g]}%</b> of workloads &nbsp;·&nbsp; "
        f"<b>{FAILURE_RATE[g]}%</b> failure rate<br>"
        f"<b>{N_MICRO[g]}</b> micro-clusters"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Key findings summary ────────────────────────────────────────────────────
st.subheader("Key Findings")
kf1, kf2 = st.columns(2)
with kf1:
    st.error(
        "**CPU carries zero discriminative signal.** All six CPU features show 0.000 mean "
        "scaled value across all four archetypes. Memory is the sole primary resource dimension."
    )
    st.warning(
        "**Over-provisioning does not reduce failures.** Group 0 is the most over-provisioned "
        "archetype yet has the highest failure rate (31.4%). Structural causes dominate."
    )
with kf2:
    st.success(
        "**The rarest workloads are the most reliable.** Group 3 (1.9% of workloads) consumes "
        "the most memory yet has the lowest failure rate (3.3%)."
    )
    st.info(
        "**Memory scale is the primary axis of variation.** PC1 (33% of variance) is driven "
        "entirely by memory features — workload typing in Borg is a memory-scale problem."
    )

st.divider()

# ── Navigation guide ────────────────────────────────────────────────────────
st.subheader("Navigation")
nav1, nav2, nav3 = st.columns(3)
nav1.markdown(
    "**🔍 Cluster Explorer**\n\n"
    "Explore the PCA projection of 10,000 sampled workloads. "
    "Drill into any archetype to see its full feature profile."
)
nav2.markdown(
    "**🎯 Workload Classifier**\n\n"
    "Input a workload's resource characteristics and classify it "
    "into one of the 4 archetypes with infrastructure recommendations."
)
nav3.markdown(
    "**📊 Key Findings**\n\n"
    "Business-level insights, novel findings beyond project objectives, "
    "and the full model comparison summary."
)
