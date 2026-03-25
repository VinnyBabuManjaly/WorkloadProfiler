import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES, DESCRIPTIONS, SIZE_PCT, FAILURE_RATE, N_MICRO

st.title("⚙️ Workload Profiler")
st.caption("Unsupervised workload archetype discovery on Google Borg 2019 cluster trace data")

st.markdown(
    "Large compute clusters - whether Google Borg, AWS, Azure, or an on-premises Kubernetes fleet - "
    "run hundreds of thousands of tasks simultaneously, all competing for the same memory and CPU. "
    "The problem: infrastructure systems treat every task identically, applying the same scheduling "
    "rules and resource allocations regardless of how the task actually behaves. "
    "The result is memory wasted on idle tasks, failures nobody predicted, and no way to intervene "
    "because nobody knows which *type* of workload they are dealing with.\n\n"
    "This project applied **HDBSCAN** (unsupervised density-based clustering) to **405,894 real "
    "workload instances** from Google's 2019 Borg trace - automatically discovering **4 operationally "
    "distinct archetypes** from raw resource usage patterns, with no manual labelling. "
    "Each archetype carries a different failure risk, memory profile, and set of infrastructure actions."
)

st.divider()

# ── Model headline metrics ──────────────────────────────────────────────────
st.subheader("Model Performance - HDBSCAN `mcs=50, ms=10` → KMeans `k=4`")
st.caption(
    "HDBSCAN finds 617 fine-grained density-based micro-clusters. "
    "KMeans nearest-centroid assignment then consolidates these into 4 macro-level archetypes for business use."
)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Silhouette Score", "0.874", help=(
    "Measures how well each workload fits its own cluster versus its nearest neighbour cluster. "
    "Range: -1 (worst) to +1 (perfect). Above 0.6 is considered excellent. "
    "0.874 means the 4 archetypes are tightly cohesive and clearly separated from each other."
))
m2.metric("Davies-Bouldin", "0.488", help=(
    "Measures the average overlap between each cluster and its closest neighbour. "
    "Lower is better - below 1.0 is excellent. "
    "0.488 confirms that the groups are compact internally and well-separated from each other."
))
m3.metric("Micro-Clusters", "617", help=(
    "Number of fine-grained natural groupings found by HDBSCAN before macro consolidation. "
    "Each micro-cluster represents a distinct, densely-packed workload behaviour pattern. "
    "These 617 patterns are collapsed into 4 interpretable macro archetypes via KMeans."
))
m4.metric("Macro ARI", "0.9975", help=(
    "Adjusted Rand Index - measures how consistently the same 4 archetypes emerge "
    "when the full pipeline is re-run on different random 80% subsets of the data. "
    "Range: 0 (random) to 1 (perfect). 0.9975 ± 0.0006 across 5 runs confirms production-grade stability."
))
m5.metric("Noise Share", "12.6%", help=(
    "Percentage of workloads HDBSCAN flagged as noise - genuine outliers too dissimilar "
    "to belong to any dense cluster. These are not mislabels; they represent atypical workloads. "
    "Noise points are assigned to the nearest macro archetype for recommendations "
    "but are excluded from cluster quality scoring."
))

st.divider()

# ── Archetype cards ─────────────────────────────────────────────────────────
st.subheader("4 Discovered Workload Archetypes")
st.caption(
    "617 HDBSCAN micro-clusters consolidated into 4 macro archetypes via KMeans nearest-centroid assignment. "
    "Each archetype has a distinct memory behaviour, failure profile, and set of infrastructure implications."
)

cols = st.columns(4)
for g, col in enumerate(cols):
    col.markdown(
        f"<div style='"
        f"background:{COLORS[g]}12;"
        f"border-left:5px solid {COLORS[g]};"
        f"padding:16px;"
        f"border-radius:6px;"
        f"height:240px;"
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
st.caption("Novel findings that emerged beyond the original project objectives.")
kf1, kf2 = st.columns(2)
with kf1:
    st.error(
        "**🔇 CPU carries zero discriminative signal**\n\n"
        "All six CPU features (`avg_cpu`, `max_cpu`, `req_cpu`, and the three derived CPU ratios) "
        "show a mean z-score of **0.000** across all four archetypes - confirmed by both PCA and the "
        "feature heatmap. **Memory is the sole primary resource dimension** in this dataset. "
        "Any CPU-based autoscaling model built on Borg 2019 would have nothing to work with."
    )
    st.warning(
        "**📦 Over-provisioning does not reduce failures - structural causes dominate**\n\n"
        "Group 0 has below-average `memory_utilization_avg` despite above-average runtime - "
        "the classic over-provisioning pattern. Yet it carries the **highest failure rate (31.4%)**. "
        "Resource slack is not the problem. Memory reclamation, not more provisioning, is the fix."
    )
with kf2:
    st.success(
        "**💎 The rarest workloads are the most reliable**\n\n"
        "Group 3 makes up just **1.9%** of all workloads yet consumes the most memory "
        "(+5.8 SD above the dataset mean) and fails least often (**3.3% failure rate**). "
        "This strongly suggests dedicated node pool placement - an approach that is clearly "
        "working and should be formalised."
    )
    st.info(
        "**📐 Memory scale is the primary axis of workload variation**\n\n"
        "PCA confirms **PC1 (33% of variance)** is driven entirely by memory features. "
        "PC2 (21%) captures utilisation efficiency. Together they explain 54% of all variation. "
        "Workload typing in Borg is fundamentally a **memory-scale problem**, not CPU or runtime."
    )

st.divider()

# ── Navigation guide ────────────────────────────────────────────────────────
st.subheader("Explore the App")
nav1, nav2, nav3 = st.columns(3)
nav1.markdown(
    "**🔍 Cluster Explorer**\n\n"
    "Visualise how 10,000 sampled workloads distribute across the 4 archetypes "
    "in a 2D PCA projection (PC1 = memory scale, PC2 = utilisation efficiency). "
    "Drill into any archetype to see its full standardised feature profile."
)
nav2.markdown(
    "**🎯 Workload Classifier**\n\n"
    "Enter a workload's resource characteristics - memory allocation, usage, "
    "page cache, and runtime - and the trained KMeans model classifies it into "
    "one of the 4 archetypes with concrete scheduling, autoscaling, and SLA recommendations."
)
nav3.markdown(
    "**📊 Key Findings**\n\n"
    "Deep-dive into the failure rate gradient, feature heatmap, novel findings "
    "beyond project scope, per-archetype infrastructure recommendations, "
    "and the full 6-model comparison summary with metrics."
)
