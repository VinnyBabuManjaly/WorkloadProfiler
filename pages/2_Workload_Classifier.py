import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.graph_objects as go
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES, RECOMMENDATIONS, FAILURE_RATE


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler  = joblib.load("deployment/scaler.pkl")
    kmeans  = joblib.load("deployment/kmeans_model.pkl")
    with open("deployment/feature_cols.json") as f:
        feature_cols = json.load(f)
    return scaler, kmeans, feature_cols

@st.cache_data
def load_examples():
    df = pd.read_csv("deployment/sample_data.csv")
    examples = {}
    for g in range(4):
        subset = df[df["macro_cluster"] == g]
        if len(subset) > 0:
            examples[g] = subset.sample(1, random_state=42).iloc[0].to_dict()
    return examples

@st.cache_data
def load_mean_profile():
    return pd.read_csv("deployment/mean_profile.csv", index_col=0)

scaler, kmeans, feature_cols = load_models()
examples     = load_examples()
mean_profile = load_mean_profile()

INPUT_KEYS = [
    "assigned_memory", "req_memory", "avg_memory",
    "max_memory", "page_cache_memory", "runtime_seconds",
]

DEFAULTS = {
    "assigned_memory":   0.010,
    "req_memory":        0.010,
    "avg_memory":        0.005,
    "max_memory":        0.010,
    "page_cache_memory": 0.0,
    "runtime_seconds":   0.500,
}

# Initialise session state
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Feature vector builder ─────────────────────────────────────────────────────
def build_feature_vector(vals: dict, feature_cols: list) -> pd.DataFrame:
    eps = 1e-9
    am, rm, avg_m, max_m, pc, rt = (
        vals["assigned_memory"], vals["req_memory"],
        vals["avg_memory"],      vals["max_memory"],
        vals["page_cache_memory"], vals["runtime_seconds"],
    )
    raw = {
        "runtime_seconds":               rt,
        "assigned_memory":               am,
        "page_cache_memory":             pc,
        "avg_cpu":                       0.0,
        "avg_memory":                    avg_m,
        "max_cpu":                       0.0,
        "max_memory":                    max_m,
        "req_cpu":                       0.0,
        "req_memory":                    rm,
        "memory_overprovisioning_ratio": am / (rm + eps),
        "avg_cpu_utilization":           0.0,
        "peak_cpu_utilization":          0.0,
        "memory_utilization_avg":        avg_m / (am + eps),
        "memory_utilization_peak":       max_m / (am + eps),
        "cpu_peak_to_avg_ratio":         0.0,
        "runtime_efficiency":            rt / (am + eps),
        "page_cache_ratio":              pc / (am + eps),
    }
    return pd.DataFrame([[raw[c] for c in feature_cols]], columns=feature_cols)

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🎯 Workload Classifier")
st.markdown(
    "Enter the resource characteristics of any workload - how much memory was allocated, "
    "how much was actually used, and how long it ran. The classifier builds a 17-feature "
    "vector, applies `StandardScaler` normalisation, then uses the trained **KMeans model** "
    "to identify which of the 4 discovered archetypes it most closely matches.\n\n"
    "The result includes the archetype's **failure rate**, its infrastructure meaning, "
    "and concrete **scheduling, autoscaling, and SLA recommendations** - plus a chart "
    "showing how your workload compares to the archetype average."
)
st.info(
    "**Why are CPU inputs missing?** All six CPU features (`avg_cpu`, `max_cpu`, `req_cpu`, "
    "`avg_cpu_utilization`, `peak_cpu_utilization`, `cpu_peak_to_avg_ratio`) showed a "
    "mean z-score of 0.000 across all four archetypes in the full dataset analysis. "
    "They carry **zero discriminative signal** - memory and runtime drive all classification "
    "in the Google Borg 2019 trace.",
    icon="ℹ️",
)

# ── Preset examples ────────────────────────────────────────────────────────────
st.subheader("Load a Preset Example")
st.caption(
    "Pre-fills the sliders with a real workload sampled directly from each archetype in the dataset. "
    "A good starting point before adjusting values manually."
)
preset_cols = st.columns(4)
for g, col in enumerate(preset_cols):
    if col.button(
        f"{EMOJIS[g]} Group {g}\n{NAMES[g]}",
        width='stretch',
        key=f"btn_{g}",
    ):
        row = examples.get(g, {})
        SLIDER_BOUNDS = {
            "assigned_memory":   (0.0001, 0.99),
            "req_memory":        (0.0001, 0.99),
            "avg_memory":        (0.0001, 0.99),
            "max_memory":        (0.0001, 0.99),
            "page_cache_memory": (0.0,    0.50),
            "runtime_seconds":   (0.0001, 1.0),
        }
        for k in INPUT_KEYS:
            if k in row and pd.notna(row[k]):
                lo, hi = SLIDER_BOUNDS[k]
                st.session_state[k] = float(np.clip(float(row[k]), lo, hi))
        st.rerun()

st.divider()

# ── Input sliders ──────────────────────────────────────────────────────────────
st.subheader("Workload Resource Inputs")
st.caption(
    "All values are in Borg-normalised units (0–1 scale relative to cluster capacity). "
    "Adjust the sliders to describe your workload's memory footprint and runtime."
)
left, right = st.columns(2)

with left:
    st.slider(
        "Assigned Memory - `assigned_memory`",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="assigned_memory",
        help=(
            "Memory allocated (provisioned) to this workload by the scheduler. "
            "In Borg-normalised units: 1.0 = entire cluster memory. "
            "Compare to avg_memory to detect over-provisioning."
        ),
    )
    st.slider(
        "Requested Memory - `req_memory`",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="req_memory",
        help=(
            "Memory the workload declared it needs at submission time. "
            "High assigned_memory relative to req_memory indicates the scheduler "
            "padded the allocation - a key over-provisioning signal."
        ),
    )
    st.slider(
        "Average Memory Used - `avg_memory`",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="avg_memory",
        help=(
            "Average memory actually consumed across the workload's lifetime. "
            "The gap between assigned_memory and avg_memory is the idle/wasted capacity. "
            "Used to compute memory_utilization_avg."
        ),
    )

with right:
    st.slider(
        "Peak Memory Used - `max_memory`",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="max_memory",
        help=(
            "Highest memory usage recorded at any single point during execution. "
            "Used to compute memory_utilization_peak. "
            "A large gap between max_memory and avg_memory indicates bursty memory behaviour."
        ),
    )
    st.slider(
        "Page Cache Memory - `page_cache_memory`",
        min_value=0.0, max_value=0.50, step=0.0001, format="%.4f",
        key="page_cache_memory",
        help=(
            "Memory used for OS page cache - a proxy for I/O intensity. "
            "High values indicate the workload reads heavily from disk. "
            "Used to compute page_cache_ratio = page_cache_memory / assigned_memory."
        ),
    )
    st.slider(
        "Runtime Duration - `runtime_seconds`",
        min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f",
        key="runtime_seconds",
        help=(
            "How long the workload ran, in Borg-normalised time units (0–1). "
            "Used to compute runtime_efficiency = runtime_seconds / assigned_memory. "
            "High runtime + low memory usage = long-running over-provisioned pattern (Group 0)."
        ),
    )

# Derived features preview
with st.expander("🔧 Derived features - computed automatically from the inputs above"):
    st.caption(
        "These 5 ratios are computed from your slider inputs and added to the 6 raw features "
        "to form the full 17-feature vector used for classification. "
        "CPU features are fixed at 0.0 (zero discriminative signal in this dataset)."
    )
    am  = st.session_state["assigned_memory"]
    rm  = st.session_state["req_memory"]
    avm = st.session_state["avg_memory"]
    mxm = st.session_state["max_memory"]
    pc  = st.session_state["page_cache_memory"]
    rt  = st.session_state["runtime_seconds"]
    eps = 1e-9
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Overprovisioning Ratio",  f"{am/(rm+eps):.3f}",
              help="assigned_memory / req_memory - how much more was given than requested.")
    d2.metric("Avg Memory Utilisation",  f"{avm/(am+eps):.3f}",
              help="avg_memory / assigned_memory - fraction of allocation actually used on average.")
    d3.metric("Peak Memory Utilisation", f"{mxm/(am+eps):.3f}",
              help="max_memory / assigned_memory - peak fraction of allocation used at any point.")
    d4.metric("Runtime Efficiency",      f"{rt/(am+eps):.3f}",
              help="runtime_seconds / assigned_memory - duration relative to memory footprint.")
    d5.metric("Page Cache Ratio",        f"{pc/(am+eps):.3f}",
              help="page_cache_memory / assigned_memory - I/O intensity proxy.")

st.divider()

# ── Classify ───────────────────────────────────────────────────────────────────
if st.button("🔍  Classify Workload", type="primary", width='stretch'):
    vals = {k: st.session_state[k] for k in INPUT_KEYS}
    fv     = build_feature_vector(vals, feature_cols)
    scaled = scaler.transform(fv)
    g      = int(kmeans.predict(scaled)[0])

    st.success(f"**Classified as: {EMOJIS[g]} Group {g} - {NAMES[g]}** &nbsp;|&nbsp; Failure rate: {FAILURE_RATE[g]}%")

    # Recommendation boxes
    recs = RECOMMENDATIONS[g]
    rc1, rc2, rc3 = st.columns(3)
    for col, (label, text) in zip([rc1, rc2, rc3], recs.items()):
        col.markdown(
            f"<div style='"
            f"background:{COLORS[g]}14;"
            f"border-left:4px solid {COLORS[g]};"
            f"padding:14px;"
            f"border-radius:5px;"
            f"min-height:130px;"
            f"'>"
            f"<b>{label}</b><br><br>{text}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### How this workload compares to the archetype average")
    st.caption(
        "Each pair of bars shows one feature. **Coloured = your workload's standardised value. "
        "Grey = the Group average.** Features further from zero (in either direction) are what "
        "most strongly define this archetype. CPU features are excluded - zero signal."
    )

    non_cpu = [c for c in feature_cols if "cpu" not in c.lower()]
    idx     = [feature_cols.index(c) for c in non_cpu]

    this_scaled      = scaled[0][idx]
    archetype_scaled = mean_profile.loc[f"Group {g}", non_cpu].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="This Workload",
        x=non_cpu, y=this_scaled,
        marker_color=COLORS[g], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name=f"Group {g} Average",
        x=non_cpu, y=archetype_scaled,
        marker_color="#95a5a6", opacity=0.75,
    ))
    fig.update_layout(
        barmode="group", height=360,
        xaxis_tickangle=-35,
        yaxis_title="Standardised Value",
        margin=dict(t=20, b=80),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, width='stretch')
