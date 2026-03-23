import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.graph_objects as go
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import COLORS, EMOJIS, NAMES, RECOMMENDATIONS, FAILURE_RATE

st.set_page_config(page_title="Workload Classifier", page_icon="🎯", layout="wide")

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
    "page_cache_memory": 0.001,
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
    "Describe a workload using its resource characteristics. "
    "The classifier scales the inputs, applies the trained KMeans model, "
    "and identifies which of the 4 archetypes best matches — along with "
    "concrete infrastructure recommendations."
)
st.info(
    "**Note on CPU features:** All CPU-related features (`avg_cpu`, `max_cpu`, `req_cpu`, "
    "utilisation ratios) are set to zero automatically. Analysis confirmed these carry no "
    "discriminative signal in the Google Borg 2019 trace — memory and runtime drive all classification.",
    icon="ℹ️",
)

# ── Preset examples ────────────────────────────────────────────────────────────
st.subheader("Load a preset example")
st.caption("Pre-fills the sliders with a real workload sampled from each archetype.")
preset_cols = st.columns(4)
for g, col in enumerate(preset_cols):
    if col.button(
        f"{EMOJIS[g]} Group {g}\n{NAMES[g]}",
        use_container_width=True,
        key=f"btn_{g}",
    ):
        row = examples.get(g, {})
        for k in INPUT_KEYS:
            if k in row and pd.notna(row[k]):
                st.session_state[k] = float(row[k])
        st.rerun()

st.divider()

# ── Input sliders ──────────────────────────────────────────────────────────────
st.subheader("Workload Features")
left, right = st.columns(2)

with left:
    st.slider(
        "assigned_memory",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="assigned_memory",
        help="Memory allocated to this workload (Borg normalised units, 0–1)",
    )
    st.slider(
        "req_memory",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="req_memory",
        help="Memory requested by this workload",
    )
    st.slider(
        "avg_memory",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="avg_memory",
        help="Average memory actually consumed during execution",
    )

with right:
    st.slider(
        "max_memory",
        min_value=0.0001, max_value=0.99, step=0.0001, format="%.4f",
        key="max_memory",
        help="Peak memory usage recorded during execution",
    )
    st.slider(
        "page_cache_memory",
        min_value=0.0, max_value=0.50, step=0.0001, format="%.4f",
        key="page_cache_memory",
        help="Memory used for page cache (proxy for I/O intensity)",
    )
    st.slider(
        "runtime_seconds",
        min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f",
        key="runtime_seconds",
        help="Workload runtime duration (Borg normalised time units, 0–1)",
    )

# Derived features preview
with st.expander("Derived features (computed automatically)"):
    am  = st.session_state["assigned_memory"]
    rm  = st.session_state["req_memory"]
    avm = st.session_state["avg_memory"]
    mxm = st.session_state["max_memory"]
    pc  = st.session_state["page_cache_memory"]
    rt  = st.session_state["runtime_seconds"]
    eps = 1e-9
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("memory_overprovisioning_ratio", f"{am/(rm+eps):.3f}")
    d2.metric("memory_utilization_avg",        f"{avm/(am+eps):.3f}")
    d3.metric("memory_utilization_peak",       f"{mxm/(am+eps):.3f}")
    d4.metric("runtime_efficiency",            f"{rt/(am+eps):.3f}")
    d5.metric("page_cache_ratio",              f"{pc/(am+eps):.3f}")

st.divider()

# ── Classify ───────────────────────────────────────────────────────────────────
if st.button("🔍  Classify Workload", type="primary", use_container_width=True):
    vals = {k: st.session_state[k] for k in INPUT_KEYS}
    fv     = build_feature_vector(vals, feature_cols)
    scaled = scaler.transform(fv)
    g      = int(kmeans.predict(scaled)[0])

    st.success(f"**Classified as: {EMOJIS[g]} Group {g} — {NAMES[g]}** &nbsp;|&nbsp; Failure rate: {FAILURE_RATE[g]}%")

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
    st.caption("Bars show mean standardised feature value. CPU features excluded.")

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
    st.plotly_chart(fig, use_container_width=True)
