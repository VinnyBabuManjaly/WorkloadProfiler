"""Shared constants for the Workload Profiler Streamlit app."""

COLORS = {
    0: "#e74c3c",   # red   — Long-Running Over-Provisioned
    1: "#3498db",   # blue  — Memory-Efficient Standard
    2: "#27ae60",   # green — Short-Running Lightweight
    3: "#8e44ad",   # purple— Memory-Intensive Specialist
}

EMOJIS = {0: "⚠️", 1: "✅", 2: "⚡", 3: "🔬"}

NAMES = {
    0: "Long-Running Over-Provisioned",
    1: "Memory-Efficient Standard",
    2: "Short-Running Lightweight",
    3: "Memory-Intensive Specialist",
}

DESCRIPTIONS = {
    0: "Long runtime with below-average memory utilisation. Primary over-provisioning and high-failure-risk group.",
    1: "Largest group. Near-average memory allocation but highest utilisation efficiency — uses what it is given.",
    2: "Short runtime, low memory footprint, most consistent behaviour. Ideal for high-throughput bin-packing.",
    3: "Extreme memory scale (+5.8 SD), very low failure rate. Rare but critical production workloads.",
}

RECOMMENDATIONS = {
    0: {
        "Scheduling": (
            "Flag for memory reclamation. Apply memory limits or offer-back mechanisms — "
            "allocated memory is consistently idle despite long runtimes."
        ),
        "Autoscaling": (
            "Primary target for allocation reduction. Reduce `assigned_memory` toward actual "
            "`avg_memory` to recover significant cluster capacity without affecting performance."
        ),
        "SLA Risk": (
            "Highest risk (31.4% failure). Resource slack is not preventing failures — "
            "structural causes dominate. Prioritise failure alerting and retry policies over further provisioning."
        ),
    },
    1: {
        "Scheduling": (
            "Primary candidate for bin-packing. High memory utilisation (+1.0 SD) means "
            "these workloads fill their allocation without leaving slack on the node."
        ),
        "Autoscaling": (
            "Right-sizing opportunity — current allocations are approximately correct. "
            "Account for I/O burst events (above-average page cache ratio)."
        ),
        "SLA Risk": (
            "Moderate risk (21.2% failure). Monitor the most variable micro-clusters within "
            "this group — they carry disproportionate failure risk (15.2% noise share)."
        ),
    },
    2: {
        "Scheduling": (
            "High-throughput bin-packing target. Short runtime and consistent behaviour "
            "(7.4% noise — lowest of all groups) make these ideal for dense node packing. Preemption-tolerant."
        ),
        "Autoscaling": (
            "Minimal provisioning required. Static small allocations are sufficient — "
            "startup and teardown overhead dominates over memory scaling needs."
        ),
        "SLA Risk": (
            "Second-lowest risk (14.7% failure). Most predictable group for SLA modelling — "
            "failure events are isolated and non-systemic."
        ),
    },
    3: {
        "Scheduling": (
            "Assign to dedicated high-memory node pools. Do not preempt or bin-pack "
            "alongside other memory-intensive workloads."
        ),
        "Autoscaling": (
            "Size allocation to actual memory request — utilisation is already high (+1.35 SD). "
            "Scaling events should be rare and deliberate."
        ),
        "SLA Risk": (
            "Lowest risk (3.3% failure). Strong SLA commitments are defensible. "
            "Monitor for sudden deviations — the narrow 6-micro-cluster footprint makes anomalies easy to detect."
        ),
    },
}

SIZE_PCT     = {0: 33.7, 1: 39.0, 2: 25.4, 3: 1.9}
FAILURE_RATE = {0: 31.4, 1: 21.2, 2: 14.7, 3: 3.3}
N_MICRO      = {0: 204,  1: 267,  2: 140,  3: 6}
