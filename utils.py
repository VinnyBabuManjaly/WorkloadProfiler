"""Shared constants for the Workload Profiler Streamlit app."""

COLORS = {
    0: "#e74c3c",   # red   - Long-Running Over-Provisioned
    1: "#3498db",   # blue  - Memory-Efficient Standard
    2: "#27ae60",   # green - Short-Running Lightweight
    3: "#8e44ad",   # purple- Memory-Intensive Specialist
}

EMOJIS = {0: "⚠️", 1: "✅", 2: "⚡", 3: "🔬"}

NAMES = {
    0: "Long-Running Over-Provisioned",
    1: "Memory-Efficient Standard",
    2: "Short-Running Lightweight",
    3: "Memory-Intensive Specialist",
}

DESCRIPTIONS = {
    0: (
        "Long runtimes, but far less memory used than allocated - a textbook over-provisioning pattern "
        "(low memory_utilization_avg). Despite the generous allocation, this group fails most often. "
        "More resources are not the fix."
    ),
    1: (
        "The well-behaved majority. Uses almost exactly the memory it was given "
        "(memory_utilization_avg ~+1.0 SD above mean) - efficient, predictable, and the strongest "
        "candidate for bin-packing and dense co-scheduling."
    ),
    2: (
        "Short, lightweight, and rock-solid consistent. Lowest noise fraction (7.4%) of any group - "
        "behaviour barely varies task to task. Ideal for high-throughput dense scheduling. "
        "Preemption-tolerant."
    ),
    3: (
        "Rare but resource-critical. Memory sits +5.8 standard deviations above the dataset mean - "
        "yet this group almost never fails (3.3% failure rate). Likely running on dedicated "
        "infrastructure. Handle with care."
    ),
}

RECOMMENDATIONS = {
    0: {
        "⚙️ Scheduling": (
            "Flag for **memory reclamation** - apply memory limits or offer-back mechanisms. "
            "Allocated memory (`assigned_memory`) sits idle despite long runtimes. "
            "Reclaiming unused capacity is the single highest-yield action for this group."
        ),
        "📉 Autoscaling": (
            "Primary target for **allocation right-sizing**: reduce `assigned_memory` toward "
            "actual `avg_memory`. This recovers cluster capacity with zero performance impact. "
            "Do not provision more - resource slack is not the problem."
        ),
        "🚨 SLA Risk": (
            "**Highest failure risk: 31.4%.** Structural causes dominate, not resource shortage. "
            "Invest in failure alerting, retry policies, and root-cause analysis - "
            "not in giving these workloads more memory."
        ),
    },
    1: {
        "⚙️ Scheduling": (
            "**Primary bin-packing candidate.** High `memory_utilization_avg` (~+1.0 SD) means "
            "these workloads fill their allocation tightly - safe to co-schedule on shared nodes "
            "without leaving large gaps. Watch for I/O bursts (elevated `page_cache_ratio`)."
        ),
        "📉 Autoscaling": (
            "Allocations are approximately **right-sized** already. Small, reactive autoscaling "
            "is appropriate. Account for occasional disk-read spikes - page cache activity can "
            "cause short-lived memory pressure on co-located tasks."
        ),
        "🚨 SLA Risk": (
            "**Moderate risk: 21.2% failure.** The most variable micro-clusters within this "
            "group carry disproportionate risk (15.2% noise share). Segment monitoring by "
            "micro-cluster to catch outliers before they escalate."
        ),
    },
    2: {
        "⚙️ Scheduling": (
            "**Ideal for dense, high-throughput scheduling.** Short runtimes and rock-solid "
            "consistency (7.4% noise - lowest of all groups) make these the easiest workloads "
            "to pack tightly. **Preemption-tolerant** - safe to pause and restart without impact."
        ),
        "📉 Autoscaling": (
            "**Minimal provisioning needed.** Static small allocations are sufficient. "
            "Startup and teardown overhead dominates over memory scaling needs - "
            "autoscaling overhead would exceed any benefit."
        ),
        "🚨 SLA Risk": (
            "**Lower risk: 14.7% failure.** The most predictable group for SLA modelling - "
            "failures are isolated and non-systemic. Strong candidate for aggressive "
            "bin-packing without SLA exposure."
        ),
    },
    3: {
        "⚙️ Scheduling": (
            "**Dedicated high-memory node pools only.** Do not bin-pack alongside other "
            "memory-intensive workloads. Do not preempt. These tasks need isolation to "
            "maintain their near-perfect reliability."
        ),
        "📉 Autoscaling": (
            "Allocation is already well-matched to usage (`memory_utilization_avg` ~+1.35 SD). "
            "**Scaling events should be rare and deliberate** - over-scaling risks displacing "
            "other high-value workloads from dedicated nodes."
        ),
        "🚨 SLA Risk": (
            "**Lowest risk: 3.3% failure.** Strong SLA commitments are fully defensible. "
            "The narrow footprint (only 6 HDBSCAN micro-clusters) makes anomaly detection "
            "straightforward - any deviation from baseline is immediately visible."
        ),
    },
}

SIZE_PCT     = {0: 33.7, 1: 39.0, 2: 25.4, 3: 1.9}
FAILURE_RATE = {0: 31.4, 1: 21.2, 2: 14.7, 3: 3.3}
N_MICRO      = {0: 204,  1: 267,  2: 140,  3: 6}
