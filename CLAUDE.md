# CLAUDE.md — WorkloadProfiler

This file captures project context, conventions, and guidelines for working in this repository. Read it fully before making changes.

---

## Project Overview

**Goal:** Discover and categorise distinct workload types in Google's large-scale compute clusters (Google Borg 2019 trace data) using unsupervised clustering, to inform scheduling, autoscaling, and resource allocation strategies.

**Stakeholders:** Infrastructure and network operations teams.

**Dataset:** Google 2019 Cluster Sample (`borg_traces_data.csv`, 313 MB, 405,894 rows, 34 columns) sourced via Kaggle Hub. A 30% random sample (~121,535 rows) is used during development for computational efficiency.

---

## Repository Structure

```
WorkloadProfiler/
├── Workload_Profiler.ipynb          # Main analysis notebook (single source of truth)
├── data/
│   └── google_cluster_sample/
│       └── borg_traces_data.csv     # Raw dataset (313 MB, do not modify)
├── plots/                           # All exported visualisations (PNG)
├── docs/
│   └── index.html                   # Project summary page (GitHub Pages)
└── README.md                        # Full project documentation
```

The **notebook is the single source of truth**. All logic, outputs, and reasoning live there. Do not create separate Python scripts unless explicitly asked.

---

## Notebook Structure (126 cells)

| Section | Content |
|---------|---------|
| 1 | Business understanding, objectives, success criteria |
| 2 | Data loading, exploration, quality assessment |
| 3 | Data preparation: column selection, cleaning, JSON parsing, feature engineering, scaling |
| 4 | Baseline models: KMeans (k=1), Runtime Quantile bins |
| 4.1–4.3 | Primary models: DBSCAN, HDBSCAN |
| 4.4, 5.1 | Model comparison, evaluation dashboard |
| 6 | (Planned) Deployment / production pipeline |

---

## Data Pipeline

### 1. Column Selection
Keep: collection_id, instance_index, machine_id, cluster, priority, start_time, end_time, assigned_memory, page_cache_memory, average_usage, maximum_usage, resource_request, cpu_usage_distribution, tail_cpu_usage_distribution, failed.

Exclude: administrative metadata (user, event, constraint), low-quality columns (cycles_per_instruction, memory_accesses_per_instruction — 31% missing), event type codes.

### 2. Cleaning
- Remove records where `end_time <= start_time`
- Drop rows missing: assigned_memory, average_usage, maximum_usage, resource_request
- Fill `vertical_scaling` with 0 (absence = no scaling)
- Fill `scheduler` with mode value
- Parse JSON string columns: `average_usage` → avg_cpu, avg_memory; `maximum_usage` → max_cpu, max_memory; `resource_request` → req_cpu, req_memory
- Convert timestamps to datetime; compute `runtime_seconds`
- Cap extreme values to 1st–99th percentile (CPU, memory, runtime)

### 3. Feature Engineering (8 derived features)
| Feature | Formula | Captures |
|---------|---------|---------|
| memory_overprovisioning_ratio | assigned_memory / req_memory | Over-allocation severity |
| avg_cpu_utilization | avg_cpu / assigned_memory | CPU efficiency |
| peak_cpu_utilization | max_cpu / assigned_memory | Peak CPU load |
| memory_utilization_avg | avg_memory / assigned_memory | Average memory use |
| memory_utilization_peak | max_memory / assigned_memory | Peak memory pressure |
| cpu_peak_to_avg_ratio | max_cpu / avg_cpu | Burstiness |
| runtime_efficiency | runtime_seconds / assigned_memory | Duration vs footprint |
| page_cache_ratio | page_cache_memory / assigned_memory | IO intensity |

### 4. Scaling
`StandardScaler` applied to all 17 clustering features. Required for distance-based metrics; always fit on training data only.

---

## Models

### Baselines
- **KMeans k=1:** Trivial single-cluster sanity check (no discriminative power by design)
- **Runtime Quantile (6 bins):** Simple benchmark; shows real models must substantially beat runtime bucketing (Silhouette ~0.13)

### Primary Models
| Model | Params | Clusters | Silhouette | DB Index | CH Score | Noise % |
|-------|--------|----------|-----------|----------|----------|---------|
| DBSCAN | eps=0.5, min_samples=10 | 216 | -0.046 | 0.583 | 3,095 | 1.4% |
| **HDBSCAN** | **min_cluster_size=50, min_samples=10** | **615** | **0.877** | **0.486** | **51,991** | **12.6%** |

**HDBSCAN is the chosen model.** It discovers natural cluster density automatically, handles varying cluster sizes, and flags true outliers as noise without forcing every workload into a group.

---

## Evaluation Criteria

| Metric | Target | Excellent | Notes |
|--------|--------|-----------|-------|
| Silhouette Score | > 0.4 | > 0.6 | Range -1 to 1; higher = better separation |
| Davies-Bouldin Index | < 1.5 | < 1.0 | Lower = better; penalises large, diffuse clusters |
| Calinski-Harabasz Score | Higher = better | — | Ratio of between-cluster to within-cluster dispersion |
| Dunn Index | > 0.5 | — | Ratio of min inter-cluster to max intra-cluster distance |
| Max cluster size | < 40% | < 20% | Prevents one dominant catch-all cluster |
| Noise fraction | < 20% | — | Too high = model too conservative |

**Stability validation (planned):** 10× random 80% subsamples, Adjusted Rand Index (ARI) target > 0.8.

---

## Key Findings

- **PCA:** PC1 (33% variance) is driven by memory scale (assigned_memory, avg_memory, max_memory, req_memory). PC2 (21%) captures utilisation efficiency. Together they explain 54% of variance.
- **Primary axis of workload variation is memory scale**, not CPU or runtime.
- HDBSCAN discovers 615 micro-patterns with exceptional cohesion (Silhouette 0.877) and no dominant cluster (max 2.2%).
- ~12.6% of workloads are classified as noise — genuine outliers, not mislabels.
- DBSCAN with fixed epsilon fails on multi-scale workload data (Silhouette -0.046).

---

## Writing Style & Documentation Conventions

> **Scope:** These conventions apply to data science and analytics projects structured around a Jupyter notebook workflow (like this one). They are not general rules for all projects.

This section is critical. The README is a direct mirror of the notebook markdown cells. Understanding this relationship is essential before writing or editing either file.

### The Notebook → README Mirror Rule

The README is **not** independently written. It is assembled by copying content directly from the notebook's markdown cells — specifically the `#### Key Insights` subsections and the section intro cells. The workflow is:

1. Write a notebook markdown cell (intro paragraph + `#### Key Insights` cell after code)
2. The content of that cell becomes the body of the corresponding README section verbatim or near-verbatim

This means: **every Key Insights cell must be written to a quality and completeness that allows it to be pasted directly into the README without rewriting**.

### Section Structure Pattern

Every subsection in both notebook and README follows this template:

```
### N.N Section Title

[1–3 intro paragraphs explaining what this section does and why — BEFORE presenting findings]

[Optional: sub-group label (plain text, not a heading) followed by bullets]

#### Key Insights

[Findings, observations, takeaways — the substance that goes into the README]
```

The intro paragraphs explain *what* the step covers and *why* it matters. The Key Insights cell contains the actual findings from the code cells.

### Heading Hierarchy

| Level | Usage | Example |
|-------|-------|---------|
| `#` | Project title only | `# Workload Profiler` |
| `##` | CRISP-DM phase (numbered) | `## 2. Data understanding` |
| `###` | Sub-section (numbered) | `### 2.3 Explore Data` |
| `####` | Sub-sub-section (unnumbered) | `#### Key Insights`, `#### Background` |

Always use the exact CRISP-DM numbering (1., 1.1, 2., 2.1 etc.) — do not invent new numbering.

### Bullet Style

- Top-level bullets: `*`
- Sub-bullets: `  *` (two-space indent + `*`)
- Derived attribute lists and format-heavy sections use `-` instead of `*`
- Never mix `*` and `-` within the same list

### Sub-group Labels (not headings)

Within a section, information is grouped using plain-text or bold labels followed by bullets — NOT as markdown headings:

```
Missing Values:

* Filled `vertical_scaling` with 0 (assumed no scaling).
* Replaced missing `scheduler` values with the most common value.

Resource Usage Conversion:

* Extracted numeric features from JSON fields:
  * `avg_cpu`, `avg_memory`
```

Labels like "Missing Values:", "Resource Usage Conversion:", "Final Status:", "Reason:", "Use:" are plain text (not `####`).

### Inline Formatting Rules

- **Column/field names:** always in `backticks` (e.g., `assigned_memory`, `vertical_scaling`)
- **Function names and values:** always in `backticks`
- **Key numbers on first mention:** **bold** (e.g., **405,894 workload instance records**, **34 features**)
- **Approximate numbers:** `~` prefix (e.g., ~22.8%, ~121k, ~31%)
- **Exact numbers:** comma-separated (e.g., 405,894 not 405894)
- **Percentages:** `%` not "percent"

### Tone & Voice

- Formal, no contractions ("we are not", never "we're not")
- "We" for describing steps taken ("we aim to", "we apply")
- Present tense for describing what models/functions *do*; past tense for reporting *results*
- "In short, ..." is the standard closing summary for a section or function description
- "Here's what is found:" is an acceptable informal lead-in before bullet findings

### Function Description Pattern

Every utility/generic function cell in the notebook follows this template:

```
**Function Name**

One-sentence description of what the function does.

It:

* Point 1
* Point 2
* Point 3

In short, it [verb] [what it accomplishes in one line].
```

### Derived Attribute Format

```
- `feature_name` = `col1` / `col2`
  One-sentence description of what it captures (parenthetical qualifier if needed).
```

Always group derived features under labeled sub-groups: "Resource Efficiency & Overprovisioning Features:", "Workload Intensity & Variability:", "Page Cache Dependency:".

### Stage/Phase Descriptions (Project Plan)

```
Stage N: Title

Goal: [one line]

* Bullet 1
* Bullet 2

Inputs: [what goes in]
Outputs: [what comes out]
Risks: [what could go wrong → mitigation]
```

### Plot References in README

Always embed plots with a descriptive alt text:

```
![Plot Title Describing What It Shows](plots/filename.png)
```

Alt text describes the *content* of the plot, not just its filename. Plot filenames are snake_case.

### Incomplete Sections

Sections not yet completed are marked with just `TODO` — nothing else. Do not write placeholder content. Do not add notes explaining what will go there.

### Model Results Format in README

```
#### Model Name

[1–2 sentences of overall characterisation]

Key observations:
- Observation 1 with metric values inline (e.g., Silhouette score of 0.1283 is low but positive, indicating...)
- Observation 2
- Observation 3

[Closing "In short" or comparison sentence]

![Model Dashboard](plots/clustering_eval_modelname.png)
```

### Data Quality / Findings Format

Structured findings use labeled groups (not headings) with sub-bullets:

```
**Summary Label**

* Group Label

  * Sub-point 1
  * Sub-point 2

* Another Group

  * Sub-point
```

**Overall** or **Final Status:** label closes quality sections with a brief summary.

### Tables

Three standard table types:
1. **Column schema:** `| Column | Data Type | Description |`
2. **Risk register:** `| Risk | Impact | Contingency Plan |`
3. **Model comparison:** custom columns per need

Always include header separator row (`| --- | --- | --- |`).

---

## Coding Conventions

- **Language:** Python 3, inside Jupyter notebook cells
- **Imports:** All imports at the top of the relevant notebook section, not scattered inline
- **Random seeds:** Always set `random_state=42` for reproducibility on any stochastic step
- **Sampling:** 30% random sample is used for development; use `random_state=42` when sampling
- **Plots:** All plots saved to `plots/` with descriptive filenames (snake_case). Always use `plt.tight_layout()` before saving. Use `seaborn` or `matplotlib`; keep consistent colour palette across related plots.
- **Metrics:** Always compute and log all four metrics (Silhouette, DB, CH, Dunn) for every model, including baselines — for direct comparison.
- **No silent failures:** If JSON parsing or type conversion fails on a row, log how many rows were dropped and why.

---

## General Clustering Best Practices (applied here)

1. **Always build baselines first.** A trivial model (single cluster, random assignment, or domain heuristic like runtime bucketing) establishes a floor. Any real model must meaningfully beat it.

2. **Scale before clustering.** Distance-based algorithms (KMeans, DBSCAN, HDBSCAN) are sensitive to feature magnitude. StandardScaler is mandatory. Refit scaler only on training/analysis data.

3. **Never rely on a single metric.** Silhouette measures cohesion + separation; DB index penalises diffuse clusters; CH favours compact clusters; Dunn measures worst-case separation. Use all four together.

4. **Cluster balance matters operationally.** A model where one cluster holds 80% of data is useless for scheduling insights. Track max/min cluster size as a first-class constraint.

5. **Noise is a feature, not a bug (for HDBSCAN/DBSCAN).** Don't try to eliminate noise by tuning until it disappears. Noise often represents genuinely atypical workloads worth investigating separately.

6. **Validate stability.** A clustering that changes significantly across data subsets is not trustworthy. ARI > 0.8 across 10 subsamples is a reasonable production bar.

7. **Dimensionality reduction for diagnosis, not for clustering input.** PCA plots are for understanding and storytelling. Feed the full scaled feature set into HDBSCAN, not PCA components.

8. **Feature engineering over raw features.** Derived ratios (overprovisioning ratio, CPU peak-to-avg) carry more signal for workload typing than raw measurements alone.

9. **Hyperparameter sensitivity.** For HDBSCAN, `min_cluster_size` controls granularity. Lower = more fine-grained clusters. Document the chosen values and rationale explicitly; don't just pick defaults.

10. **Outlier handling before clustering.** Extreme outliers (>99th percentile) distort distance-based algorithms. Cap or remove them before scaling, but document how many rows were affected.

---

## Open Questions / TODO

Modeling (implement in this order):
- [ ] Implement GMM with BIC-selected k — probabilistic primary model
- [ ] Implement KMeans elbow method (k-means++ init, inertia vs k plot, percent differential) — not as a competitor; use the elbow k to group HDBSCAN micro-clusters into macro-level workload type labels for the profiling stage
- [ ] Tune DBSCAN eps using k-distance graph method rather than guessing

Profiling and evaluation:
- [ ] Cluster profiling: `df_formatted.groupby('cluster')[feature_cols].mean()` on HDBSCAN results, using elbow k to define macro-type groupings with business labels
- [ ] Business insights: map macro-type clusters to scheduling/autoscaling recommendations
- [ ] SLA risk simulation and cost estimation per cluster
- [ ] Stability validation (ARI across 10 subsamples)

Outstanding:
- [ ] Clarify deployment scope with stakeholders: production pipeline vs proof-of-concept (see `docs/additional_docs/questions.md`)
- [ ] Final deliverable format: report, slides, or demo video (pending stakeholder input)

---

## What NOT to Do

- Do not modify `borg_traces_data.csv` directly
- Do not create new Python scripts for logic that belongs in the notebook
- Do not add new clustering algorithms without first documenting the rationale in the notebook markdown
- Do not commit large intermediate data files or model artefacts
- Do not use `random_state` values other than 42 unless doing a deliberate stability sweep
- Do not ignore noise points in HDBSCAN results — they are a meaningful output
