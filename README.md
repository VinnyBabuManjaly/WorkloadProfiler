# Workload Profiler

## Project Overview

This project presents a data-driven framework for automatically discovering and characterizing distinct workload types in large-scale compute clusters. Using the Google 2019 Cluster Sample (derived from Google’s production cluster traces), the system analyzes job and task-level resource requests, observed CPU and memory utilization patterns, runtime characteristics, and scheduling metadata to identify meaningful behavioral segments.

By transforming raw time-series usage data into structured workload profiles, the project applies dimensionality reduction and unsupervised clustering techniques to uncover natural workload categories. These categories may include short, bursty batch jobs, long-running latency-sensitive services, memory-intensive analytics tasks, and other operationally distinct patterns.

The primary objective is to provide actionable insights for infrastructure and network operations teams. By understanding how workloads differ in resource intensity, variability, and efficiency, stakeholders can design differentiated scheduling policies, optimize capacity planning, and implement predictive autoscaling strategies tailored to each workload type.

Ultimately, this approach enhances resource utilization, reduces overprovisioning, mitigates performance bottlenecks, and supports more intelligent, behavior-aware cluster management.

The full analysis and implementation is available in the **[Jupyter Notebook: Workload_Profiler.ipynb](./Workload_Profiler.ipynb)**, which delivers actionable insights.

 The final report is available at **[Workload Profiler Report](https://vinnybabumanjaly.github.io/WorkloadProfiler/)**.

## 1 Business understanding

### 1.1 Business Objectives

#### Task 

The goal is to understand, from a business perspective, what the organization wants to achieve with workload analysis and resource optimization. The analyst must uncover the key factors influencing operational efficiency, resource usage, and scheduling, ensuring the project focuses on solving the right business problems rather than just producing technical results.


#### Background

The organization manages large-scale compute clusters with diverse workloads, including short batch jobs, long-running latency-sensitive services, and memory-intensive analytics tasks. Current challenges include inefficient resource utilization, overprovisioning, unpredictable job performance, and limited insights for designing effective scheduling and autoscaling policies.

#### Business Objectives

Primary Objective:

* Automatically discover and categorize distinct workload types to improve operational efficiency and resource allocation across clusters.

Related Business Questions:

* Which workloads consume the most resources and how consistently?
* Are there overprovisioned tasks or jobs that could be scaled down without impacting performance?
* How can workload grouping inform predictive autoscaling and differentiated scheduling policies?
* Can clusters be managed to reduce potential SLA violations or job slowdowns?

#### Business Success Criteria

* Clear identification of meaningful workload groups with distinct resource and runtime patterns.
* Actionable insights for infrastructure teams to optimize scheduling and autoscaling strategies.
* Improved resource efficiency, measured by simulated KPIs such as CPU/memory utilization, cost estimation, and SLA risk scores.
* Subjective validation by stakeholders that the workload profiles and metrics are relevant, interpretable, and support operational decision-making.

### 1.2 Assess Situation


#### Inventory of Resources

Data Resources

* Google 2019 Cluster Sample dataset (job/task events, CPU/memory usage, scheduling metadata)
* Aggregated per-job/task feature tables (engineered metrics)

Technical Resources

* Python (Pandas, NumPy, Scikit-learn)
* PCA and clustering libraries (K-Means, GMM, DBSCAN)
* Visualization tools (Matplotlib / Seaborn)
* Local workstation or cloud compute environment

Expertise

* Data analysis and machine learning knowledge
* Understanding of distributed systems and workload behavior
* Basic business understanding of infrastructure operations

#### Requirements, Assumptions, and Constraints

Requirements

* Identify 4–8 meaningful workload clusters
* Ensure results are interpretable for operations teams
* Maintain data privacy and comply with dataset usage terms
* Deliver clear documentation and visualizations

Assumptions

* CPU and memory usage patterns reflect workload behavior accurately
* Resource usage is a reasonable proxy for cost and SLA risk
* Clusters discovered represent operationally meaningful categories

Constraints

* No real cost, SLA, or business KPI data available
* Large dataset size may require sampling or aggregation
* Simulated cost and SLA metrics are approximations, not exact values

#### Risks and Contingencies

| Risk                            | Impact                  | Contingency Plan                                          |
| ------------------------------- | ----------------------- | --------------------------------------------------------- |
| Clusters are not well separated | Low interpretability    | Try alternative algorithms (GMM, DBSCAN), refine features |
| High dimensionality noise       | Poor clustering quality | Apply PCA and feature selection                           |
| Dataset scale limitations       | Performance issues      | Use sampling or distributed processing                    |
| Simulated KPIs lack realism     | Limited business value  | Clearly communicate assumptions and limitations           |

#### Terminology

Business Terms

* Workload – A job or task executed in the cluster
* Overprovisioning – Allocating more resources than actually used
* Autoscaling – Automatically adjusting resources based on demand
* SLA (Service Level Agreement) – Performance and reliability commitments

Data Science Terms

* PCA (Principal Component Analysis) – Reduces correlated features into key components
* Clustering – Grouping similar workloads based on patterns
* Silhouette Score – Measures cluster separation quality
* Feature Engineering – Transforming raw data into modeling-ready metrics

#### Costs and Benefits

Costs

* Engineering time for data processing and modeling
* Compute resources for large-scale clustering
* Time spent validating and interpreting clusters

Potential Benefits

* Improved resource utilization and reduced overprovisioning
* Smarter scheduling and autoscaling policies
* Reduced risk of slowdowns and SLA violations
* Better operational visibility into workload behavior

Even without direct cost data, the project can support measurable improvements in efficiency and decision-making.

### 1.3 Determine Data Mining Goals


#### Data Mining Goals

The goal of this project is to use clustering techniques to automatically discover distinct workload types based on CPU usage, memory consumption, runtime behavior, and scheduling metadata.

Specifically, we aim to:

* Engineer meaningful features from raw task and job-level data
* Reduce dimensionality using PCA to simplify complex patterns
* Apply clustering algorithms (e.g., K-Means, GMM) to group similar workloads
* Profile and interpret each cluster in terms of usage patterns, efficiency, and SLA risk
* Generate simulated KPIs such as resource efficiency, cost estimation, and potential SLA violations

The final output should be clearly defined workload categories that operations teams can use for smarter scheduling and autoscaling decisions.

#### Data Mining Success Criteria

The project will be considered successful if:

* Clusters show clear separation (e.g., strong Silhouette Score or other validation metrics)
* Each cluster has distinct and interpretable resource usage patterns
* Results are stable across sampling or parameter changes
* The identified workload groups translate into actionable operational insights

Ultimately, success means the technical outputs (clusters and metrics) are reliable, explainable, and useful for decision-making — not just mathematically correct.

### 1.4 Project Plan


This project will be completed in structured phases to ensure both the business goals and technical goals are achieved.


Stage 1: Data Understanding

Goal: Understand the dataset structure and available features.

* Review all columns and data types
* Identify missing values and inconsistencies
* Understand job/task relationships

Inputs: Raw cluster dataset
Outputs: Clean understanding of data schema
Risks: Large dataset size → may require sampling

Stage 2: Data Preparation & Feature Engineering

Goal: Convert raw logs into meaningful workload features.

* Calculate runtime, average/peak CPU and memory
* Compute efficiency and overprovisioning metrics
* Normalize and scale features

Inputs: Raw dataset
Outputs: Modeling-ready feature table
Dependencies: Stage 1 must be completed

Stage 3: Dimensionality Reduction

Goal: Simplify high-dimensional features.

* Apply PCA
* Select components explaining most variance
* Visualize component behavior

Outputs: Reduced feature dataset
Risk: Too much information loss → adjust number of components


Stage 4: Clustering & Modeling 

Goal: Identify workload groups.

* Apply K-Means (k = 4–8)
* Evaluate using Silhouette Score
* Compare with alternative models (GMM or DBSCAN if needed)

Outputs: Cluster labels for workloads
Iteration: Repeat modeling if clusters are not well separated

Stage 5: Cluster Profiling & KPI Simulation

Goal: Translate clusters into business insights.

* Analyze CPU/memory patterns per cluster
* Simulate cost, SLA risk, efficiency metrics
* Assign meaningful names to workload types

Outputs: Interpretable workload categories + KPI dashboard


Stage 6: Evaluation & Review

Goal: Validate usefulness and stability.

* Check cluster consistency
* Validate interpretability
* Review with stakeholders

Success Criteria:

* Clear separation between clusters
* Actionable insights for scheduling and autoscaling


#### Risk Management & Contingency

* If clustering quality is poor → refine features or try different algorithms
* If dataset is too large → apply sampling or aggregation
* If results lack interpretability → simplify features and improve profiling

#### Review Points

At the end of each stage, results will be reviewed and the plan updated if needed. The modeling and evaluation stages may be repeated until meaningful workload groups are found.

## 2. Data understanding

### 2.1 Collect Initial Data

The first step of this project is to acquire and load the Google 2019 Cluster Sample dataset. This dataset contains job and task-level logs, including resource requests, CPU/memory usage patterns, scheduling metadata, and task lifecycle events.

This step ensures the data is accessible and ready for deeper understanding and preparation.

Dataset Name: [Google 2019 Cluster Sample](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample)
Source: Kaggle (derived from Google cluster trace v3)

Storage Location:[Borg Traces Data](./data/borg_traces_data.csv)

Method of Acquisition:
* Downloaded dataset from Kaggle.
* Extracted raw CSV file from compressed file.
* Loaded selected tables into Python using Pandas.

### 2.2 Describe Data

After collecting the initial dataset, the next step is to understand its overall structure and surface characteristics. At this stage, we are not cleaning or transforming the data yet. Instead, we examine:

* The format of the data
* The number of records and features
* The meaning of each column
* Basic statistical properties
* Whether the dataset is suitable for our SLA, KPI, and cost modeling objectives

This step helps confirm that the data aligns with the project goals before moving into deeper preparation and modeling.

Each row represents a single instance execution record, and each column represents either:

* Identification information (e.g., collection_id, instance_index)
* Scheduling metadata
* Resource request details
* Resource usage measurements
* Performance indicators (e.g., failure status)

The dataset is structured and tabular, making it well-suited for analysis using Python (Pandas).

The dataset contains **405,894 workload instance records** with **34 features**, providing a large and detailed view of cluster activity. It includes scheduling metadata, resource requests, actual CPU and memory usage, timing information, and a failure indicator.

All critical fields required for SLA, KPI, and cost modeling, such as `start_time`, `end_time`, `assigned_memory`, `average_usage`, and `failed` are complete with no missing values. Some performance-specific metrics (e.g., cycles per instruction) contain missing values, but they do not affect the core objectives of this project.

A few columns are stored as strings and will require preprocessing before modeling. Overall, the dataset is structurally sound, sufficiently large, and well-suited for analyzing resource efficiency, reliability, and cost behavior in a cloud workload environment.

| Column                            | Data Type | Description                                                 |
| --------------------------------- | --------- | ----------------------------------------------------------- |
| `Unnamed: 0`                      | int64     | Original index column from CSV, can be ignored.             |
| `time`                            | int64     | Timestamp of the event (Unix epoch format).                 |
| `instance_events_type`            | int64     | Type of event for a workload instance (numeric code).       |
| `collection_id`                   | int64     | Unique identifier for a collection of workloads.            |
| `scheduling_class`                | int64     | Scheduling category of the workload.                        |
| `collection_type`                 | int64     | Type of workload collection (numeric code).                 |
| `priority`                        | int64     | Priority level assigned to the workload.                    |
| `alloc_collection_id`             | int64     | ID of the allocated collection for this instance.           |
| `instance_index`                  | int64     | Index of the workload instance within its collection.       |
| `machine_id`                      | int64     | Identifier of the machine executing the workload.           |
| `resource_request`                | str       | Requested resources (CPU, memory) for the workload.         |
| `constraint`                      | str       | Constraints on workload placement or execution.             |
| `collections_events_type`         | int64     | Type of event for the collection (numeric code).            |
| `user`                            | str       | User or owner of the workload.                              |
| `collection_name`                 | str       | Name of the collection.                                     |
| `collection_logical_name`         | str       | Logical name of the collection (business-readable).         |
| `start_after_collection_ids`      | str       | Dependencies: collection IDs that must finish first.        |
| `vertical_scaling`                | float64   | Whether vertical scaling is enabled for the workload (0/1). |
| `scheduler`                       | float64   | Scheduler assigned to handle the workload.                  |
| `start_time`                      | int64     | Start timestamp of the workload (Unix epoch).               |
| `end_time`                        | int64     | End timestamp of the workload (Unix epoch).                 |
| `average_usage`                   | str       | JSON string of average resource usage (CPU, memory).        |
| `maximum_usage`                   | str       | JSON string of maximum resource usage (CPU, memory).        |
| `random_sample_usage`             | str       | JSON string of sampled resource usage.                      |
| `assigned_memory`                 | float64   | Memory assigned to the workload (GB or normalized).         |
| `page_cache_memory`               | float64   | Memory used for page cache.                                 |
| `cycles_per_instruction`          | float64   | CPU cycles per instruction (performance metric).            |
| `memory_accesses_per_instruction` | float64   | Memory accesses per CPU instruction.                        |
| `sample_rate`                     | float64   | Sampling rate for resource usage measurements.              |
| `cpu_usage_distribution`          | str       | JSON string representing CPU usage distribution.            |
| `tail_cpu_usage_distribution`     | str       | JSON string representing tail CPU usage.                    |
| `cluster`                         | int64     | Cluster ID where the workload ran.                          |
| `event`                           | str       | Description or type of event (text).                        |
| `failed`                          | int64     | Whether the workload failed (0 = success, 1 = failed).      |

### 2.3 Explore Data

In this step, we dive deeper into the dataset to uncover patterns, distributions, and relationships that will guide modeling and analysis. The goal is to answer preliminary data mining questions and identify interesting trends or anomalies.

* Total workloads: 405,894
* Failed workloads: 92,678 (~22.8% failure rate)
* Clusters identified: 8 distinct workload groups

Runtime (seconds) insights:

* Most workloads are short, with the median runtime at 300 seconds (5 minutes).
* 25% of workloads run less than 41 seconds, and 75% run 300 seconds or less, indicating a large concentration of very short tasks.
* Maximum runtime is capped at 300 seconds, suggesting possible system-imposed limits or sampling constraints.

CPU usage:

* Average CPU usage is generally low, with a median of 0.0010 (normalized or fractional usage).
* Only a few workloads use a significant portion of CPU, with a max observed usage of 0.538.
* Most workloads are lightweight in CPU demand.

Memory usage:

* Memory assigned is modest for the majority of workloads, median 0.0027.
* 75% of workloads use less than 0.0067, but the highest memory-consuming task uses 0.286.
* Most workloads are memory-light, with only a few memory-intensive tasks.

Priority levels:

* Workloads are highly varied across priority levels, with some clusters like `103` and `360` having tens of thousands of workloads.
* Lower-numbered priorities (0, 25) have relatively few workloads, while mid-range priorities dominate.
* Indicates that the system schedules a mix of high-priority and batch workloads.

Cluster distribution:

* Workloads are fairly evenly distributed across 8 clusters, each ranging between ~42k and ~59k workloads.
* Suggests a good separation of workload types for clustering and behavior profiling.

Overall takeaway:

* The dataset is dominated by short, lightweight CPU and memory workloads, with a few outliers using more resources.
* Failure rate (~23%) is significant and may require investigation or prioritization in scheduling.
* Clusters appear balanced, making them useful for differentiated scheduling and autoscaling policies.

### 2.4 Data Quality

**Data Quality Summary**

* Missing Values

  * Most columns are complete.
  * Only `memory_accesses_per_instruction` and `cycles_per_instruction` have significant missing data (~31%).
  * Minor missing values in `vertical_scaling` (0.24%), `scheduler` (0.24%), and `resource_request` (0.19%).

* Duplicate Rows

  * No duplicate rows were found (excluding complex dict/list columns).

* Numeric Data Overview

  * Runtime (seconds): Most workloads complete within 300 seconds; mean runtime ~212s.
  * CPU Usage: Mostly low; median CPU usage ~0.001, maximum ~0.54.
  * Memory Usage: Mostly small; median assigned memory ~0.0027, maximum ~0.29.

* Invalid or Negative Values

  * No negative runtime, CPU, or memory usage values detected.

* Workload Failures

  * ~23% of workloads failed (`failed = 1`).

* Priority Levels

  * Majority of workloads use priorities 103, 200, 0, or 360.
  * Other priority levels appear less frequently, indicating uneven distribution across priorities.

* *Cluster Distribution

  * Workloads are distributed across 8 clusters; cluster 3 and 6 have the highest number of workloads (~58k each).

* Event Types

  * Both `instance_events_type` and `collections_events_type` have identical distributions.
  * Most common event types: 3, 2, 0, 6, and 5. Rare types like 8, 9, 10 appear very infrequently.

**Overall** 

* Data is mostly complete, numeric fields are valid, and no duplicates exist.
* Some missing data in CPU/memory instruction metrics may need attention for advanced analysis.
* Categorical fields (`priority`, `cluster`, `failed`) show expected distributions and are consistent.

## 3. Data preparation

### 3.1 Select Data

At this stage, we decide which parts of the dataset will actually be used for modeling.
Data selection is not only about choosing the right columns (features), but also about filtering the appropriate rows (records).

The goal is to retain data that is:

- Relevant to workload behavior and clustering
- Sufficient in quality and completeness
- Technically manageable given memory and computation constraints

Since this project focuses on discovering workload types using resource behavior and runtime characteristics, only features that meaningfully describe workload performance are selected

#### Included Columns

The following columns were selected because they directly contribute to workload behavior analysis, efficiency measurement, or clustering:

Identification Fields (Retained for grouping, not clustering):

* `collection_id`
* `instance_index`
* `machine_id`
* `cluster`
* `priority`

These help profile and interpret clusters after modeling but are not necessarily used as clustering features.

Time & Runtime Features (Critical):

* `start_time`
* `end_time`

Used to compute:

* Runtime
* Scheduling behavior
* SLA risk indicators

Resource Allocation & Usage (Core Features):

* `assigned_memory`
* `page_cache_memory`
* `average_usage`
* `maximum_usage`
* `cpu_usage_distribution`
* `tail_cpu_usage_distribution`
* `resource_request`

These features describe:

* CPU intensity
* Memory intensity
* Usage variability
* Overprovisioning behavior

These are central to workload profiling.

Reliability & KPI Fields:

* `failed`

Used to simulate SLA risk and analyze workload stability.


#### Excluded Columns

The following fields were excluded because they do not contribute meaningfully to clustering or workload behavior modeling:

Administrative / Metadata Fields:

* `Unnamed: 0`
* `collection_name`
* `collection_logical_name`
* `user`
* `constraint`
* `start_after_collection_ids`
* `event`

Reason:
These fields are descriptive or textual metadata and do not represent quantitative workload behavior.

Low-Level Performance Metrics (Excluded for Simplicity):

* `cycles_per_instruction`
* `memory_accesses_per_instruction`
* `sample_rate`

Reason:
These metrics:

* Contain significant missing values (~31%)
* Add noise due to hardware-level variability
* Are not required for workload grouping objectives

They may be considered in future advanced analysis.

Event Type Columns:

* `instance_events_type`
* `collections_events_type`

Reason:
These represent lifecycle event codes and are not indicators of workload resource patterns.

#### Row Selection Criteria

In addition to column filtering, we apply row-level filtering:

Remove Invalid Runtime Records:

We exclude rows where:

* `end_time <= start_time`
* Runtime is zero or negative

Remove Missing Critical Resource Values:

Rows missing:

* `assigned_memory`
* `average_usage`
* `maximum_usage`

are removed because clustering depends on these metrics.


#### Optional Sampling (Technical Constraint)

Because the dataset contains 405,894 records, sampling may be applied during experimentation to:

* Improve iteration speed
* Reduce memory load
* Test model stability

Full dataset is used for final modeling where feasible.

#### Key Insights
Final Outcome of Data Selection

After applying column and row selection:

* Only behavior-relevant workload features remain
* Noise from metadata and unused fields is removed
* Invalid or incomplete records are excluded
* Dataset becomes modeling-ready
* Dimensionality is reduced, improving clustering quality

This structured selection ensures that the clustering process focuses purely on workload behavior patterns rather than administrative or irrelevant metadata.


Data selection was guided by three principles:

1. Relevance to workload behavior and clustering
2. Data quality and completeness
3. Practical computational constraints

The resulting dataset provides a clean, focused representation of workload performance characteristics, suitable for dimensionality reduction and unsupervised clustering.

### 3.2 Clean Data

After selecting relevant columns and records, the next step is to improve data quality to ensure it is suitable for clustering and workload profiling.

Cleaning focuses on:
* Handling missing values
* Fixing incorrect data types
* Removing invalid or inconsistent records
* Parsing structured fields (JSON strings)
* Standardizing formats for modeling

The goal is not to “perfect” the data, but to make it reliable and consistent enough for dimensionality reduction and clustering algorithms.

* Cleaned the dataset to make it fully ready for clustering and workload profiling.
* Focused on fixing small missing values, converting JSON fields to numeric features, standardizing time data, and controlling extreme values.

Missing Values:

* Filled `vertical_scaling` with 0 (assumed no scaling).
* Replaced missing `scheduler` values with the most common value.
* Dropped a small number of rows with missing `resource_request`, since it is essential for efficiency analysis.

Resource Usage Conversion:

* Extracted numeric features from JSON fields:

  * `avg_cpu`, `avg_memory`
  * `max_cpu`, `max_memory`
  * `req_cpu`, `req_memory`
* Removed the original JSON columns, leaving a clean numeric dataset.

Time & Runtime:

* Converted timestamps to proper datetime format.
* Created `runtime_seconds` as a new feature.
* All runtimes are valid and positive.

Outlier Control:

* Capped extreme values (1st–99th percentile) for key features like CPU, memory, and runtime.
* This prevents rare extreme workloads from distorting clustering results.

Final Status:

* No missing values in retained fields.
* All features numeric and consistent.
* Dataset is clean, stable, and ready for feature engineering and clustering.

### 3.3 Construct Data


In this step, we create derived attributes from the cleaned dataset to better capture workload characteristics for clustering and profiling. These new features emphasize resource efficiency, overprovisioning behavior, and usage patterns that directly relate to workload types.

The construction focuses on ratios and efficiency metrics rather than raw values, as these normalized measures are more stable across different scales and better reveal behavioral patterns in distance-based clustering.

#### Derived Attributes Created

Resource Efficiency & Overprovisioning Features:

- `memory_overprovisioning_ratio` = `assigned_memory` / `resource_request`  
  Measures how much more memory was assigned than requested (values > 1 indicate overprovisioning).

- `avg_cpu_utilization` = `avg_cpu` / `assigned_memory`  
  CPU utilization relative to assigned resources (captures underutilization).

- `peak_cpu_utilization` = `max_cpu` / `assigned_memory`  
  Peak CPU demand relative to assigned capacity.

- `memory_utilization_avg` = `avg_memory` / `assigned_memory`  
  Average memory utilization efficiency.

- `memory_utilization_peak` = `max_memory` / `assigned_memory`  
  Peak memory utilization efficiency.

Workload Intensity & Variability:

- `cpu_peak_to_avg_ratio` = `max_cpu` / `avg_cpu`  
  Indicates bursty vs steady CPU workloads (higher values = more bursty).

- `runtime_efficiency` = `runtime_seconds` / `assigned_memory`  
  Runtime normalized by resource allocation (long-running low-resource jobs vs short high-resource jobs).

Page Cache Dependency:

- `page_cache_ratio` = `page_cache_memory` / `assigned_memory`  
  Proportion of assigned memory used for caching (IO-intensive workloads tend to have higher values).

These derived attributes transform the raw allocation/usage data into behavioral signals that are more suitable for discovering workload types through clustering.

hese 8 new attributes provide a compact, interpretable representation of:
- Resource provisioning efficiency (overprovisioning ratios)
- Utilization patterns (avg vs peak behavior)  
- Workload burstiness (peak-to-average ratios)
- IO characteristics (page cache dependency)

The derived dataset now contains both raw measurements and behavioral ratios, enabling clustering algorithms to discover workload types based on actual resource usage patterns rather than just absolute scale.

### 3.4 Integrate Data

Since this project uses a single Google cluster workload dataset, no table merging was needed. All relevant fields-identification, timestamps, resource usage, and failure status-are already co-located in one table per workload instance.

The only "integration" happened within records: parsing JSON strings from `average_usage`, `maximum_usage`, and `resource_request` columns into separate numeric CPU/memory fields (`avg_cpu`, `avg_memory`, etc.). This transformed semi-structured data into a flat, fully numeric format ready for modeling.

**Output:** Single integrated dataset with 121k+ records combining all cleaned, parsed, and derived workload features.

### 3.5 Format Data

No major formatting changes were needed since scikit-learn clustering algorithms accept the current DataFrame structure directly. The dataset is already fully numeric (after JSON parsing) with proper data types.

Minor formatting applied:
- Reordered columns to group features logically: identification fields first (`collection_id`, `machine_id`, etc.), then time features, then raw resource measurements, then derived efficiency ratios.
- Randomized row order using `df.sample(frac=1, random_state=42)` to prevent any modeling bias from the original collection sequence.
- Confirmed all numeric features are `float64` (no strings/integers remaining).

Choose which columns are features for clustering

For unsupervised clustering:

* Exclude pure ID fields (collection_id, instance_index, machine_id, cluster, priority).
* Exclude raw datetimes (start_time, end_time), but keep runtime_seconds.
* Exclude the target-like field failed (used later for profiling, not clustering).

Scale the features

Clustering methods (HDBSCAN, DBSCAN, KMeans) assume distances in feature space are meaningful. To avoid any one feature dominating because of its scale (e.g., seconds vs ratios), standardization is done.

- `X` = raw numeric features from `df_formatted`.  
- `X_scaled` = standardized version (mean 0, variance 1), used in `evaluate_clustering_model`.

## 4. Modeling

### 4.1 Modeling Techniques

#### Primary Models

HDBSCAN (Main Model)

* Automatically detects clusters of different sizes and densities.
* Identifies outliers without forcing every workload into a group.
* No need to predefine the number of clusters.
* Best suited for discovering natural workload types.
Use: Primary production model for workload profiling.

DBSCAN (Validation Model)

* Classic density-based clustering.
* Requires tuning of `eps`, but useful for confirming HDBSCAN results.
* Also detects noise and outliers.
Use: Cross-validation of cluster structure.

#### Baseline Models (Sanity Checks)

Single Cluster (KMeans, k=1)

* Assumes all workloads are identical.
* Used to confirm that real clustering adds value.
* Real models must clearly outperform this trivial case.

Runtime Quantile Split (5–6 bins)

* Groups workloads by runtime only (short → long).
* Simple and intuitive benchmark.
* Real clustering should show better structure than this basic segmentation.


#### Assumptions Before Modeling

* No missing values (handled in Clean Data).
* All features numeric and formatted.
* Large enough sample size (~121k workloads).
* Feature scaling required → apply `StandardScaler`.


#### Workflow

1. Scale features
2. Fit HDBSCAN and evaluate results
3. Fit DBSCAN for validation
4. Compare both against baseline models
5. Select the best-performing technique

HDBSCAN is used as the primary technique, validated against DBSCAN and simple baselines to ensure the discovered workload types are stable, meaningful, and operationally useful.

### 4.2 Test Design


Because this is an unsupervised clustering project, a train/test split is not required. Instead, we run all models on the full dataset (~121k workloads) and validate whether the discovered clusters are high quality, stable, and operationally meaningful.


#### Model Validation

**Quality Check - Are clusters well formed?**

* Silhouette Score → target > 0.4 (good), > 0.6 (excellent)
* Davies-Bouldin Index → target < 1.0
* Dunn Index → target > 0.5
* Calinski-Harabasz Score → maximize
* WCSS (Within-Cluster Sum of Squares) → assess compactness

Most importantly, real models must outperform the baselines:

* Single Cluster baseline (expected silhouette ~0.0)
* Runtime Quantile baseline (expected silhouette ~0.2-0.3)

Success rule: Real clustering must improve silhouette by at least +0.3 over the Single Cluster baseline.

**Stability Check - Are results repeatable?**

To ensure robustness:

* Create 10 random 80% subsets (~97k records each)
* Re-run HDBSCAN and DBSCAN
* Compare runs using Adjusted Rand Index (ARI)

Target: Mean ARI > 0.8 → stable workload types.

For HDBSCAN, we also check:

* Cophenetic Correlation (> 0.7) to confirm hierarchical structure quality.

**Structural & Balance Checks**

* No cluster larger than 40% of the dataset
* No cluster smaller than 1%
* Reasonable cluster tightness and separation

This ensures practical and balanced profiling.

**Business Validation**

Beyond metrics, clusters must make real-world sense. For that, analyze:

* Runtime patterns
* CPU and memory usage behavior
* Failure rate differences (`failed`)

Clusters should provide insights relevant to SLA monitoring and capacity planning.


#### Final Success Criteria

The model is considered successful if it:

* Clearly beats both baselines
* Achieves Silhouette > 0.4
* Shows ARI stability > 0.8
* Produces balanced, interpretable workload types

In short, the clusters must be statistically strong, repeatable, and business-relevant before being approved for production workload profiling.

### 4.3 Build Model

#### Generic Functions

**Dunn Index Function** 

This function computes the Dunn Index, a metric to evaluate clustering quality.

It:

* Ignores noise points (`-1` labels).
* Checks early exit: returns NaN if fewer than 2 clusters.
* For large datasets (>10k samples), it skips the expensive calculation and approximates using the silhouette score.
* Computes:

  * Intra-cluster distances: maximum distance within each cluster (compactness).
  * Inter-cluster distances: minimum distance between clusters (separation).
* Returns the Dunn Index: min inter-cluster distance ÷ max intra-cluster distance.

Higher values : well-separated, tight clusters.

**Cluster size distribution statistics**

This function summarizes the cluster size distribution.

* Removes noise points (label `-1`).
* If no valid clusters remain, it returns zeros/NaN.
* Counts how many clusters exist.
* Calculates:
  - Total number of clusters
  - Largest cluster size (as % of data)
  - Smallest cluster size (as % of data)

Overall, it helps check whether clusters are balanced or if one cluster dominates the dataset.

**Generic Visualization Function**

This function generates a complete 2×2 visual dashboard to evaluate clustering results and saves it as PNG and PDF.

It shows:

* Cluster size distribution (excluding noise) with counts and percentages.
* Clustering quality metrics (Silhouette, DB, Dunn, etc.) in a bar chart.
* PCA 2D projection of the data, colored by cluster, plus printed PCA variance and top feature loadings.
* Silhouette score distribution with mean score and quality indicator (Fair / Good / Excellent).

It also prints PCA diagnostics, displays noise percentage, handles edge cases (single cluster or all noise), and saves the final plot automatically.

In short, it turns clustering results into a clear, executive-ready visual and analytical summary.

**Generic Clustering Evaluation Function**

This function is a complete clustering evaluation wrapper.

It:

* Fits the clustering model (or uses precomputed labels for baselines).
* Measures fit time.
* Counts clusters and noise points.
* Computes quality metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz, Dunn) when valid.
* Calculates cluster size statistics (largest and smallest cluster %).
* Stores all results in a structured dictionary for comparison.
* Optionally generates the visualization dashboard.

In short, it standardizes how every clustering model in the project is evaluated, compared, and reported.

#### Single Cluster Baseline

The single-cluster baseline behaves exactly as expected and serves as a “worst-case” reference, not a useful clustering solution.

- The model placed all 121,535 workloads into one cluster, with no noise points. This makes the largest and smallest cluster both 100%, confirming there is no segmentation at all in this baseline.

- Because there is only one cluster, all clustering quality metrics (silhouette, Davies-Bouldin, Dunn, etc.) are undefined or skipped, which is correct and highlights that this model does not provide any structure to evaluate.

- The PCA summary simply tells you about overall variance structure of the dataset, not clustering quality:

  - PC1 and PC2 together explain about 54% of the variance, so a 2D projection captures over half of the data’s variability.
  - The listed “Feature i” loadings show which features drive PC1 and PC2, but in a single-cluster baseline they only describe dominant directions of variation, not distinct groups.

In short, this baseline confirms that “everything is one workload type” is trivial and uninformative; any meaningful clustering model only needs to produce more than one coherent cluster with valid metrics to improve on this baseline.

#### Runtime Quantile Baseline

The Runtime Quantile baseline provides a modest but credible reference, representing the hypothesis "workload types = runtime length buckets."

Key observations:
- Created 3 clusters (not 6 due to `duplicates='drop'` handling ties in quantiles), with a highly imbalanced distribution: 66.5% in the largest cluster (80,795 workloads), 15.4% in the smallest (21,979). This shows runtime data has natural concentration points.

- Silhouette score of 0.1283 is low but positive, indicating weak but real structure - runtime quantiles capture *some* separation in the feature space.
- Davies-Bouldin = 2.70 (high/poor) confirms clusters are not well-separated; Dunn Index approximation (0.103) is also weak.

- Calinski-Harabasz = 8,794 is decent but uninformative without comparison.
- PCA structure identical to single-cluster baseline (as expected - same data), showing runtime quantiles provide some segmentation but don't align perfectly with the dominant variance directions (PC1/PC2).

This baseline beats the single-cluster dummy (0.128 > undefined) by creating runtime-based groups, but its imbalanced clusters and low silhouette (0.13) set a low bar. Real models (HDBSCAN/DBSCAN) should target silhouette >0.3 and more balanced cluster sizes to show they capture richer resource behavior patterns beyond just runtime length.

### 4.4 Assess Model

#### Generic functions for comparison

**Get Summary Table**

This function aggregates results from the `all_results` list into a formatted Pandas DataFrame for easy model comparison.

* Consolidates Metrics: Gathers Silhouette, DB Index, and CH Score into one view.
* Auto-Formatting: Converts raw decimals into readable percentages ($1.0\%$) and rounded strings ($3$ decimals).
* Data Safety: Uses `.get()` to handle missing metrics gracefully with `NaN`.
* Dual Output: Prints an ASCII table for immediate review and returns a DataFrame for further analysis.

**Plotting clustering comparison**

This function generates a grouped bar chart to visually compare key clustering metrics across different models.

* Multi-Metric Visualization: Plots Silhouette, DB Index, and Noise % side-by-side for each model.
* Data Cleaning: Automatically converts formatted table strings (like "85%") back into numeric floats for accurate plotting.
* Smart Labeling: Includes logic to display "NaN" in red on the baseline if a metric is missing, ensuring no data gaps are ignored.
* Dynamic Scaling: Automatically adjusts bar widths and x-axis ticks based on the number of models and metrics provided.

#### Assessing models

ummary table shows both baselines evaluated successfully, with clear quality gap:

KMeans Single Cluster (1 cluster):
- No metrics (all NaN) as expected for trivial baseline
- 100% max cluster confirms no segmentation

Runtime Quantile Baseline (3 clusters):
- Silhouette 0.128 - weak separation, but better than nothing
- DB Index 2.70 - poor cluster quality 
- 66.5% max cluster - highly imbalanced
- Valid metrics confirm it provides minimal structure

Key takeaway: Runtime baseline sets a low but realistic bar (silhouette ~0.13). Any proper model must:
1. Beat silhouette > 0.20 
2. Lower DB Index < 2.0
3. More balanced clusters (<40% max size)

Ready for HDBSCAN/DBSCAN - they should substantially outperform these baselines to justify density-based clustering over simple runtime bucketing.

## 5. Evaluation

### 5.1 Evaluate results

At this stage, we have only evaluated the baseline models. These serve as reference points, not final solutions. Based on the current results, the business objectives have not yet been met.

The main goal was to discover meaningful workload types based on CPU, memory, and runtime behavior, and then use those types to support capacity planning and SLA risk profiling.

Here’s what is found:

* Single Cluster Baseline
  This model grouped everything into one cluster.
  It provides no segmentation, no insight, and no business value.
  It simply confirms the worst-case scenario: treating all workloads as identical.

* Runtime Quantile Baseline
  This split workloads based only on runtime.
  While it produced three clusters, they were highly uneven, with about 66% of workloads in one group.
  It captures some structure (Silhouette = 0.128), but separation is weak and resource behavior (CPU/memory) is ignored.

In short, neither baseline meets the objectives of discovering meaningful workload types or enabling actionable resource profiling.

#### Key Observations

1. Data Quality is Strong

    * 121,535 clean records available
    * All resource ratios successfully computed
    * PCA shows 54% variance explained by first two components
    This confirms the dataset is suitable for clustering.

2. Runtime Alone is Not Enough

The quantile split collapsed into only three clusters instead of six due to skewed runtime distribution.
Most workloads fall into a single bin, which explains the imbalance.

This clearly shows that runtime by itself is not sufficient for identifying workload types.

3. Feature Engineering is Ready

Derived metrics like overprovisioning ratios and utilization are stable and usable.
Special cases (e.g., zero-memory jobs) were handled properly.

The foundation is solid, we now need stronger clustering models.

At this stage:

* No meaningful workload types discovered yet
* No actionable segmentation for overprovisioning analysis
* No SLA risk profiling possible
* Baselines confirm the problem is real but unsolved

The runtime baseline gives us a minimum benchmark (Silhouette = 0.128). Any serious clustering model must clearly outperform this.

#### What This Means

The project is in progress, not complete.

We have:

* Established baselines
* Validated data quality
* Confirmed runtime segmentation is insufficient

What remains:

* Run HDBSCAN and DBSCAN on the full feature set
* Aim for silhouette > 0.3
* Achieve more balanced clusters (<40% max cluster share)
* Profile clusters using CPU, memory, overprovisioning, and failure rates

#### Clear Next Step

Move forward to advanced density-based clustering (Phase 4.3–4.4).

If HDBSCAN produces stable, balanced clusters with stronger separation, we can begin real workload profiling and capacity planning analysis.

For now, the baselines show that the challenge is valid, but meaningful workload discovery requires more sophisticated modeling.

### 5.2 Review Process

TODO

### 5.3 Next Steps

TODO

## 6. Deployment

TODO

### 6.1 Deployment Plan

TODO

### 6.2 Monitoring and Maintenance Plan

TODO

### 6.3 Final Report

TODO

### 6.4 Review Project

TODO