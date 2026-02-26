# Workload Profiler

## Project Overview

This project presents a data-driven framework for automatically discovering and characterizing distinct workload types in large-scale compute clusters. Using the Google 2019 Cluster Sample (derived from Google’s production cluster traces), the system analyzes job and task-level resource requests, observed CPU and memory utilization patterns, runtime characteristics, and scheduling metadata to identify meaningful behavioral segments.

By transforming raw time-series usage data into structured workload profiles, the project applies dimensionality reduction and unsupervised clustering techniques to uncover natural workload categories. These categories may include short, bursty batch jobs, long-running latency-sensitive services, memory-intensive analytics tasks, and other operationally distinct patterns.

The primary objective is to provide actionable insights for infrastructure and network operations teams. By understanding how workloads differ in resource intensity, variability, and efficiency, stakeholders can design differentiated scheduling policies, optimize capacity planning, and implement predictive autoscaling strategies tailored to each workload type.

Ultimately, this approach enhances resource utilization, reduces overprovisioning, mitigates performance bottlenecks, and supports more intelligent, behavior-aware cluster management.

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