# Official Requirements Summary: PathSnap Hackathon

**Course:** PES University — UE24CS352A Machine Learning  
**Project:** PathSnap: Distributed Map-Matching & Route Reconstruction  
**Component:** Member A — Data Processing, Candidate Generation, Feature Engineering & EDA  
**Date:** November 2026  

---

## 1. Official Documentation Inventory

All five official guideline documents in `C:\Users\vaini\OneDrive\Documents\PathSnap_Documentation` were thoroughly inspected:
1. **Main Guidelines.pdf** (2 pages) — Challenge overview, 2-laptop deployment architecture, overall work process.
2. **Dataset Exploration.pdf** (4 pages) — Dataset format specifications, partitions, raw file types, geographical data considerations, data leakage rules.
3. **Models and Training Guidelines.pdf** (6 pages) — Multi-model experimentation requirements, Bayesian optimisation, HMM/Viterbi integration, evaluation metrics, pipeline integrity.
4. **Deliverables.pdf** (7 pages) — GitHub submission rules, required technical report structure, compulsory visualisations, Folium map requirements, academic integrity.
5. **Assessment and Plagiarism Policy.pdf** (3 pages) — Evaluation standards, strict independent-work rules, anti-leakage mandates, AI policy.

---

## 2. Dataset Specifications & Structure

*(Reference: Dataset Exploration.pdf, Section 3 & 7; Deliverables.pdf, Section 2.2)*

### 2.1 File Formats
Each trip is contained in a dedicated folder containing four raw files:
- **`.track`**: 3 whitespace-separated columns without headers (`longitude`, `latitude`, `timestamp`). Nominally 1 Hz (~97% of steps exactly 1.0 s apart).
- **`.nodes`**: 2 whitespace-separated columns without headers (`longitude`, `latitude`). Zero-based row index defines `node_id`.
- **`.arcs`**: 2 integer columns without headers (`from_node_id`, `to_node_id`). Zero-based row index defines `arc_id` (representing a directed road segment).
- **`.route`**: 1 integer column without headers (`arc_id`). Represents the chronological sequence of reference road segments traversed by the vehicle.

### 2.2 Dataset Partitions
- **`train/`** (80 trips): Allowed for dataset inspection, coordinate projection, candidate generation, training label alignment, feature engineering, and cross-validation.
- **`student_test/`** (10 trips): Reserved for development-stage model comparisons and evaluation. Prediction pipeline MUST NOT read or depend on `.route` files in this partition.
- **`evaluator_test/`** (10 trips): Reserved by evaluators for live evaluation. Never touched during student development.

---

## 3. Critical Data Leakage & Evaluation Rules

*(Reference: Dataset Exploration.pdf, Section 7; Models and Training Guidelines.pdf, Section 4 & 6; Assessment and Plagiarism Policy.pdf, Section 7)*

1. **No Target Leakage:** `.route` must never be used as a model feature, nor can candidate features be derived using future/target route knowledge.
2. **Student-Test Isolation:** `.route` files in `student_test` must never be read, inspected, or used during candidate generation, feature engineering, parameter tuning, or inference.
3. **Trip-Level Grouping:** All cross-validation must be group-aware (GroupKFold by `trip_id`). No observations from the same trip may be split across train and validation folds.
4. **Prediction-Time Autonomy:** Feature engineering and candidate generation must execute seamlessly when no route or ground-truth label exists.
5. **No Identifier/Coordinate Leakage:** Machine learning feature matrices must exclude raw spatial coordinates (`latitude`, `longitude`), entity identifiers (`point_idx`, `arc_id`, `node_id`, `trip_id`), and route labels.
6. **No Hardcoding:** Hardcoded outputs, manual label overrides, or bypassing the API are strictly prohibited and will result in penalties.

---

## 4. Geographical & Projection Requirements

*(Reference: Dataset Exploration.pdf, Section 5; Main Guidelines.pdf, Section 1)*

- **Non-Euclidean Space:** Raw longitude and latitude are spherical angular coordinates; Euclidean distance on raw degrees is invalid.
- **Metric Coordinate System:** GPS coordinates and road network nodes must be projected into a suitable metric Coordinate Reference System (CRS) where distances are expressed in metres.
- **Consistent CRS:** The same CRS must be used for both GPS track points and network nodes within a trip.
- **Preservation of WGS84:** Original geographic coordinates (EPSG:4326) must be preserved for Folium map visualisations and standard interchange.

---

## 5. Candidate Generation & Recall Requirements

*(Reference: Models and Training Guidelines.pdf, Section 5; Dataset Exploration.pdf, Section 3)*

- **Observation-to-Segment Mapping:** Each GPS point must be projected onto nearby directed road arcs within a configurable spatial search radius.
- **Geometric Metrics per Candidate:** For each candidate arc, compute perpendicular distance (in metres), along-segment projection fraction $[0, 1]$, projected metric coordinates $(x, y)$, and road segment length.
- **Recall Evaluation (Training Only):** Candidate recall must be explicitly computed and reported on permitted training data (verifying whether the true reference arc is present in the candidate set).
- **Miss Case Analysis:** Must identify and document candidate misses resulting from search radius limits, GPS noise, or road network disconnects.

---

## 6. Feature Engineering & Schema Standards

*(Reference: Deliverables.pdf, Section 2.2; Models and Training Guidelines.pdf, Section 5)*

Features must cover:
- **Geometric:** Perpendicular distance to road, relative distance differences among candidates, distance rank, projection fraction, distance to start/end nodes, segment length.
- **Directional:** GPS heading, road segment bearing, absolute angular delta, cosine similarity, low-speed handling.
- **Temporal & Kinematic:** Sampling time gap ($\Delta t$), step distance, estimated speed, local movement context, candidate count per point.
- **Topological:** Node degrees (in-degree, out-degree), connectivity context, reverse twin arc indicator.
- **Data Quality:** Zero `NaN` or `inf` values; clean handling of zero-speed points and segment boundaries.

---

## 7. Folium Visualisation & Reporting Deliverables

*(Reference: Deliverables.pdf, Section 6; Main Guidelines.pdf, Section 1)*

- **Mandatory Interactive Folium Maps:** Must generate real HTML interactive maps visualizing:
  1. GPS track points (markers/circles).
  2. Candidate road segments and projections.
  3. Road network segments.
  4. Reference ground-truth routes (training trips only).
- **Deliverables:** Documented source code, unit test suite with high coverage, clean handoffs for Member B (ML models) and Member C (HMM/Viterbi/FastAPI/Streamlit), and an EDA notebook.
