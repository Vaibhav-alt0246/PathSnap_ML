# Dataset Inspection and Quality Analysis

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Component:** Member A — Data Processing, Inspection & Quality Validation  
**Scope:** 80 Training Trips (`train/`) and 10 Student-Test Trips (`student_test/`)  
**Generated Dataset Summary:** [`member_a/results/dataset_summary.csv`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/results/dataset_summary.csv)  

---

## 1. Executive Summary & Key Discoveries

An automated, non-destructive audit was performed across all 90 trips (80 training, 10 student-test) to verify schema adherence, numerical validity, spatial bounds, temporal consistency, and network topology.

### Key Discoveries:
1. **Global Geographic Diversity (Critical Finding):**
   - Coordinates span globally: Longitude $[-100.418538^\circ, 137.237029^\circ]$ and Latitude $[-34.640291^\circ, 61.699590^\circ]$.
   - Trips are not localized to a single European region or UTM zone; trajectories originate from diverse global locations (North America, Europe, Australia/Asia).
   - **Architectural Implication:** Projection to metric space cannot use a fixed global EPSG code (such as EPSG:32632 or EPSG:25832). It must dynamically determine the optimal Universal Transverse Mercator (UTM) zone or local transverse Mercator projection per trip based on the trip's median spatial centroid.
2. **Strict Route File Protection:**
   - In adherence to non-leakage rules, the 10 `student_test` `.route` files were confirmed to exist on disk for completeness audits, but **their contents were never opened, parsed, or processed**.
3. **Data Integrity & Cleanliness:**
   - Zero null, missing, NaN, or non-numeric values exist across `.track`, `.nodes`, and `.arcs` across all 90 trips.
   - All timestamps are strictly monotonic ($t_{k+1} > t_k$).
   - All arc endpoints ($u, v$) strictly refer to valid zero-based `node_id` indices ($0 \le u, v < N_{\text{nodes}}$).
   - In training data, all reference route arcs refer to valid `arc_id` indices ($0 \le \text{arc\_id} < N_{\text{arcs}}$).

---

## 2. Dataset Inventory & Partition Summary

| Partition | Trips | Total GPS Observations | Road Network Nodes (Range) | Road Network Arcs (Range) | Route Arcs (Range) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** | 80 | 208,417 | 299 to 155,003 (Mean: 24,169) | 649 to 307,392 (Mean: 48,461) | 59 to 2,266 (Mean: 391.7) |
| **`student_test`** | 10 | 19,160 | 1,770 to 26,635 (Mean: 9,069) | 3,577 to 56,027 (Mean: 18,504) | *Protected / Unread* |
| **Combined** | 90 | 227,577 | 299 to 155,003 | 649 to 307,392 | — |

---

## 3. Structural & Temporal Characteristics

### 3.1 GPS Observations (`.track`)
- **Sampling Frequency:** Nominally 1.0 Hz. On average, $93.83\%$ (train) and $98.20\%$ (student-test) of consecutive timestamp deltas ($\Delta t = t_{k+1} - t_k$) are exactly $1.0\text{ s}$ ($\pm 0.01\text{ s}$).
- **Timestamp Gaps:**
  - Median $\Delta t$ is $1.0\text{ s}$.
  - Occasional tracking dropouts occur. In training trips, 75% of trips have max gaps $\le 73.75\text{ s}$. An outlier trip exhibits a long gap ($153,064\text{ s}$), representing multi-session or paused recording.
  - In student-test trips, the maximum time gap across all trips is $210.0\text{ s}$, with 75% of trips having max gaps $\le 2.75\text{ s}$.

### 3.2 Road Network Topology (`.nodes`, `.arcs`)
- **Zero-Based Indexing:** Confirmed. Node IDs strictly correspond to row indices in `.nodes`; Arc IDs strictly correspond to row indices in `.arcs`.
- **Directed Graph Representation:** Arcs represent directed road segments ($u \to v$).
- **Self-Loops:** 9 of the 90 trips contain a negligible count of self-loop arcs ($u = v$), which are filtered or safely handled during segment length and candidate generation.
- **Network Dimensions:** Wide variance in network subgraphs, ranging from small local towns ($299$ nodes, $649$ arcs) to large metropolitan areas ($155,003$ nodes, $307,392$ arcs). Efficient spatial indexing (KDTree / BallTree / R-Tree) is required to ensure candidate generation scales efficiently without exhaustive scans.

### 3.3 Reference Ground-Truth Route (`.route`, Train Only)
- **Sequence Mapping:** A vehicle traverses a sequence of directed road arcs. Because vehicles travel along arcs over multiple seconds, `num_route_arcs` (mean $391.7$) is substantially smaller than `num_gps` (mean $2,605.2$).
- **Route Continuity Audit:**
  - 70 out of 80 training trips ($87.5\%$) exhibit $100.0\%$ contiguous physical connectivity ($\text{target}(a_i) = \text{source}(a_{i+1})$ for all steps).
  - 10 training trips contain minor topological disconnects (1 to 3 non-contiguous transitions across the trip), corroborating the official documentation's note that reference routes should be treated as operational reference data rather than perfect ground truth.

---

## 4. Quality Findings & Preprocessing Decisions

1. **No Imputation Needed:** With 0 missing or NaN values across all 90 trips, synthetic numerical imputation is unnecessary and would risk injecting noise.
2. **Coordinate Reference System Selection:**
   - Universal Transverse Mercator (UTM) is selected dynamically per trip.
   - For any given trip with centroid $(\text{lon}_c, \text{lat}_c)$:
     $$\text{Zone} = \lfloor(\text{lon}_c + 180)/6\rfloor + 1$$
     $$\text{EPSG} = 32600 + \text{Zone}\ (\text{if } \text{lat}_c \ge 0) \quad \text{or} \quad 32700 + \text{Zone}\ (\text{if } \text{lat}_c < 0)$$
   - Provides conformal, area-preserving, metric-accurate coordinates where Euclidean distance equals true physical distance in metres.
3. **Low-Speed and Zero-Displacement GPS Handling:**
   - Stationary GPS points ($\Delta d \approx 0$) produce unstable bearing angles. Features must guard against zero-division and flag low-speed points explicitly.
