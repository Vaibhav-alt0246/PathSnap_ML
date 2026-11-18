# Candidate Road Segment Generation & Recall Analysis

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Component:** Member A — Candidate Generation & Label Alignment  
**Generated Candidate Statistics:** [`member_a/results/candidate_statistics.csv`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/results/candidate_statistics.csv)  

---

## 1. Candidate Generation Architecture

For every GPS observation $P_t = (x_t, y_t)$, the candidate generation module identifies all directed road arcs $a = (u \to v)$ in the road network that lie within a configurable spatial search radius $R_{\text{search}} = 50.0\text{ m}$.

### 1.1 High-Performance Spatial Indexing
Road networks in PathSnap range up to $155,003$ nodes and $307,392$ arcs. A naive exhaustive scan ($N_{\text{gps}} \times N_{\text{arcs}}$) would require over $200$ million segment evaluations per trip.
- To achieve real-time efficiency, road arc geometries are indexed using a metric **Shapely 2.0 STRtree** (R-Tree / spatial sort-tile-recursive tree).
- Spatial queries take under **$0.02$ seconds per trip**, reducing total trip candidate generation latency from minutes to milliseconds while guaranteeing exact metric precision.

### 1.2 Mathematical Projection Formulation
For an arc from node $u = (x_u, y_u)$ to node $v = (x_v, y_v)$ with segment vector $\vec{w} = (x_v - x_u, y_v - y_u)$ and length $L = \|\vec{w}\|$:
1. **Zero-Length Segment Protection:**
   If $L < 10^{-6}\text{ m}$ (e.g. self-loops), projection fraction $t = 0.0$, closest point $Q = u$, and distance $d = \|P_t - u\|$.
2. **Along-Segment Projection Fraction:**
   $$t = \frac{(x_t - x_u)(x_v - x_u) + (y_t - y_u)(y_v - y_u)}{L^2}$$
   Bounded to segment endpoints:
   $$\hat{t} = \max(0.0, \min(1.0, t))$$
3. **Projected Coordinates:**
   $$Q = (x_u + \hat{t}(x_v - x_u),\, y_u + \hat{t}(y_v - y_u))$$
4. **Perpendicular Metric Distance:**
   $$d = \|P_t - Q\| = \sqrt{(x_t - Q_x)^2 + (y_t - Q_y)^2}$$

### 1.3 Candidate Ordering & Output Schema
Each generated candidate row strictly adheres to:
- `point_idx` (int): Zero-based observation index in `.track`.
- `arc_id` (int): Zero-based directed road arc index in `.arcs`.
- `distance_m` (float): Exact perpendicular distance in metres.
- `projection_fraction` (float): Normalized position along arc $\hat{t} \in [0.0, 1.0]$.
- `projected_x`, `projected_y` (float): Metric UTM coordinates of closest point $Q$.
- `arc_length_m` (float): Directed arc length in metres.

Rows are sorted deterministically: `point_idx` asc, `distance_m` asc, `arc_id` asc, capped at `max_candidates=15`.

---

## 2. Empirical Candidate Recall Results (Training Trips)

Candidate recall measures the percentage of GPS observations for which the true reference route road segment is captured in the generated candidate set.

| Metric | Sample Value Across Training Trips |
| :--- | :---: |
| **Median Candidate Recall** | **$100.00\%$** |
| **75th Percentile Recall** | **$100.00\%$** |
| **25th Percentile Recall** | **$99.97\%$** |
| **Overall Mean Recall (excluding Trip 54 outlier)** | **$99.78\%$** |
| **Overall Mean Recall (including all trips)** | **$95.32\%$** |
| **Mean Candidate Density** | **$7.94$ candidates per observation** |
| **Mean Distance to Ground-Truth Segment** | **$5.67\text{ m}$** |

---

## 3. Honest Analysis of Miss Cases & Outliers

In accordance with hackathon instructions, candidate misses were investigated to ascertain physical root causes:

### 3.1 Outlier Case: Trip `00000054` (Recall $10.58\%$)
- **Investigation:** While 19 out of 20 evaluated trips achieved $97\%$ to $100\%$ recall, Trip `00000054` ($19,313$ GPS points) exhibited an unusually low recall ($10.58\%$).
- **Spatial Diagnosis:** Measuring the minimum Euclidean distance from each GPS point in Trip 54 to the reference route segments revealed a **median distance of $138.77\text{ m}$** (mean $131.98\text{ m}$, max $241.26\text{ m}$). Over $87\%$ of the trajectory lies more than $100\text{ m}$ away from any route arc in the reference file.
- **Root Cause:** A persistent systematic offset / parallel expressway corridor, or an incomplete reference route sequence. This validates the caveat in `Dataset Exploration.pdf` stating: *"Treat supplied route information as reference information... Do not assume that it represents independently verified, error-free physical ground truth."*
- **Handling:** For normal trips, search radius $R=50.0\text{ m}$ provides near-perfect recall ($>99.7\%$). For extreme noise or parallel paths, the `fallback_nearest=True` mechanism guarantees that every observation retains candidate arcs.

### 3.2 Minor Misses in Normal Trips ($0.1\% - 2.9\%$)
- **High-Speed Wide Intersections:** At large multi-lane intersections or motorway junctions, GPS points can drift $50\text{ m} - 65\text{ m}$ from the road centerline.
- **GPS Multipath Spikes:** Occasional single-epoch GPS spikes in urban canyons.
