# Feature Engineering Documentation & Schema

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Component:** Member A — Candidate Feature Engineering  
**Schema Definition:** [`member_a/src/features/schema.py`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/src/features/schema.py)  
**Feature Summary Statistics:** [`member_a/results/feature_summary.csv`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/results/feature_summary.csv)  

---

## 1. Overview & Data Leakage Prevention Guarantees

The feature engineering pipeline transforms candidate road-arc projections and GPS movement dynamics into 23 numerical and boolean features for candidate classification by Member B.

### Strict Leakage Prevention Architecture:
1. **No Target Leakage:** Ground-truth reference routes (`.route`) or target labels (`label`) are strictly excluded from the feature extraction process. `build_features` executes identically whether labels exist or not.
2. **No Identifier Leakage:** Identifiers (`point_idx`, `arc_id`, `u`, `v`, `trip_id`) and geographic/metric coordinates (`lat`, `lon`, `x_m`, `y_m`) are segregated into metadata (`meta`) and excluded from the ML feature matrix $X$.
3. **Partition Protection:** Features are purely relational, normalized, and coordinate-free, preventing geographic memorization across different global regions.
4. **Row Preservation:** The output DataFrame strictly preserves the exact row count and order of the input candidate DataFrame.

---

## 2. Feature Schema & Definitions

### 2.1 Geometric Features (Road-to-Observation Proximity)
| Feature Name | Type | Unit / Range | Mathematical / Physical Definition |
| :--- | :---: | :---: | :--- |
| `distance_m` | float | $[0.0, 50.0]\text{ m}$ | Shortest perpendicular Euclidean distance from GPS point to candidate arc in metric UTM space. |
| `distance_to_closest_cand` | float | $\ge 0.0\text{ m}$ | Relative distance penalty: $d_i - \min_{j \in \text{cands}} d_j$. Equal to 0.0 for the closest candidate. |
| `distance_ratio_closest` | float | $\ge 1.0$ | Relative distance ratio: $d_i / (\min_j d_j + 10^{-4})$. Highlights dominant candidates. |
| `dist_rank` | float | $\{0, 1, \dots, K-1\}$ | Zero-based rank of the candidate among all candidates for that GPS observation sorted by distance. |
| `cand_count_for_point` | float | $[1, K]$ | Total number of candidates generated for this GPS observation. Reflects road density. |
| `projection_fraction` | float | $[0.0, 1.0]$ | Along-arc fraction $\hat{t} = \text{clip}((\vec{P} - \vec{u}) \cdot \vec{w} / L^2, 0, 1)$. |
| `dist_to_segment_start_m` | float | $\ge 0.0\text{ m}$ | Physical distance from projected point $Q$ to start node $u$: $\hat{t} \times L$. |
| `dist_to_segment_end_m` | float | $\ge 0.0\text{ m}$ | Physical distance from projected point $Q$ to end node $v$: $(1 - \hat{t}) \times L$. |
| `arc_length_m` | float | $> 0.0\text{ m}$ | Total physical length of directed road segment $L = \|v - u\|$. |

### 2.2 Directional & Alignment Features
| Feature Name | Type | Unit / Range | Mathematical / Physical Definition |
| :--- | :---: | :---: | :--- |
| `gps_heading_deg` | float | $[0.0^\circ, 360.0^\circ)$ | GPS trajectory heading computed via central difference vector $\vec{P}_{t+1} - \vec{P}_{t-1}$. Masked to 0.0 if stationary. |
| `road_bearing_deg` | float | $[0.0^\circ, 360.0^\circ)$ | Geographic compass bearing of directed road segment from start node $u$ to end node $v$. |
| `heading_bearing_diff_deg`| float | $[0.0^\circ, 180.0^\circ]$ | Minimal angular difference $|\theta_{\text{heading}} - \theta_{\text{bearing}}|$. Masked to $90^\circ$ for stationary points. |
| `heading_bearing_cos_sim` | float | $[-1.0, 1.0]$ | Cosine similarity $\cos(\Delta\theta)$. $+1.0$ indicates perfect alignment; $-1.0$ indicates opposing direction (one-way violation). Neutral $0.0$ when stationary. |
| `is_low_speed` | int | $\{0, 1\}$ | Binary indicator ($1$ if GPS speed $< 0.5\text{ m/s}$ or step displacement $< 0.5\text{ m}$). |

### 2.3 Kinematic & Temporal Trajectory Context
| Feature Name | Type | Unit / Range | Mathematical / Physical Definition |
| :--- | :---: | :---: | :--- |
| `gps_delta_t_s` | float | $\ge 1.0\text{ s}$ | Sampling time delta between consecutive observations ($t_k - t_{k-1}$). |
| `gps_step_distance_m` | float | $\ge 0.0\text{ m}$ | Metric step displacement traversed since previous observation. |
| `gps_speed_mps` | float | $\ge 0.0\text{ m/s}$ | Instantaneous GPS speed: $\Delta d / \Delta t$. |
| `gps_accel_mps2` | float | $\text{m/s}^2$ | Estimated longitudinal acceleration: $(v_k - v_{k-1}) / \Delta t$. |
| `speed_ratio_to_mean` | float | $\ge 0.0$ | Normalized speed relative to the trip's mean speed ($v / \bar{v}_{\text{trip}}$). |

### 2.4 Topological Road Network Features
| Feature Name | Type | Unit / Range | Mathematical / Physical Definition |
| :--- | :---: | :---: | :--- |
| `u_in_degree` | float | $\ge 1.0$ | Number of incoming directed arcs entering start node $u$. |
| `u_out_degree` | float | $\ge 1.0$ | Number of outgoing directed arcs leaving start node $u$. |
| `v_in_degree` | float | $\ge 1.0$ | Number of incoming directed arcs entering end node $v$. |
| `v_out_degree` | float | $\ge 1.0$ | Number of outgoing directed arcs leaving end node $v$. |
| `has_reverse_twin` | int | $\{0, 1\}$ | Binary indicator ($1$ if opposite arc $v \to u$ exists in road network, signifying a two-way street; $0$ for one-way). |

---

## 3. Verification & Quality Metrics

- **Missing Values:** $0.0\%$ NaN or infinite values across all 24 features in 42,479 evaluated candidate rows.
- **Label Class Balance:** In training candidates, positive class prevalence is $\approx 16.1\%$, reflecting a natural and manageable $1:5$ class imbalance ratio per GPS point.
- **Inference Ready:** Compatible with scikit-learn, XGBoost, LightGBM, and PyTorch pipelines via `split_features_and_identifiers`.
