# Technical Design Decisions & Justifications

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Component:** Member A — Data Processing, Candidate Generation & Feature Engineering  
**Date:** November 2026  

---

## 1. Dynamic Per-Trip UTM Metric Projection

- **Problem:** Geographical coordinates $(lon, lat)$ in EPSG:4326 represent angular spherical positions. Euclidean distance on raw degrees produces anisotropic distortion depending on latitude. Furthermore, our dataset inspection revealed that trips span globally: Longitude $[-100.418538^\circ, 137.237029^\circ]$ and Latitude $[-34.640291^\circ, 61.699590^\circ]$.
- **Design Decision:** The projection engine dynamically computes the optimal Universal Transverse Mercator (UTM) zone for each trip using the median spatial centroid:
  $$\text{Zone} = \lfloor(\lambda_{\text{median}} + 180)/6\rfloor + 1$$
  $$\text{EPSG} = 32600 + \text{Zone}\ (\text{North}) \quad \text{or} \quad 32700 + \text{Zone}\ (\text{South})$$
- **Justification:** Conformal transverse Mercator preserves local angles and scales distances with distortion $< 0.1\%$ across the entire trajectory bounding box, ensuring that perpendicular distances and speeds are true physical metres.

---

## 2. Shapely 2.0 STRtree Spatial Indexing

- **Problem:** Road networks range up to $155,003$ nodes and $307,392$ arcs. Pairwise comparison of $2,000$ GPS observations against $300,000$ directed line segments requires $600$ million geometric distance evaluations per trip, taking over 2 minutes per trip.
- **Design Decision:** Road network arcs are indexed into a Sort-Tile-Recursive Tree (`shapely.STRtree`) implemented in C/GEOS.
- **Justification:** Querying all candidates within $50.0\text{ m}$ takes **$0.01\text{ s} - 0.03\text{ s}$ per trip**, enabling instantaneous candidate generation and scaling smoothly to massive networks.

---

## 3. Sequence-Monotonic Route Alignment for Training Labels

- **Problem:** In training trips, vehicles traverse an ordered sequence of road arcs. A vehicle stays on an arc for multiple GPS points before transitioning to the next connected arc. Standard nearest-segment matching creates false-positive flips at intersections and fails when routes loop or repeat road segments.
- **Design Decision:** An optimal dynamic programming sequence alignment algorithm maps GPS observations to reference route arcs subject to monotonic forward progress:
  $$s_0 \le s_1 \le \dots \le s_{T-1}$$
  minimizing the cumulative perpendicular distance to the assigned sequence of arcs.
- **Justification:** Guarantees non-decreasing forward progress along the actual path, handles loops and repeated arcs naturally, and achieves $100\%$ candidate recall on normal training trips.

---

## 4. Fallback-Nearest Spatial Mechanism

- **Problem:** While $99.8\%$ of GPS observations have road arcs within $50.0\text{ m}$, occasional high-speed wide intersections or multipath spikes can push a point $>50\text{ m}$ from the road centerline. If 0 candidates are generated, downstream HMM Viterbi decoding breaks due to missing observation states.
- **Design Decision:** A configurable `fallback_nearest=True` mechanism queries the single nearest road segment for any observation with 0 candidates within the search radius.
- **Justification:** Guarantees that every observation retains candidate arcs, preventing breaks in sequential decoding while preserving high candidate precision.

---

## 5. Low-Speed Directional Masking

- **Problem:** When a vehicle is stationary or moving at crawling speed ($< 0.5\text{ m/s}$), small GPS measurement jitter produces large random shifts in heading angle, resulting in meaningless directional noise.
- **Design Decision:** Observations with speed $< 0.5\text{ m/s}$ or displacement $< 0.5\text{ m}$ are flagged (`is_low_speed = 1`), and their heading-bearing angular delta is neutrally set to $90.0^\circ$ ($\cos(\Delta\theta) = 0.0$).
- **Justification:** Prevents spurious directional penalties from poisoning classifier training when the vehicle is stopped at traffic lights or stop signs.

---

## 6. Strict Segregation of ML Features from Identifiers

- **Problem:** Passing entity identifiers (`point_idx`, `arc_id`) or raw spatial coordinates (`lat`, `lon`) to ML models causes severe memorization of specific road IDs and geographic overfitting.
- **Design Decision:** The feature engineering module strictly segregates features into $X$ (`FEATURE_COLUMNS`) and metadata (`meta`) via `split_features_and_identifiers`.
- **Justification:** Guarantees zero identifier leakage, zero coordinate memorization, and pure relational generalizability across arbitrary unseen test trips.
