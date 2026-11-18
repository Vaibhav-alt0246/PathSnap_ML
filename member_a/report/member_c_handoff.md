# Member C Handoff Specification: HMM, Viterbi Decoding & System Integration

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Producer:** Member A (Data Processing & Candidate Generation)  
**Consumer:** Member C (HMM Sequence Modelling, Viterbi Decoding, FastAPI & Streamlit)  
**Date:** November 2026  

---

## 1. Candidate Structure & Observation Correspondence

Member C receives the candidate DataFrame alongside predicted candidate probabilities from Member B:

| Column Name | Type | Description |
| :--- | :---: | :--- |
| `point_idx` | int | Zero-based index into `.track` ($0 \le \text{point\_idx} < T$). Multiple candidates share the same `point_idx`. |
| `arc_id` | int | Zero-based index into `.arcs`. Identifies the hidden state (directed road segment $u \to v$). |
| `distance_m` | float | Perpendicular metric distance from GPS observation to candidate segment in metres. |
| `projection_fraction` | float | Position $\hat{t} \in [0.0, 1.0]$ along segment ($0.0$ at start node $u$, $1.0$ at end node $v$). |
| `projected_x`, `projected_y` | float | Metric UTM coordinates $(x, y)$ of the closest point $Q$ on the candidate arc in metres. |
| `arc_length_m` | float | Physical length of the directed road segment in metres. |
| `prob` *(from Member B)* | float | Predicted classifier probability $P(\text{match} \mid \text{features})$. |

---

## 2. Road Network Topology & Graph Representation

Member C can build the directed routing graph directly from `trip.arcs` and `trip.nodes`:

```python
import networkx as nx
import numpy as np

# trip.arcs has columns ['u', 'v'] where row index == arc_id
u_nodes = trip.arcs['u'].to_numpy(dtype=int)
v_nodes = trip.arcs['v'].to_numpy(dtype=int)

# 1. Directed graph of nodes
G = nx.DiGraph()
# Lengths can be derived directly from projected node coordinates or arc_length_m
for arc_id, (u, v) in enumerate(zip(u_nodes, v_nodes)):
    G.add_edge(u, v, arc_id=arc_id)
```

### Directed Edge Conventions:
- Arcs are **directed**: Arc $a = (u \to v)$ allows travel only from $u$ to $v$.
- Two-way streets are represented by two distinct directed arcs: $a_1 = (u \to v)$ and $a_2 = (v \to u)$.
- Zero-based indexing: `arc_id` strictly corresponds to row indices in `.arcs`; `node_id` strictly corresponds to row indices in `.nodes`.

---

## 3. Recommended Formulation for HMM & Viterbi Decoding

### A. Hidden States
For GPS observation $t$, the set of hidden states $S_t$ consists of the candidate arcs generated for `point_idx == t`:
$$S_t = \{a \mid \text{candidates for observation } t\}$$

### B. Emission Probabilities
Member C can combine the spatial distance Gaussian with Member B's classifier probabilities:
$$P(P_t \mid a) \propto \text{prob}_a \times \exp\left(-\frac{d(P_t, a)^2}{2\sigma_{\text{GPS}}^2}\right)$$
where $\sigma_{\text{GPS}} \approx 5.0\text{ m}$ to $10.0\text{ m}$.

### C. Transition Probabilities & Network Routing
For consecutive candidate states $a_{t-1} = (u_1 \to v_1)$ and $a_t = (u_2 \to v_2)$:
- Let $\Delta t = \text{timestamp}_t - \text{timestamp}_{t-1}$.
- Great-circle / metric Euclidean distance between observations: $\Delta d = \|P_t - P_{t-1}\|$.
- Network routing distance $d_{\text{net}}(a_{t-1}, a_t)$: Shortest path length on directed graph from $v_1$ to $u_2$ plus along-segment distances:
  $$d_{\text{net}} = (1 - \hat{t}_{t-1}) L_{a_{t-1}} + \text{shortest\_path\_length}(G, v_1, u_2) + \hat{t}_t L_{a_t}$$
- Transition probability:
  $$P(a_t \mid a_{t-1}) \approx \frac{1}{\beta} \exp\left(-\frac{|\Delta d - d_{\text{net}}|}{\beta}\right)$$
- If no directed path exists on $G$, assign a small penalty $\epsilon \approx 10^{-7}$ or allow virtual jumps across gaps.

---

## 4. API & UI Deployment Boundaries

- **Laptop 1 (FastAPI backend):** Exposes `POST /predict` taking `{trip_id: str}`. Executes Member A's `load_trip` (with `include_route=False`), `generate_candidates`, `build_features`, Member B's model inference, and Member C's Viterbi decoding. Evaluates accuracy server-side on training data without exposing ground-truth route to client.
- **Laptop 2 (Streamlit UI):** Sends `trip_id` to API and renders the decoded route and interactive Folium maps.
