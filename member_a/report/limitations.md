# Known Limitations and Architectural Boundaries

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Component:** Member A — Data Processing, Candidate Generation & Feature Engineering  
**Date:** November 2026  

---

## 1. Reference Route Data Limitations

- **Empirical Observation:** In accordance with `Dataset Exploration.pdf`, reference routes (`.route`) represent recorded operational reference paths rather than perfect ground truth.
- **Topological Disconnects:** 10 out of 80 training trips contain minor topological discontinuities ($1$ to $3$ non-contiguous transitions across hundreds of arcs).
- **Outlier Discrepancies (Trip 54):** In Trip `00000054`, the vehicle trajectory runs parallel to the supplied reference route at a median offset of $138.8\text{ m}$, yielding low reference recall ($10.58\%$). Downstream model training should either omit Trip 54 from supervised training sets or weight it accordingly.

---

## 2. Temporal Gaps & Trajectory Pauses

- **Nominal Sampling:** While $>97\%$ of observations occur at exactly $1.0\text{ Hz}$, occasional tracking dropouts occur.
- **Extreme Pause Example:** Trip `00000090` contains a pause of $153,064\text{ s}$ (over 42 hours), representing a multi-day logging session.
- **Impact on Downstream Models:** Member B and Member C should treat transitions where $\Delta t > 60\text{ s}$ as trajectory breakpoints or apply transition probability resets in the HMM rather than enforcing tight temporal continuity.

---

## 3. Search Radius vs. Candidate Density Trade-off

- **Search Radius Selection:** The default candidate search radius is set to $R_{\text{search}} = 50.0\text{ m}$ with `max_candidates = 15`.
- **Trade-off Analysis:**
  - At $R = 50.0\text{ m}$, median candidate recall is $100.00\%$ with an efficient density of $\approx 7.9$ candidates per observation.
  - Expanding the radius to $R = 100.0\text{ m}$ would marginally improve recall on extreme GPS drift cases but would double candidate density ($\approx 16-20$ candidates per point), increasing false positive candidates and doubling HMM Viterbi trellis computation cost ($O(K^2)$ per transition).
- **Mitigation:** The `fallback_nearest=True` setting ensures that even when GPS observations drift beyond $50\text{ m}$, the closest road segment is retained.

---

## 4. 2D Planar Geometric Assumption

- **Elevation & Grade:** Coordinates are projected onto a 2D metric UTM plane $(x, y)$. Altitude / vertical elevation ($z$) is not provided in raw `.track` files.
- **Implication:** Overpasses, bridges, and multi-level stacked highway interchanges cannot be differentiated by vertical clearance; candidate selection in multi-level structures relies purely on heading alignment and network connectivity.

---

## 5. Heading Estimation at Trajectory Boundaries

- **Boundary Differencing:** Central differencing $\frac{\vec{P}_{t+1} - \vec{P}_{t-1}}{2}$ is used for all interior points ($0 < t < T-1$). The first point ($t=0$) uses forward differencing $\vec{P}_1 - \vec{P}_0$, and the final point ($t=T-1$) uses backward differencing $\vec{P}_{T-1} - \vec{P}_{T-2}$.
- **Sensitivity:** If the vehicle is stationary at the exact first or last epoch, heading estimation is masked to $0^\circ$ and flagged as low speed.
