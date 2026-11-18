# Member B Handoff Specification: Machine Learning Candidate Classification

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Producer:** Member A (Data Processing & Feature Engineering)  
**Consumer:** Member B (Machine Learning Model Selection, Training & Evaluation)  
**Date:** November 2026  

---

## 1. Quick-Start Integration Code

Member B can generate the exact training and evaluation datasets using the following standardized Python workflow:

```python
from member_a.src.data.loader import load_trip
from member_a.src.candidates.generator import generate_candidates
from member_a.src.labels.alignment import align_route_labels
from member_a.src.features.engineering import build_features
from member_a.src.features.schema import split_features_and_identifiers, FEATURE_COLUMNS

# -------------------------------------------------------------
# 1. TRAINING PIPELINE (Uses permitted train/ partition)
# -------------------------------------------------------------
trip = load_trip(partition="train", trip_id="00000001", include_route=True)

# Step A: Candidate road arcs within 50m search radius
candidates = generate_candidates(trip, search_radius_m=50.0, max_candidates=15)

# Step B: Ground-truth binary label alignment
labeled_candidates = align_route_labels(trip, candidates)

# Step C: Extract 24 geometric, directional, kinematic, and topological features
features_df = build_features(trip, labeled_candidates)

# Step D: Split into ML feature matrix X, metadata identifiers, and target vector y
X_train, meta_train, y_train = split_features_and_identifiers(features_df)

print("X_train shape:", X_train.shape)       # (N_candidates, 24)
print("y_train distribution:\n", y_train.value_counts(normalize=True))

# -------------------------------------------------------------
# 2. INFERENCE / STUDENT-TEST PIPELINE (No route access)
# -------------------------------------------------------------
test_trip = load_trip(partition="student_test", trip_id="00000000", include_route=False)
test_candidates = generate_candidates(test_trip, search_radius_m=50.0, max_candidates=15)
test_features_df = build_features(test_trip, test_candidates)

X_test, meta_test, _ = split_features_and_identifiers(test_features_df)
# Member B feeds X_test to trained model: y_prob = model.predict_proba(X_test)[:, 1]
# Attach probabilities back to metadata: meta_test['prob'] = y_prob
```

---

## 2. Column Roles & Data Leakage Rules

### A. Pure Machine Learning Feature Matrix $X$ (`FEATURE_COLUMNS`)
Only the following 24 numerical columns should be fed into your classification algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, etc.):

1. **Geometric:** `distance_m`, `distance_to_closest_cand`, `distance_ratio_closest`, `dist_rank`, `cand_count_for_point`, `projection_fraction`, `dist_to_segment_start_m`, `dist_to_segment_end_m`, `arc_length_m`
2. **Directional:** `gps_heading_deg`, `road_bearing_deg`, `heading_bearing_diff_deg`, `heading_bearing_cos_sim`, `is_low_speed`
3. **Kinematic & Temporal:** `gps_delta_t_s`, `gps_step_distance_m`, `gps_speed_mps`, `gps_accel_mps2`, `speed_ratio_to_mean`
4. **Topological:** `u_in_degree`, `u_out_degree`, `v_in_degree`, `v_out_degree`, `has_reverse_twin`

### B. Prohibited Columns (DO NOT pass to ML models)
- **Identifiers:** `point_idx`, `arc_id`, `u`, `v`, `trip_id` (Passing these will cause memorization/overfitting).
- **Absolute Coordinates:** `lat`, `lon`, `x_m`, `y_m` (Trips are from different global UTM zones).
- **Target Label:** `label` (Target leakage!).

### C. Target Variable $y$
- Column name: `label`
- Binary classification: `1` = ground-truth road arc for that observation; `0` = non-matching candidate.
- Class prevalence: $\approx 16.1\%$ positive (natural $1:5$ to $1:6$ class imbalance). Member B may use `scale_pos_weight` in boosting or `class_weight='balanced'` in logistic regression/SVM.

---

## 3. Recommended Validation Scheme

In compliance with official hackathon guidelines:
- **Group-Aware Splitting:** Always use `GroupKFold(n_splits=5)` grouped by `trip_id` across the 80 training trips.
- **Never split observations from the same trip across train and validation sets.**
- **Evaluation Metrics:**
  - Macro F1-score & Binary F1-score
  - Log Loss / Brier Score (critical for well-calibrated emission probabilities in Member C's HMM)
  - Observation Top-1 Accuracy: $\text{Accuracy} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left(\arg\max_{a} P(a \mid P_t) = a^*_t\right)$.
