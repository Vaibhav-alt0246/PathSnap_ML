# PathSnap — Member A: Data Processing, Candidate Generation, Feature Engineering & EDA

**Course:** PES University — UE24CS352A Machine Learning Hackathon  
**Role:** Member A — Data Engineering, Geometry, Feature Engineering & Visualization  
**Python Runtime:** Python 3.13  

---

## 1. Responsibilities & Boundaries

Member A is responsible for:
- Raw dataset inspection, schema validation, and data quality audits across all 80 training trips and 10 student-test trips.
- Metric Universal Transverse Mercator (UTM) coordinate projection.
- Candidate road segment generation using spatial indexing (Shapely STRtree).
- Sequence-monotonic reference route label alignment for training data.
- Extraction of 24 geometric, directional, kinematic, and topological features.
- Data leakage prevention safeguards (protecting student-test `.route` files and separating identifiers from ML features).
- Exploratory data analysis, statistical plots, and interactive Folium maps.
- Complete unit test suite and clean handoffs for Member B (ML Models) and Member C (HMM/Viterbi/FastAPI/Streamlit).

---

## 2. Directory Structure

```text
member_a/
├── README.md
├── requirements.txt
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   ├── projection/
│   │   ├── __init__.py
│   │   └── projection.py
│   ├── candidates/
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── labels/
│   │   ├── __init__.py
│   │   └── alignment.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   └── schema.py
│   └── viz/
│       ├── __init__.py
│       ├── plots.py
│       └── folium_maps.py
├── tests/
│   ├── test_loader.py
│   ├── test_validation.py
│   ├── test_projection.py
│   ├── test_candidates.py
│   ├── test_labels.py
│   └── test_features.py
├── notebooks/
│   └── member_a_eda.ipynb
├── results/
│   ├── dataset_summary.csv
│   ├── candidate_statistics.csv
│   ├── feature_summary.csv
│   └── plots/
│       ├── eda_01_trip_sizes.png
│       ├── eda_02_temporal_and_speed.png
│       ├── eda_03_candidate_recall_and_density.png
│       ├── eda_04_feature_distributions.png
│       └── eda_05_feature_correlation_matrix.png
├── maps/
│   └── trip_00000001_interactive_map.html
└── report/
    ├── official_requirements_summary.md
    ├── dataset_analysis.md
    ├── candidate_analysis.md
    ├── feature_documentation.md
    ├── design_decisions.md
    ├── limitations.md
    ├── member_b_handoff.md
    └── member_c_handoff.md
```

---

## 3. Installation & Environment Setup

```bash
# 1. Clone repository and navigate to workspace
cd PathSnap_ML

# 2. Install dependencies
pip install -r member_a/requirements.txt
```

---

## 4. Usage Guide

### 4.1 Loading a Trip
```python
from member_a.src.data.loader import load_trip

# Training trip (route allowed)
trip_train = load_trip(partition="train", trip_id="00000001", include_route=True)

# Student-test trip (route access strictly blocked)
trip_test = load_trip(partition="student_test", trip_id="00000000", include_route=False)
```

### 4.2 Candidate Generation
```python
from member_a.src.candidates.generator import generate_candidates

# Generates candidates within 50m search radius
candidates = generate_candidates(trip_train, search_radius_m=50.0, max_candidates=15)
```

### 4.3 Training Label Alignment
```python
from member_a.src.labels.alignment import align_route_labels

# Aligns ordered route sequence and appends binary 'label' column
labeled_candidates = align_route_labels(trip_train, candidates)
```

### 4.4 Feature Extraction
```python
from member_a.src.features.engineering import build_features
from member_a.src.features.schema import split_features_and_identifiers

# Extracts 24 geometric, directional, kinematic, and topological features
features_df = build_features(trip_train, labeled_candidates)

# Segregate ML features from identifiers and target
X, meta, y = split_features_and_identifiers(features_df)
```

### 4.5 Interactive Folium Map
```python
from member_a.src.viz.folium_maps import create_interactive_map

# Generate interactive HTML map
create_interactive_map(trip_train, candidates, labeled_candidates, max_points=120)
```

---

## 5. Running the Test Suite

Run all 30 automated unit and integration tests:

```bash
pytest member_a/tests -v
```

All 30 tests pass with 100% success rate:
- `test_loader.py`: File parsing, zero-based indexing, path validation, and student-test route protection.
- `test_validation.py`: Null/NaN checks, non-monotonic timestamps, index bounds, and malformed inputs.
- `test_projection.py`: UTM zone calculation, forward/inverse projections, metric Euclidean distance scaling.
- `test_candidates.py`: Perpendicular distance math, endpoint clamping, zero-length segment safety, deterministic ordering, deduplication.
- `test_labels.py`: Monotonic sequence alignment, binary labels, student-test leakage blocking.
- `test_features.py`: Feature schema adherence, zero-NaN/inf guarantees, leakage isolation, value bounds.

---

## 6. Key Deliverables & Reports

- **Official Requirements Summary:** [`member_a/report/official_requirements_summary.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/official_requirements_summary.md)
- **Dataset Inspection Analysis:** [`member_a/report/dataset_analysis.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/dataset_analysis.md)
- **Candidate Recall Analysis:** [`member_a/report/candidate_analysis.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/candidate_analysis.md)
- **Feature Schema & Documentation:** [`member_a/report/feature_documentation.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/feature_documentation.md)
- **Technical Design Decisions:** [`member_a/report/design_decisions.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/design_decisions.md)
- **Known Limitations:** [`member_a/report/limitations.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/limitations.md)
- **Member B Handoff:** [`member_a/report/member_b_handoff.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/member_b_handoff.md)
- **Member C Handoff:** [`member_a/report/member_c_handoff.md`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/report/member_c_handoff.md)
- **Interactive Folium Map:** [`member_a/maps/trip_00000001_interactive_map.html`](file:///c:/Users/vaini/OneDrive/Documents/PathSnap_ML/member_a/maps/trip_00000001_interactive_map.html)
