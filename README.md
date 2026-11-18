# PathSnap

PathSnap is a GPS map-matching system that identifies travelled road segments from noisy GPS observations and a directed road network.

## Architecture

Data loading and coordinate projection → candidate generation → feature engineering → ML candidate classification → HMM/Viterbi decoding → evaluation → FastAPI → Streamlit/Folium visualisation.

## Team responsibilities

- **Member A:** data loading, projection, candidates, development labels, features, EDA, and Folium maps.
- **Member B:** five or more ML models, trip-grouped cross-validation, comparison, metrics, Bayesian optimisation, and artifact export.
- **Member C:** integration, road graph/shortest paths, HMM/Viterbi, FastAPI, Streamlit, and testing.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dataset root remains unset until the dataset schema and location are confirmed. This is an initial scaffold; no complete pipeline or experimental results are claimed.

## Dataset format

To be documented after confirming observation fields, road-network format, CRS, and reference-route availability.
