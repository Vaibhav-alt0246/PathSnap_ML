# Frontend/backend integration audit

Development Streamlit prediction now calls `POST {PATHSNAP_API_BASE_URL}/predict`
and obtains its map from `GET {PATHSNAP_API_BASE_URL}/map/{trip_id}`. Streamlit no
longer loads the development model or runs candidate generation, feature extraction,
classification, or HMM decoding in development mode.

The backend logs request receipt, candidate count, classifier completion, HMM
completion, prediction count, duration, and errors through `pathsnap.api` and
`pathsnap.inference` loggers.

The sklearn warning was caused by the stored estimator being fitted from arrays
while inference passed a pandas DataFrame. Inference now selects the configured
24 columns and passes their NumPy values in the same order; feature definitions and
model weights are unchanged.

Verified with trip `00000033`:

- `POST /predict`: HTTP 200
- 43,219 candidates
- 5,844 predictions
- Backend duration: 5.12 seconds
- Application logs showed all inference stages
- Full test suite: 30 passed

The evaluator mode already uses the evaluator API endpoints. The upload-trip mode
retains its local upload workflow; development and evaluator workflows are fully
API-backed. Browser-level interaction was not automated.
