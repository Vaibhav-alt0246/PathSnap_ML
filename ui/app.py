"""Streamlit dashboard for real PathSnap predictions."""
import os
import streamlit as st
import requests
import streamlit.components.v1 as components
from src.config import load_config

st.set_page_config(page_title="PathSnap | GPS Map Matching", page_icon="🗺️", layout="wide")
st.markdown("""<style>
.stApp{background:#f7f9fc;color:#172033}.block-container{max-width:1400px;padding-top:2rem}
.hero{background:white;border:1px solid #dfe6ef;border-radius:14px;padding:1.6rem 2rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(23,32,51,.04)}
.hero h1{color:#102a43;margin:0;font-size:2.5rem}.hero p{color:#486581;margin:.4rem 0 0;font-size:1.05rem}
.section{color:#102a43;font-size:1.25rem;font-weight:700;margin:1.2rem 0 .6rem}
.method{background:white;border-left:4px solid #168aad;border-radius:8px;padding:1rem 1.2rem;color:#334e68;line-height:1.55;border:1px solid #e5ebf2}
[data-testid="stMetric"]{background:white;border:1px solid #dfe6ef;border-radius:10px;padding:.7rem}
[data-testid="stExpander"]{background:white;border:1px solid #dfe6ef;border-radius:10px;color:#172033}
[data-testid="stExpanderDetails"]{background:white;color:#172033}
[data-testid="stElementContainer"] iframe{opacity:1!important;filter:none!important}
[data-testid="stElementContainer"]:has(iframe){opacity:1!important;filter:none!important}
</style>""", unsafe_allow_html=True)
config=load_config()
dataset_root=os.environ.get("PATHSNAP_DATASET_ROOT") or config["paths"].get("dataset_root")
api_base=os.environ.get("PATHSNAP_API_BASE_URL", config.get("ui",{}).get("api_base_url", "http://127.0.0.1:8000")).rstrip("/")
st.markdown("""<div class="hero"><h1>PathSnap</h1><p><b>Intelligent GPS Map Matching</b></p><p>Transform noisy GPS trajectories into plausible road-network routes using machine learning and sequence modelling.</p></div>""",unsafe_allow_html=True)
with st.sidebar:
    mode=st.radio("Workflow", ["Development prediction", "Upload trip", "Evaluator mode"])
    st.header("Trip selection")
    st.caption("Choose an existing trip, upload a trip, or use the server-side evaluator workflow.")
    if st.button("Refresh trips"): st.rerun()
    if mode == "Upload trip":
        uploaded = st.file_uploader("Upload .track, .nodes and .arcs files", type=["track", "nodes", "arcs"], accept_multiple_files=True)
        upload_run = st.button("Run uploaded trip", type="primary", use_container_width=True)
    else:
        try:
            trips_response = requests.get(f"{api_base}/trips", timeout=10); trips_response.raise_for_status()
            trip_ids = trips_response.json().get("trips", [])
        except requests.RequestException as exc:
            st.error(f"Backend unavailable at {api_base}: {exc}"); st.stop()
        if not trip_ids: st.warning("No training trips are available from the backend."); st.stop()
        trip_id=st.selectbox("Available trip",trip_ids)
        st.caption(f"{len(trip_ids)} trips discovered")
        run=st.button("Run prediction",type="primary",use_container_width=True)

if mode == "Upload trip":
    st.markdown('<div class="section">Upload and predict</div>', unsafe_allow_html=True)
    st.info("Upload the three network/track files. Reference .route files are not accepted or used for prediction.")
    if upload_run:
        try:
            with st.spinner("Running map matching for uploaded trip…"):
                response = requests.post(f"{api_base}/upload/predict", files=[("files", item) for item in (uploaded or [])], timeout=600)
                response.raise_for_status(); result=response.json()
            st.session_state.prediction=result; st.session_state.prediction_trip=result["trip_id"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            st.session_state.prediction=None; st.error(f"Uploaded trip failed: {exc}")
    result=st.session_state.get("prediction")
    if result is None:
        st.info("Choose the three files above and click Run uploaded trip.")
    else:
        diagnostics=result["diagnostics"]; confidence=result["confidence"]
        cols=st.columns(5); cols[0].metric("GPS observations",diagnostics["observations"]); cols[1].metric("Candidate rows",diagnostics["candidate_rows"]); cols[2].metric("Classifier selections",len(result["classifier_arc_ids"])); cols[3].metric("HMM route arcs",len(result["predicted_arc_ids"])); cols[4].metric("Mean confidence",f"{confidence['mean_max_probability']:.3f}")
        st.markdown('<div class="section">Interactive route map</div>', unsafe_allow_html=True)
        components.html(result["map_html"], width=None, height=650, scrolling=False)
    st.stop()

if mode == "Evaluator mode":
    st.markdown('<div class="section">Reserved evaluator trips</div>', unsafe_allow_html=True)
    st.caption("Evaluator routes remain server-side. The UI receives predictions and aggregate metrics only.")
    try:
        response=requests.get(f"{api_base}/evaluation/trips", timeout=10); response.raise_for_status(); evaluator_ids=response.json().get("trips", [])
        if not evaluator_ids: st.warning("No evaluator trips are available from the configured API."); st.stop()
        eval_id=st.selectbox("Evaluator trip", evaluator_ids); eval_run=st.button("Run evaluation", type="primary")
        if eval_run:
            with st.spinner(f"Running server-side evaluation for {eval_id}…"):
                result=requests.post(f"{api_base}/evaluation/predict", json={"trip_id":eval_id}, timeout=300); result.raise_for_status(); payload=result.json()
            st.session_state.evaluation_payload=payload
        payload=st.session_state.get("evaluation_payload")
        if payload and payload.get("trip_id")==eval_id:
            metrics=payload.get("evaluation",{}); cols=st.columns(4)
            cols[0].metric("Arc precision",f"{metrics.get('arc_set_precision',0):.3f}"); cols[1].metric("Arc recall",f"{metrics.get('arc_set_recall',0):.3f}"); cols[2].metric("Arc F1",f"{metrics.get('arc_set_f1',0):.3f}"); cols[3].metric("LCS recall",f"{metrics.get('lcs_route_recall',0):.3f}")
            st.json(payload.get("diagnostics",{}))
            map_response=requests.get(f"{api_base}/evaluation/map/{eval_id}", timeout=300); map_response.raise_for_status(); components.html(map_response.text, height=700, scrolling=False)
    except requests.RequestException as exc:
        st.error(f"Evaluator API unavailable: {exc}")
    st.stop()
if "prediction" not in st.session_state: st.session_state.prediction=None
if "prediction_trip" not in st.session_state: st.session_state.prediction_trip=None
if run:
    try:
        with st.spinner(f"Running map matching for trip {trip_id}…"):
            response = requests.post(f"{api_base}/predict", json={"trip_id": trip_id}, timeout=600)
            response.raise_for_status(); result = response.json()
        st.session_state.prediction=result; st.session_state.prediction_trip=trip_id
    except (requests.RequestException, ValueError, KeyError) as exc:
        st.session_state.prediction=None; st.session_state.prediction_trip=None; st.error(f"Prediction failed for {trip_id}: {exc}")
result=st.session_state.prediction
if result is None:
    st.markdown('<div class="section">How PathSnap works</div>',unsafe_allow_html=True)
    st.markdown("""<div class="method"><b>1. Candidate generation</b> finds plausible directed road arcs near each GPS observation.<br><b>2. Feature engineering</b> computes the approved 24 candidate-level features.<br><b>3. Machine learning</b> uses HistGradientBoosting to estimate candidate probabilities.<br><b>4. HMM + Viterbi</b> uses directed connectivity and sequence information to decode a plausible route.<br><b>5. Final route</b> displays the ordered trip-local road arcs on an interactive map.</div>""",unsafe_allow_html=True)
    st.info("Select a trip in the sidebar and choose Run prediction to begin.")
else:
    selected=result["trip_id"]; st.markdown(f'<div class="section">Prediction result — trip {selected}</div>',unsafe_allow_html=True)
    diagnostics=result["diagnostics"]; confidence=result["confidence"]; cols=st.columns(5)
    cols[0].metric("GPS observations",diagnostics["observations"]); cols[1].metric("Candidate rows",diagnostics["candidate_rows"]); cols[2].metric("Classifier selections",len(result["classifier_arc_ids"])); cols[3].metric("HMM route arcs",len(result["predicted_arc_ids"])); cols[4].metric("Mean confidence",f"{confidence['mean_max_probability']:.3f}")
    st.caption(f"Processing time: {result.get('processing_seconds',0):.2f}s · Missing candidates: {len(diagnostics['missing_candidate_observations'])}")
    try:
        map_response=requests.get(f"{api_base}/map/{selected}", timeout=600); map_response.raise_for_status(); st.markdown('<div class="section">Interactive route map</div>',unsafe_allow_html=True); components.html(map_response.text,width=None,height=650,scrolling=False)
    except requests.RequestException as exc: st.error(f"Map rendering failed: {exc}")
    with st.expander("Classifier versus HMM route"):
        st.write("Classifier selections are highest-probability candidates per observation. The HMM route is sequence-decoded; neither is ground truth.")
        st.write({"classifier_arc_count":len(result["classifier_arc_ids"]),"hmm_arc_count":len(result["predicted_arc_ids"]),"first_hmm_arcs":result["predicted_arc_ids"][:10]})
    with st.expander("Prediction diagnostics"): st.json(diagnostics)

