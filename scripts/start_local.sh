#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$ROOT/.venv/bin/python"
: "${PATHSNAP_DATASET_ROOT:?Set PATHSNAP_DATASET_ROOT}"
: "${PATHSNAP_MODEL_PATH:?Set PATHSNAP_MODEL_PATH}"
test -f "$PYTHON" || { echo "Missing project interpreter: $PYTHON" >&2; exit 1; }
test -f "$PATHSNAP_MODEL_PATH" || { echo "Missing model: $PATHSNAP_MODEL_PATH" >&2; exit 1; }
echo "Using interpreter: $PYTHON"
echo "API: http://127.0.0.1:8000"
exec "$PYTHON" -m uvicorn api.main:app --host "${PATHSNAP_API_HOST:-127.0.0.1}" --port "${PATHSNAP_API_PORT:-8000}"
