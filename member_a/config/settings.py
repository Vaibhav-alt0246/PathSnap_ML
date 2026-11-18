"""PathSnap Member A Configuration Settings.

Centralizes paths, parameters, schemas, and safeguards across data processing,
candidate generation, and feature engineering.
"""

from pathlib import Path
import os

# Base directory for member_a
BASE_DIR = Path(__file__).resolve().parent.parent

# Default dataset root
DEFAULT_DATA_ROOT = Path(
    os.getenv(
        "PATHSNAP_DATA_ROOT",
        r"C:\Users\vaini\OneDrive\Documents\PathSnap_Dataset\Student folder",
    )
)

# Partitions
TRAIN_PARTITION = "train"
STUDENT_TEST_PARTITION = "student_test"
PERMITTED_PARTITIONS = {TRAIN_PARTITION, STUDENT_TEST_PARTITION}

# Results and Output Directories
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MAPS_DIR = BASE_DIR / "maps"
REPORT_DIR = BASE_DIR / "report"

# Candidate Generation Parameters
CANDIDATE_SEARCH_RADIUS_M = float(os.getenv("PATHSNAP_SEARCH_RADIUS_M", 50.0))
MAX_CANDIDATES_PER_POINT = int(os.getenv("PATHSNAP_MAX_CANDIDATES", 15))

# Kinematic and Low Speed Thresholds
MIN_SPEED_FOR_HEADING_MPS = 0.5  # Below ~1.8 km/h, GPS heading is considered unreliable

# Column schemas
TRACK_COLUMNS = ["lon", "lat", "timestamp"]
NODES_COLUMNS = ["lon", "lat"]
ARCS_COLUMNS = ["u", "v"]
ROUTE_COLUMNS = ["arc_id"]

CANDIDATE_COLUMNS = [
    "point_idx",
    "arc_id",
    "distance_m",
    "projection_fraction",
    "projected_x",
    "projected_y",
    "arc_length_m",
]

IDENTIFIER_COLUMNS = ["point_idx", "arc_id", "trip_id"]
TARGET_COLUMN = "label"
