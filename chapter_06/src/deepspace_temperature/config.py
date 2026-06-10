from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "temperature_observations.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
PREPROCESS_BUNDLE_PATH = MODEL_DIR / "preprocess_bundle.joblib"
MODEL_PATH = MODEL_DIR / "mlp_temperature_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "solar_flux",
    "distance_au",
    "orbital_radius",
    "albedo",
    "heater_power",
    "radiator_area",
    "instrument_load",
    "attitude_angle",
]

TARGET_COLUMN = "temperature_c"


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        REPORT_DIR,
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
