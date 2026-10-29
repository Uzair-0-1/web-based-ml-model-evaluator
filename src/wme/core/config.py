import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
DATASETS_DIR = STORAGE_DIR / "datasets"
REPORTS_DIR = STORAGE_DIR / "reports"
LOGS_DIR = STORAGE_DIR / "logs"
DATABASE_PATH = STORAGE_DIR / "wme.db"

# Ensure runtime directories exist
for directory in [STORAGE_DIR, MODELS_DIR, DATASETS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Constraints (per SRS Section 2.5 & Section 5.1)
MAX_FILE_SIZE_MB = int(os.environ.get("WME_MAX_FILE_SIZE_MB", "100"))
SESSION_TIMEOUT_MINUTES = int(os.environ.get("WME_SESSION_TIMEOUT_MINUTES", "120"))
ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib", ".pt", ".pth", ".onnx"}
ALLOWED_DATASET_EXTENSIONS = {".csv", ".json", ".parquet"}

# JWT / Secret settings
SECRET_KEY = os.environ.get("WME_SECRET_KEY", "wme-super-secret-key-2026-ml-evaluator")
ALGORITHM = "HS256"
