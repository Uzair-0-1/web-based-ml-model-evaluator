import os
import uuid
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from src.wme.core.config import DATASETS_DIR, ALLOWED_DATASET_EXTENSIONS, MAX_FILE_SIZE_MB
from src.wme.core.database import get_db_connection
from src.wme.core.audit import log_audit_event
from src.wme.models.model_service import ModelService

class DatasetService:
    @staticmethod
    def load_dataframe(file_path: str) -> pd.DataFrame:
        """Loads a supported dataset file into a pandas DataFrame."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported dataset format '{ext}'. Supported: {', '.join(ALLOWED_DATASET_EXTENSIONS)}")
        return df

    @staticmethod
    def inspect_dataset(file_path: str) -> dict:
        """
        Validates dataset structure and extracts schema, null counts, sample preview (WME-F-006).
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in ALLOWED_DATASET_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_DATASET_EXTENSIONS)}")

        try:
            df = DatasetService.load_dataframe(file_path)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid dataset file: {str(e)}")

        if df.empty:
            raise ValueError("Dataset is empty. At least one row of data is required.")

        columns = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing = int(df[col].isnull().sum())
            unique_vals = int(df[col].nunique())
            columns.append({
                "name": str(col),
                "dtype": dtype_str,
                "missing_count": missing,
                "unique_count": unique_vals,
                "is_numeric": pd.api.types.is_numeric_dtype(df[col])
            })

        preview_records = df.head(10).fillna("").to_dict(orient="records")

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": columns,
            "preview": preview_records
        }

    @staticmethod
    def save_dataset_artifact(user_id: str, name: str, original_filename: str, file_bytes: bytes) -> dict:
        """
        Saves dataset to storage and registers it in the database (WME-F-005, TC-Data-01).
        """
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"Dataset size ({size_mb:.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB).")

        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        ext = Path(original_filename).suffix.lower()
        saved_filename = f"{dataset_id}{ext}"
        destination_path = DATASETS_DIR / saved_filename

        with open(destination_path, "wb") as f:
            f.write(file_bytes)

        # Validate
        try:
            meta = DatasetService.inspect_dataset(str(destination_path))
        except Exception as e:
            if destination_path.exists():
                destination_path.unlink()
            raise e

        now = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO datasets (id, user_id, name, filename, file_path, file_size, row_count, column_count, columns_json, preview_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                user_id,
                name or original_filename,
                original_filename,
                str(destination_path),
                len(file_bytes),
                meta["row_count"],
                meta["column_count"],
                json.dumps(meta["columns"]),
                json.dumps(meta["preview"]),
                now
            )
        )
        conn.commit()
        conn.close()

        log_audit_event("DATASET_UPLOAD", f"Dataset:{dataset_id}", user_id=user_id, details={"name": name, "rows": meta["row_count"], "columns": meta["column_count"]})

        return {
            "id": dataset_id,
            "name": name or original_filename,
            "filename": original_filename,
            "row_count": meta["row_count"],
            "column_count": meta["column_count"],
            "columns": meta["columns"],
            "preview": meta["preview"],
            "created_at": now
        }

    @staticmethod
    def check_compatibility(model_id: str, dataset_id: str, target_column: str = None) -> dict:
        """
        Verifies compatibility between dataset and selected model (WME-F-006, TC-Data-02).
        Checks:
        - Target column presence
        - Feature column presence
        - Feature count compatibility
        - Feature name matching
        """
        model = ModelService.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found.")

        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found.")

        ds_cols = [c["name"] for c in dataset["columns"]]
        model_meta = model["metadata"]
        expected_feature_names = model_meta.get("feature_names") or []
        expected_n_features = model_meta.get("n_features")

        issues = []
        warnings = []

        # If target column specified, verify it exists
        if target_column:
            if target_column not in ds_cols:
                issues.append(f"Target column '{target_column}' is missing from dataset.")

        # Suggest candidate target column if not supplied
        candidate_target = target_column
        if not candidate_target:
            possible_names = ["target", "label", "class", "y", "outcome", "survived", "price"]
            for p in possible_names:
                for col in ds_cols:
                    if col.lower() == p:
                        candidate_target = col
                        break
                if candidate_target:
                    break
            if not candidate_target and ds_cols:
                candidate_target = ds_cols[-1] # Default to last column

        # Available feature columns excluding target
        feature_candidates = [c for c in ds_cols if c != candidate_target]

        # Check feature count
        if expected_n_features:
            if expected_feature_names:
                missing_features = [f for f in expected_feature_names if f not in ds_cols]
                if missing_features:
                    issues.append(f"Model expects specific features missing in dataset: {missing_features}")
            elif len(feature_candidates) < expected_n_features:
                issues.append(f"Model expects {expected_n_features} features, but dataset has only {len(feature_candidates)} non-target columns.")

        is_compatible = len(issues) == 0

        return {
            "compatible": is_compatible,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "model_type": model["model_type"],
            "expected_n_features": expected_n_features,
            "expected_feature_names": expected_feature_names,
            "dataset_columns": ds_cols,
            "suggested_target_column": candidate_target,
            "suggested_features": expected_feature_names if expected_feature_names and all(f in ds_cols for f in expected_feature_names) else feature_candidates,
            "issues": issues,
            "warnings": warnings
        }

    @staticmethod
    def get_dataset(dataset_id: str) -> dict:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        res["columns"] = json.loads(res["columns_json"])
        res["preview"] = json.loads(res["preview_json"])
        return res

    @staticmethod
    def list_datasets(user_id: str = None) -> list[dict]:
        conn = get_db_connection()
        cursor = conn.cursor()
        if user_id:
            cursor.execute("SELECT * FROM datasets WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        else:
            cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["columns"] = json.loads(d["columns_json"])
            d["preview"] = json.loads(d["preview_json"])
            results.append(d)
        return results

    @staticmethod
    def delete_dataset(dataset_id: str, user_id: str = None, is_admin: bool = False) -> bool:
        ds = DatasetService.get_dataset(dataset_id)
        if not ds:
            return False
        if not is_admin and user_id and ds["user_id"] != user_id:
            raise PermissionError("Unauthorized to delete this dataset.")

        file_path = Path(ds["file_path"])
        if file_path.exists():
            file_path.unlink()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.commit()
        conn.close()

        log_audit_event("DATASET_DELETE", f"Dataset:{dataset_id}", user_id=user_id)
        return True
