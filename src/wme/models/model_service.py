import os
import uuid
import json
import joblib
import pickle
from pathlib import Path
from datetime import datetime, timezone
from src.wme.core.config import MODELS_DIR, ALLOWED_MODEL_EXTENSIONS, MAX_FILE_SIZE_MB
from src.wme.core.database import get_db_connection
from src.wme.core.audit import log_audit_event
from src.wme.core.sandbox import SafeExecutionSandbox, SecurityException

class ModelService:
    @staticmethod
    def validate_and_inspect_model(file_path: str) -> dict:
        """
        Validates the model file:
        1. Checks format / extension
        2. Inspects for malicious patterns
        3. Attempts loading via joblib / pickle
        4. Validates that it exposes a callable predict method
        5. Extracts metadata: model_type (classification/regression), framework, expected feature count/names.
        (WME-F-004, TC-Model-02)
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in ALLOWED_MODEL_EXTENSIONS:
            raise ValueError(f"Unsupported model extension '{ext}'. Allowed extensions: {', '.join(ALLOWED_MODEL_EXTENSIONS)}")

        # Sandbox scan
        SafeExecutionSandbox.inspect_model_file(str(file_path))

        # Attempt to load model
        try:
            model = joblib.load(file_path)
        except Exception as e_joblib:
            try:
                with open(file_path, "rb") as f:
                    model = pickle.load(f)
            except Exception as e_pickle:
                raise ValueError(f"Corrupted or unreadable model file: {str(e_joblib)} | {str(e_pickle)}")

        # Verify it has predict method
        if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
            raise ValueError("Invalid model artifact: object lacks a callable 'predict' method.")

        # Determine model type (classification vs regression)
        model_type = "unknown"
        if hasattr(model, "_estimator_type"):
            model_type = model._estimator_type # 'classifier' or 'regressor'
        elif hasattr(model, "predict_proba"):
            model_type = "classifier"
        elif hasattr(model, "classes_"):
            model_type = "classifier"
        else:
            # Fallback heuristic
            cls_name = model.__class__.__name__.lower()
            if "classifier" in cls_name or "svc" in cls_name or "forestclassifier" in cls_name or "logistic" in cls_name:
                model_type = "classifier"
            elif "regressor" in cls_name or "svr" in cls_name or "linear" in cls_name or "ridge" in cls_name:
                model_type = "regressor"

        # Extract features metadata if present
        feature_names = []
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)

        n_features = getattr(model, "n_features_in_", len(feature_names) if feature_names else None)

        classes = []
        if hasattr(model, "classes_"):
            classes = [str(c) for c in model.classes_]

        framework = model.__class__.__module__.split(".")[0]

        return {
            "valid": True,
            "class_name": model.__class__.__name__,
            "framework": framework,
            "model_type": model_type,
            "n_features": n_features,
            "feature_names": feature_names,
            "classes": classes,
            "has_predict_proba": hasattr(model, "predict_proba")
        }

    @staticmethod
    def save_model_artifact(user_id: str, name: str, original_filename: str, file_bytes: bytes) -> dict:
        """
        Saves model to storage and registers it in the database (WME-F-003, TC-Model-01).
        """
        # Size validation
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB).")

        model_id = f"mod-{uuid.uuid4().hex[:12]}"
        ext = Path(original_filename).suffix.lower()
        saved_filename = f"{model_id}{ext}"
        destination_path = MODELS_DIR / saved_filename

        with open(destination_path, "wb") as f:
            f.write(file_bytes)

        # Validate
        try:
            metadata = ModelService.validate_and_inspect_model(str(destination_path))
        except Exception as e:
            if destination_path.exists():
                destination_path.unlink()
            raise e

        now = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO models (id, user_id, name, filename, file_path, file_size, framework, model_type, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                user_id,
                name or original_filename,
                original_filename,
                str(destination_path),
                len(file_bytes),
                metadata["framework"],
                metadata["model_type"],
                json.dumps(metadata),
                now
            )
        )
        conn.commit()
        conn.close()

        log_audit_event("MODEL_UPLOAD", f"Model:{model_id}", user_id=user_id, details={"name": name, "model_type": metadata["model_type"]})

        return {
            "id": model_id,
            "name": name or original_filename,
            "filename": original_filename,
            "model_type": metadata["model_type"],
            "framework": metadata["framework"],
            "metadata": metadata,
            "created_at": now
        }

    @staticmethod
    def get_model(model_id: str) -> dict:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        res["metadata"] = json.loads(res["metadata_json"])
        return res

    @staticmethod
    def list_models(user_id: str = None) -> list[dict]:
        conn = get_db_connection()
        cursor = conn.cursor()
        if user_id:
            cursor.execute("SELECT * FROM models WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        else:
            cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata_json"])
            results.append(d)
        return results

    @staticmethod
    def delete_model(model_id: str, user_id: str = None, is_admin: bool = False) -> bool:
        model = ModelService.get_model(model_id)
        if not model:
            return False
        if not is_admin and user_id and model["user_id"] != user_id:
            raise PermissionError("Unauthorized to delete this model.")

        file_path = Path(model["file_path"])
        if file_path.exists():
            file_path.unlink()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()

        log_audit_event("MODEL_DELETE", f"Model:{model_id}", user_id=user_id)
        return True
