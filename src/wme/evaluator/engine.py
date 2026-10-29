import os
import time
import json
import uuid
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from src.wme.core.database import get_db_connection
from src.wme.core.audit import log_audit_event
from src.wme.core.sandbox import SafeExecutionSandbox
from src.wme.models.model_service import ModelService
from src.wme.datasets.dataset_service import DatasetService
from src.wme.evaluator.metrics import MetricCalculator
from src.wme.evaluator.visualizations import VisualizationGenerator

class EvaluationEngine:
    @staticmethod
    def create_evaluation(
        user_id: str,
        name: str,
        model_id: str,
        dataset_id: str,
        eval_type: str,
        target_column: str,
        feature_columns: list[str],
        selected_metrics: list[str] = None
    ) -> dict:
        """
        Configures a new evaluation job (WME-F-007, TC-Eval-01).
        Status is initially 'queued'.
        """
        # Verify model and dataset exist
        model = ModelService.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found.")

        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found.")

        # Check compatibility
        compat = DatasetService.check_compatibility(model_id, dataset_id, target_column)
        if not compat["compatible"]:
            raise ValueError(f"Model and dataset are not compatible: {'; '.join(compat['issues'])}")

        eval_id = f"eval-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO evaluations (
                id, user_id, name, model_id, dataset_id, eval_type,
                target_column, feature_columns_json, selected_metrics_json,
                status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
            """,
            (
                eval_id,
                user_id,
                name or f"Evaluation {eval_id}",
                model_id,
                dataset_id,
                eval_type,
                target_column,
                json.dumps(feature_columns),
                json.dumps(selected_metrics or []),
                now
            )
        )
        conn.commit()
        conn.close()

        log_audit_event("EVAL_CONFIGURED", f"Evaluation:{eval_id}", user_id=user_id, details={"model_id": model_id, "dataset_id": dataset_id})

        return {
            "id": eval_id,
            "name": name,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "eval_type": eval_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "status": "queued",
            "created_at": now
        }

    @staticmethod
    def run_evaluation(evaluation_id: str) -> dict:
        """
        Executes the evaluation run (WME-F-008, WME-F-009, TC-Eval-02).
        Transitions status: queued -> running -> completed (or failed).
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        eval_row = cursor.fetchone()

        if not eval_row:
            conn.close()
            raise ValueError(f"Evaluation {evaluation_id} not found.")

        # Set status to running
        cursor.execute("UPDATE evaluations SET status = 'running' WHERE id = ?", (evaluation_id,))
        conn.commit()
        conn.close()

        start_time = time.time()

        try:
            model_info = ModelService.get_model(eval_row["model_id"])
            dataset_info = DatasetService.get_dataset(eval_row["dataset_id"])
            feature_cols = json.loads(eval_row["feature_columns_json"])
            target_col = eval_row["target_column"]
            eval_type = eval_row["eval_type"].lower()

            # Load model artifact securely
            SafeExecutionSandbox.inspect_model_file(model_info["file_path"])
            try:
                model = joblib.load(model_info["file_path"])
            except Exception:
                with open(model_info["file_path"], "rb") as f:
                    model = pickle.load(f)

            # Load dataset
            df = DatasetService.load_dataframe(dataset_info["file_path"])

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset.")

            # Filter rows with missing target or features
            clean_df = df.dropna(subset=[target_col] + feature_cols)
            if clean_df.empty:
                raise ValueError("Dataset has no complete rows after dropping missing values.")

            X = clean_df[feature_cols].values
            y_true = clean_df[target_col].values

            # Execute model predictions inside sandboxed timeout
            def _predict_op():
                y_pred = model.predict(X)
                y_proba = None
                if eval_type == "classification" and hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X)
                    except Exception:
                        y_proba = None
                return y_pred, y_proba

            y_pred, y_proba = SafeExecutionSandbox.run_with_timeout(_predict_op, timeout_seconds=45)

            # Calculate metrics
            visualizations = {}
            if eval_type == "classification":
                metrics = MetricCalculator.calculate_classification_metrics(y_true, y_pred, y_proba)
                # Visualizations
                visualizations["confusion_matrix"] = VisualizationGenerator.generate_confusion_matrix_plot(
                    metrics["confusion_matrix"], metrics["labels"]
                )
                if y_proba is not None:
                    try:
                        visualizations["roc_curve"] = VisualizationGenerator.generate_roc_curve_plot(
                            y_true, y_proba, metrics["labels"]
                        )
                        visualizations["precision_recall_curve"] = VisualizationGenerator.generate_precision_recall_plot(
                            y_true, y_proba, metrics["labels"]
                        )
                    except Exception as e_vis:
                        print("Warning generating curves:", e_vis)
            else:
                metrics = MetricCalculator.calculate_regression_metrics(y_true, y_pred)
                reg_plots = VisualizationGenerator.generate_regression_plots(y_true, y_pred)
                visualizations.update(reg_plots)

            # Sample predictions for table view (first 50)
            sample_records = []
            max_samples = min(50, len(clean_df))
            for i in range(max_samples):
                record = {
                    "index": i + 1,
                    "actual": float(y_true[i]) if eval_type == "regression" else str(y_true[i]),
                    "predicted": float(y_pred[i]) if eval_type == "regression" else str(y_pred[i]),
                    "correct": bool(y_true[i] == y_pred[i]) if eval_type == "classification" else None,
                    "error": round(float(abs(y_true[i] - y_pred[i])), 4) if eval_type == "regression" else None
                }
                sample_records.append(record)

            elapsed_seconds = round(time.time() - start_time, 3)
            now = datetime.now(timezone.utc).isoformat()
            result_id = f"res-{uuid.uuid4().hex[:12]}"

            # Save results & update evaluation status
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO results (
                    id, evaluation_id, metrics_json, visualizations_json,
                    predictions_sample_json, execution_time_seconds, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    evaluation_id,
                    json.dumps(metrics),
                    json.dumps(visualizations),
                    json.dumps(sample_records),
                    elapsed_seconds,
                    now
                )
            )

            cursor.execute(
                """
                UPDATE evaluations
                SET status = 'completed', completed_at = ?, error_message = NULL
                WHERE id = ?
                """,
                (now, evaluation_id)
            )
            conn.commit()
            conn.close()

            log_audit_event("EVAL_COMPLETED", f"Evaluation:{evaluation_id}", user_id=eval_row["user_id"], details={"duration": elapsed_seconds})

            return {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "execution_time_seconds": elapsed_seconds,
                "metrics": metrics,
                "visualizations": visualizations,
                "sample_predictions": sample_records
            }

        except Exception as e:
            now = datetime.now(timezone.utc).isoformat()
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE evaluations
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (str(e), now, evaluation_id)
            )
            conn.commit()
            conn.close()

            log_audit_event("EVAL_FAILED", f"Evaluation:{evaluation_id}", user_id=eval_row["user_id"], details={"error": str(e)})
            raise e

    @staticmethod
    def get_evaluation(evaluation_id: str) -> dict:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        eval_data = dict(row)
        eval_data["feature_columns"] = json.loads(eval_data["feature_columns_json"])
        eval_data["selected_metrics"] = json.loads(eval_data["selected_metrics_json"])

        # Fetch result if available
        cursor.execute("SELECT * FROM results WHERE evaluation_id = ?", (evaluation_id,))
        res_row = cursor.fetchone()
        conn.close()

        if res_row:
            eval_data["result"] = {
                "id": res_row["id"],
                "metrics": json.loads(res_row["metrics_json"]),
                "visualizations": json.loads(res_row["visualizations_json"]),
                "predictions_sample": json.loads(res_row["predictions_sample_json"]),
                "execution_time_seconds": res_row["execution_time_seconds"],
                "created_at": res_row["created_at"]
            }
        else:
            eval_data["result"] = None

        return eval_data
