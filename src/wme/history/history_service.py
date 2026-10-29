import json
from src.wme.core.database import get_db_connection
from src.wme.evaluator.engine import EvaluationEngine

class HistoryService:
    """
    Evaluation history retrieval and filtering with access control (WME-F-015, TC-History-01).
    """

    @staticmethod
    def list_history(
        user_id: str = None,
        role: str = "evaluator",
        status_filter: str = None,
        model_id: str = None,
        dataset_id: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[dict]:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT e.*, m.name as model_name, d.name as dataset_name, u.username,
                   r.metrics_json, r.execution_time_seconds
            FROM evaluations e
            LEFT JOIN models m ON e.model_id = m.id
            LEFT JOIN datasets d ON e.dataset_id = d.id
            LEFT JOIN users u ON e.user_id = u.id
            LEFT JOIN results r ON e.id = r.evaluation_id
            WHERE 1=1
        """
        params = []

        # If user is not admin or faculty, restrict to their own evaluations
        if role.lower() not in ["administrator", "faculty"]:
            query += " AND e.user_id = ?"
            params.append(user_id)

        if status_filter:
            query += " AND e.status = ?"
            params.append(status_filter)

        if model_id:
            query += " AND e.model_id = ?"
            params.append(model_id)

        if dataset_id:
            query += " AND e.dataset_id = ?"
            params.append(dataset_id)

        query += " ORDER BY e.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        records = []
        for r in rows:
            d = dict(r)
            d["feature_columns"] = json.loads(d["feature_columns_json"])
            d["selected_metrics"] = json.loads(d["selected_metrics_json"])
            if d.get("metrics_json"):
                d["metrics"] = json.loads(d["metrics_json"])
                del d["metrics_json"]
            else:
                d["metrics"] = None
            records.append(d)

        return records
