from src.wme.evaluator.engine import EvaluationEngine
from src.wme.models.model_service import ModelService
from src.wme.datasets.dataset_service import DatasetService

class ComparisonService:
    """
    Compares two or more completed evaluation runs on common metrics (WME-F-013, TC-Result-01).
    """

    @staticmethod
    def compare_evaluations(evaluation_ids: list[str]) -> dict:
        if len(evaluation_ids) < 2:
            raise ValueError("At least 2 evaluation IDs are required for comparison.")

        evaluations_data = []
        for eid in evaluation_ids:
            eval_info = EvaluationEngine.get_evaluation(eid)
            if not eval_info:
                raise ValueError(f"Evaluation {eid} not found.")
            if eval_info["status"] != "completed" or not eval_info.get("result"):
                raise ValueError(f"Evaluation {eid} has not completed successfully.")
            
            model = ModelService.get_model(eval_info["model_id"])
            dataset = DatasetService.get_dataset(eval_info["dataset_id"])
            evaluations_data.append({
                "evaluation_id": eid,
                "evaluation_name": eval_info["name"],
                "model_name": model["name"] if model else "Unknown",
                "model_type": eval_info["eval_type"],
                "framework": model["framework"] if model else "Unknown",
                "dataset_name": dataset["name"] if dataset else "Unknown",
                "metrics": eval_info["result"]["metrics"],
                "execution_time_seconds": eval_info["result"]["execution_time_seconds"]
            })

        # Ensure all models are of the same type (classification or regression)
        first_type = evaluations_data[0]["model_type"].lower()
        if not all(e["model_type"].lower() == first_type for e in evaluations_data):
            raise ValueError("All models selected for comparison must be of the same type (Classification or Regression).")

        # Determine metric keys to compare
        if first_type == "classification":
            metric_keys = ["accuracy", "precision", "recall", "f1_score"]
            higher_is_better = {k: True for k in metric_keys}
        else:
            metric_keys = ["mae", "mse", "rmse", "r2_score"]
            higher_is_better = {"mae": False, "mse": False, "rmse": False, "r2_score": True}

        # Build comparison summary table
        comparison_table = []
        for e in evaluations_data:
            row = {
                "evaluation_id": e["evaluation_id"],
                "model_name": e["model_name"],
                "framework": e["framework"],
                "execution_time": e["execution_time_seconds"]
            }
            for m in metric_keys:
                row[m] = e["metrics"].get(m)
            comparison_table.append(row)

        # Determine winners per metric
        winners = {}
        for m in metric_keys:
            valid_vals = [(e["model_name"], e["metrics"].get(m)) for e in evaluations_data if e["metrics"].get(m) is not None]
            if valid_vals:
                if higher_is_better[m]:
                    best = max(valid_vals, key=lambda x: x[1])
                else:
                    best = min(valid_vals, key=lambda x: x[1])
                winners[m] = {"model": best[0], "value": best[1]}

        return {
            "type": first_type,
            "compared_count": len(evaluations_data),
            "metric_keys": metric_keys,
            "evaluations": evaluations_data,
            "comparison_table": comparison_table,
            "winners": winners
        }
