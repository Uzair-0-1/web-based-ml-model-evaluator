import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error
)

class MetricCalculator:
    """
    Computes standard evaluation metrics for Classification and Regression models (WME-F-010, WME-F-011).
    """

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
        """
        Calculates classification metrics: accuracy, precision, recall, f1-score (macro/micro/weighted),
        per-class metrics, and confusion matrix data (WME-F-010, TC-Metric-01).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        unique_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
        labels_str = [str(l) for l in unique_labels]

        acc = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1 across averages
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        prec_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        rec_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per class scores
        prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        per_class_metrics = {}
        for idx, lbl in enumerate(labels_str):
            per_class_metrics[lbl] = {
                "precision": round(float(prec_per_class[idx]), 4),
                "recall": round(float(rec_per_class[idx]), 4),
                "f1_score": round(float(f1_per_class[idx]), 4),
                "support": int(np.sum(y_true == unique_labels[idx]))
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        cm_list = cm.tolist()

        return {
            "type": "classification",
            "accuracy": round(acc, 4),
            "precision": round(prec_macro, 4),
            "precision_micro": round(prec_micro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall": round(rec_macro, 4),
            "recall_micro": round(rec_micro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_score": round(f1_macro, 4),
            "f1_micro": round(f1_micro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "labels": labels_str,
            "per_class": per_class_metrics,
            "confusion_matrix": cm_list,
            "sample_count": int(len(y_true))
        }

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculates regression metrics: MAE, MSE, RMSE, R2-score, Max Error, Explained Variance (WME-F-011, TC-Metric-02).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        m_err = float(max_error(y_true, y_pred))
        exp_var = float(explained_variance_score(y_true, y_pred))

        residuals = (y_true - y_pred).tolist()

        return {
            "type": "regression",
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "max_error": round(m_err, 4),
            "explained_variance": round(exp_var, 4),
            "sample_count": int(len(y_true)),
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4)
        }
