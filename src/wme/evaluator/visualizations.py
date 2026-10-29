import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless web server
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

class VisualizationGenerator:
    """
    Generates plots for classification and regression models (WME-F-012, TC-Metric-03).
    Returns base64 PNG images and chart data structures.
    """

    @staticmethod
    def _fig_to_base64(fig) -> str:
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    @classmethod
    def generate_confusion_matrix_plot(cls, cm: list[list[int]], labels: list[str]) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_array = np.array(cm)
        cax = ax.matshow(cm_array, cmap=plt.cm.Blues, alpha=0.85)

        for i in range(cm_array.shape[0]):
            for j in range(cm_array.shape[1]):
                ax.text(x=j, y=i, s=str(cm_array[i, j]), va="center", ha="center", size="medium", weight="bold")

        fig.colorbar(cax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="left")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label", weight="bold")
        ax.set_ylabel("True Label", weight="bold")
        ax.set_title("Confusion Matrix", weight="bold", pad=15)
        return cls._fig_to_base64(fig)

    @classmethod
    def generate_roc_curve_plot(cls, y_true: np.ndarray, y_proba: np.ndarray, labels: list[str]) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        n_classes = len(labels)

        if n_classes == 2:
            # Binary classification
            # If 2 columns in y_proba, take column 1
            scores = y_proba[:, 1] if (y_proba.ndim > 1 and y_proba.shape[1] > 1) else y_proba.ravel()
            # Binarize y_true
            classes_list = sorted(list(set(y_true)))
            y_bin = (y_true == classes_list[-1]).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        else:
            # Multiclass OvR
            try:
                classes_list = sorted(list(set(y_true)))
                y_bin = label_binarize(y_true, classes=classes_list)
                for i in range(min(n_classes, y_proba.shape[1])):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=1.5, label=f"Class {labels[i]} (AUC = {roc_auc:.2f})")
            except Exception:
                ax.text(0.5, 0.5, "ROC calculation unavailable", ha="center", va="center")

        ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", weight="bold")
        ax.set_ylabel("True Positive Rate", weight="bold")
        ax.set_title("Receiver Operating Characteristic (ROC)", weight="bold")
        ax.legend(loc="lower right", fontsize=8)
        return cls._fig_to_base64(fig)

    @classmethod
    def generate_precision_recall_plot(cls, y_true: np.ndarray, y_proba: np.ndarray, labels: list[str]) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        n_classes = len(labels)

        if n_classes == 2:
            scores = y_proba[:, 1] if (y_proba.ndim > 1 and y_proba.shape[1] > 1) else y_proba.ravel()
            classes_list = sorted(list(set(y_true)))
            y_bin = (y_true == classes_list[-1]).astype(int)
            precision, recall, _ = precision_recall_curve(y_bin, scores)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color="#16a34a", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
        else:
            try:
                classes_list = sorted(list(set(y_true)))
                y_bin = label_binarize(y_true, classes=classes_list)
                for i in range(min(n_classes, y_proba.shape[1])):
                    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                    pr_auc = auc(recall, precision)
                    ax.plot(recall, precision, lw=1.5, label=f"Class {labels[i]} (AUC = {pr_auc:.2f})")
            except Exception:
                ax.text(0.5, 0.5, "PR curve unavailable", ha="center", va="center")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", weight="bold")
        ax.set_ylabel("Precision", weight="bold")
        ax.set_title("Precision-Recall Curve", weight="bold")
        ax.legend(loc="lower left", fontsize=8)
        return cls._fig_to_base64(fig)

    @classmethod
    def generate_regression_plots(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, str]:
        """Generates Actual vs Predicted and Residual plots."""
        # 1. Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y_true, y_pred, alpha=0.6, color="#2563eb", edgecolors="none")
        min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
        ax1.set_xlabel("Actual Values", weight="bold")
        ax1.set_ylabel("Predicted Values", weight="bold")
        ax1.set_title("Actual vs Predicted Values", weight="bold")
        ax1.legend()
        actual_vs_pred_b64 = cls._fig_to_base64(fig1)

        # 2. Residuals Plot
        residuals = y_true - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_pred, residuals, alpha=0.6, color="#059669")
        ax2.axhline(0, color="red", linestyle="--", lw=2)
        ax2.set_xlabel("Predicted Values", weight="bold")
        ax2.set_ylabel("Residuals (Actual - Predicted)", weight="bold")
        ax2.set_title("Residual Plot", weight="bold")
        residuals_b64 = cls._fig_to_base64(fig2)

        return {
            "actual_vs_predicted": actual_vs_pred_b64,
            "residuals": residuals_b64
        }
