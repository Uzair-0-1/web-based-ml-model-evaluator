
import numpy as np
import pytest

from src.wme.evaluator.metrics import MetricCalculator


def test_classification_metrics_calculate_expected_values():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = MetricCalculator.calculate_classification_metrics(
        y_true, y_pred
    )

    assert result["type"] == "classification"
    assert result["accuracy"] == 0.75
    assert result["sample_count"] == 4
    assert result["labels"] == ["0", "1"]
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["per_class"]["0"]["support"] == 2
    assert result["per_class"]["1"]["support"] == 2


def test_classification_metrics_handle_string_labels():
    y_true = np.array(["cat", "dog", "cat", "dog"])
    y_pred = np.array(["cat", "dog", "dog", "dog"])

    result = MetricCalculator.calculate_classification_metrics(
        y_true, y_pred
    )

    assert result["labels"] == ["cat", "dog"]
    assert result["sample_count"] == 4
    assert result["accuracy"] == 0.75
    assert "cat" in result["per_class"]
    assert "dog" in result["per_class"]


def test_classification_metrics_reject_empty_inputs():
    with pytest.raises(ValueError):
        MetricCalculator.calculate_classification_metrics(
            np.array([]), np.array([])
        )


def test_classification_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        MetricCalculator.calculate_classification_metrics(
            np.array([0, 1, 0]),
            np.array([0, 1])
        )


def test_regression_metrics_calculate_expected_values():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    result = MetricCalculator.calculate_regression_metrics(
        y_true, y_pred
    )

    assert result["type"] == "regression"
    assert result["mae"] == 0.5
    assert result["mse"] == 0.375
    assert result["rmse"] == round(np.sqrt(0.375), 4)
    assert result["sample_count"] == 4
    assert result["mean_residual"] == -0.25


def test_regression_metrics_reject_empty_inputs():
    with pytest.raises(ValueError):
        MetricCalculator.calculate_regression_metrics(
            np.array([]), np.array([])
        )


def test_regression_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        MetricCalculator.calculate_regression_metrics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0])
        )
