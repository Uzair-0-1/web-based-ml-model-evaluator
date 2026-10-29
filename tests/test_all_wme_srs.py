import os
import sys
import json
import io
import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wme.api.app import app
from src.wme.core.database import get_db_connection, init_db, verify_password
from src.wme.core.auth import register_user, authenticate_user
from src.wme.models.model_service import ModelService
from src.wme.datasets.dataset_service import DatasetService
from src.wme.evaluator.metrics import MetricCalculator
from src.wme.evaluator.visualizations import VisualizationGenerator
from src.wme.evaluator.engine import EvaluationEngine
from src.wme.comparison.comparison_service import ComparisonService
from src.wme.reports.report_service import ReportService
from src.wme.history.history_service import HistoryService

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()

# ==================== 1. AUTHENTICATION & ACCESS (TC-Auth-01, TC-Auth-02, TC-Sec-02, TC-Sec-03, TC-Sec-04) ====================
def test_user_registration_and_login():
    """TC-Auth-01: Registration, Login and Session generation."""
    uname = f"testuser_{os.urandom(4).hex()}"
    reg_res = client.post("/api/auth/register", json={
        "username": uname,
        "email": f"{uname}@example.com",
        "password": "Password@123",
        "role": "evaluator"
    })
    assert reg_res.status_code == 200
    assert reg_res.json()["username"] == uname

    # Login
    login_res = client.post("/api/auth/login", json={
        "username": uname,
        "password": "Password@123"
    })
    assert login_res.status_code == 200
    token = login_res.json()["token"]
    assert token.startswith("tok-")

    # Access /api/auth/me
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == uname

def test_rbac_access_control():
    """TC-Auth-02 & TC-Sec-02: Role-based access control blocking unauthorized requests."""
    # Authenticate as regular evaluator
    login_res = client.post("/api/auth/login", json={
        "username": "evaluator",
        "password": "Eval@123"
    })
    eval_token = login_res.json()["token"]

    # Try to access admin-only endpoint: /api/admin/users
    blocked_res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {eval_token}"})
    assert blocked_res.status_code == 403

    # Authenticate as admin
    admin_login = client.post("/api/auth/login", json={
        "username": "admin",
        "password": "Admin@123"
    })
    admin_token = admin_login.json()["token"]

    # Admin access allowed
    allowed_res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert allowed_res.status_code == 200
    assert len(allowed_res.json()) >= 3

def test_credential_storage_security():
    """TC-Sec-04: Credential storage audit - ensure passwords are salted hashes, never plaintext."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, salt FROM users WHERE username = 'admin'")
    row = cursor.fetchone()
    conn.close()

    assert row["password_hash"] != "Admin@123"
    assert len(row["password_hash"]) == 64  # SHA256 hex string
    assert len(row["salt"]) == 32           # 16-byte salt hex string
    assert verify_password("Admin@123", row["password_hash"], row["salt"]) is True

# ==================== 2. MODEL MANAGEMENT (TC-Model-01, TC-Model-02) ====================
def test_model_upload_and_validation():
    """TC-Model-01 & TC-Model-02: Model upload, inspection, and invalid rejection."""
    login_res = client.post("/api/auth/login", json={"username": "evaluator", "password": "Eval@123"})
    token = login_res.json()["token"]

    # 1. Train a valid logistic regression model
    iris = load_iris()
    clf = LogisticRegression(max_iter=200)
    clf.fit(iris.data, iris.target)

    model_buf = io.BytesIO()
    joblib.dump(clf, model_buf)
    model_buf.seek(0)

    upload_res = client.post(
        "/api/models/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"name": "Test Logistic Model"},
        files={"file": ("model.pkl", model_buf.read(), "application/octet-stream")}
    )
    assert upload_res.status_code == 200
    model_data = upload_res.json()
    assert model_data["model_type"] == "classifier"
    assert model_data["metadata"]["n_features"] == 4

    # 2. Reject corrupted model file (TC-Model-02)
    bad_buf = io.BytesIO(b"Not a valid pickle stream corrupt bytes")
    bad_res = client.post(
        "/api/models/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"name": "Corrupt Model"},
        files={"file": ("bad.pkl", bad_buf.read(), "application/octet-stream")}
    )
    assert bad_res.status_code == 400
    assert "Corrupted or unreadable" in bad_res.json()["detail"]

# ==================== 3. DATASET MANAGEMENT (TC-Data-01, TC-Data-02) ====================
def test_dataset_upload_and_compatibility():
    """TC-Data-01 & TC-Data-02: Dataset upload, schema validation, and compatibility checking."""
    login_res = client.post("/api/auth/login", json={"username": "evaluator", "password": "Eval@123"})
    token = login_res.json()["token"]

    # 1. Upload valid CSV dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    upload_res = client.post(
        "/api/datasets/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={"name": "Iris Test CSV"},
        files={"file": ("iris.csv", csv_bytes, "text/csv")}
    )
    assert upload_res.status_code == 200
    ds_data = upload_res.json()
    assert ds_data["row_count"] == 150
    assert ds_data["column_count"] == 5

    # 2. Check compatibility with model
    models_res = client.get("/api/models", headers={"Authorization": f"Bearer {token}"})
    m_id = models_res.json()[0]["id"]
    compat_res = client.post(
        f"/api/datasets/{ds_data['id']}/compatibility",
        headers={"Authorization": f"Bearer {token}"},
        json={"model_id": m_id, "target_column": "target"}
    )
    assert compat_res.status_code == 200
    compat_data = compat_res.json()
    assert compat_data["compatible"] is True

# ==================== 4. METRICS & VISUALIZATION (TC-Metric-01, TC-Metric-02, TC-Metric-03) ====================
def test_metric_calculations_match_reference():
    """TC-Metric-01 & TC-Metric-02: Metric calculations match trusted scikit-learn references."""
    # Classification reference check
    y_true_cls = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_cls = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])

    calc_metrics = MetricCalculator.calculate_classification_metrics(y_true_cls, y_pred_cls)
    ref_acc = accuracy_score(y_true_cls, y_pred_cls)
    assert calc_metrics["accuracy"] == round(ref_acc, 4)

    # Regression reference check
    y_true_reg = np.array([10.5, 20.0, 30.5, 40.0])
    y_pred_reg = np.array([11.0, 19.0, 31.0, 39.0])
    reg_metrics = MetricCalculator.calculate_regression_metrics(y_true_reg, y_pred_reg)
    ref_mse = mean_squared_error(y_true_reg, y_pred_reg)
    assert reg_metrics["mse"] == round(ref_mse, 4)
    assert reg_metrics["rmse"] == round(np.sqrt(ref_mse), 4)

def test_visualizations_generated():
    """TC-Metric-03: Visualizations generation produces valid Base64 PNG images."""
    cm = [[10, 2], [1, 15]]
    labels = ["Class 0", "Class 1"]
    b64_cm = VisualizationGenerator.generate_confusion_matrix_plot(cm, labels)
    assert b64_cm.startswith("data:image/png;base64,")

    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]])
    b64_roc = VisualizationGenerator.generate_roc_curve_plot(y_true, y_proba, labels)
    assert b64_roc.startswith("data:image/png;base64,")

# ==================== 5. EVALUATION EXECUTION, REPORT & COMPARISON (TC-Eval-01, TC-Eval-02, TC-Eval-03, TC-Result-01, TC-Report-01, TC-History-01) ====================
def test_full_evaluation_workflow_and_reporting():
    """TC-Eval-01, TC-Eval-02, TC-Report-01, TC-History-01."""
    # Seed demo models
    admin_login = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@123"})
    token = admin_login.json()["token"]

    seed_res = client.post("/api/demo/seed-sample-data", headers={"Authorization": f"Bearer {token}"})
    assert seed_res.status_code == 200
    m_id1 = seed_res.json()["classification_models"][0]
    m_id2 = seed_res.json()["classification_models"][1]
    ds_id = seed_res.json()["classification_dataset"]

    # 1. Configure Evaluation 1
    eval_res1 = client.post(
        "/api/evaluations",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Iris Model 1 Test",
            "model_id": m_id1,
            "dataset_id": ds_id,
            "eval_type": "classification",
            "target_column": "target",
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }
    )
    assert eval_res1.status_code == 200
    eid1 = eval_res1.json()["id"]

    # 2. Execute Evaluation 1
    run_res1 = client.post(f"/api/evaluations/{eid1}/run", headers={"Authorization": f"Bearer {token}"})
    assert run_res1.status_code == 200
    assert run_res1.json()["status"] == "completed"
    assert "accuracy" in run_res1.json()["metrics"]

    # 3. Configure and execute Evaluation 2 for Comparison
    eval_res2 = client.post(
        "/api/evaluations",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Iris Model 2 Test",
            "model_id": m_id2,
            "dataset_id": ds_id,
            "eval_type": "classification",
            "target_column": "target",
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }
    )
    eid2 = eval_res2.json()["id"]
    client.post(f"/api/evaluations/{eid2}/run", headers={"Authorization": f"Bearer {token}"})

    # 4. Multi-model comparison (TC-Result-01)
    comp_res = client.post(
        "/api/evaluations/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"evaluation_ids": [eid1, eid2]}
    )
    assert comp_res.status_code == 200
    assert comp_res.json()["compared_count"] == 2
    assert "accuracy" in comp_res.json()["winners"]

    # 5. Report Downloads (TC-Report-01)
    html_report = client.get(f"/api/reports/{eid1}/html", headers={"Authorization": f"Bearer {token}"})
    assert html_report.status_code == 200
    assert "Web Based ML Model Evaluator" in html_report.text

    md_report = client.get(f"/api/reports/{eid1}/markdown", headers={"Authorization": f"Bearer {token}"})
    assert md_report.status_code == 200
    assert "Performance Metrics" in md_report.text

    # 6. History Retrieval (TC-History-01)
    hist_res = client.get("/api/history", headers={"Authorization": f"Bearer {token}"})
    assert hist_res.status_code == 200
    assert len(hist_res.json()) >= 2
