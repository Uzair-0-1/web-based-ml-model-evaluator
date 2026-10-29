import os
import psutil
import json
from pathlib import Path
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from src.wme.core.database import init_db, get_db_connection
from src.wme.core.auth import register_user, authenticate_user, logout_user, get_current_user
from src.wme.core.rbac import require_admin, require_evaluator_or_admin, require_any_authenticated, require_faculty_or_admin
from src.wme.core.audit import get_audit_logs, log_audit_event
from src.wme.models.model_service import ModelService
from src.wme.datasets.dataset_service import DatasetService
from src.wme.evaluator.engine import EvaluationEngine
from src.wme.comparison.comparison_service import ComparisonService
from src.wme.reports.report_service import ReportService
from src.wme.history.history_service import HistoryService

# Initialize database on startup
init_db()

app = FastAPI(
    title="Web Based ML Model Evaluator (WME)",
    description="Browser-based platform for evaluating trained machine-learning models (Problem Statement No. 25)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Auth Schemas & Endpoints (WME-F-001, WME-SR-002, WME-SR-003) -----------------
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "evaluator"

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/register")
def api_register(req: RegisterRequest):
    return register_user(req.username, req.email, req.password, req.role)

@app.post("/api/auth/login")
def api_login(req: LoginRequest):
    return authenticate_user(req.username, req.password)

@app.post("/api/auth/logout")
def api_logout(user: dict = Depends(get_current_user)):
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me")
def api_me(user: dict = Depends(get_current_user)):
    return user

# ----------------- Model Management Endpoints (WME-F-003, WME-F-004) -----------------
@app.post("/api/models/upload")
async def api_upload_model(
    name: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(require_evaluator_or_admin)
):
    contents = await file.read()
    try:
        saved = ModelService.save_model_artifact(user["id"], name, file.filename, contents)
        return saved
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/models")
def api_list_models(user: dict = Depends(require_any_authenticated)):
    # Admin & Faculty can view all models; evaluator sees their own or all public
    uid = None if user["role"].lower() in ["administrator", "faculty"] else user["id"]
    return ModelService.list_models(uid)

@app.get("/api/models/{model_id}")
def api_get_model(model_id: str, user: dict = Depends(require_any_authenticated)):
    model = ModelService.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.delete("/api/models/{model_id}")
def api_delete_model(model_id: str, user: dict = Depends(require_evaluator_or_admin)):
    is_admin = user["role"].lower() == "administrator"
    try:
        deleted = ModelService.delete_model(model_id, user_id=user["id"], is_admin=is_admin)
        if not deleted:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"message": "Model deleted successfully"}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

# ----------------- Dataset Management Endpoints (WME-F-005, WME-F-006) -----------------
@app.post("/api/datasets/upload")
async def api_upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(require_evaluator_or_admin)
):
    contents = await file.read()
    try:
        saved = DatasetService.save_dataset_artifact(user["id"], name, file.filename, contents)
        return saved
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/datasets")
def api_list_datasets(user: dict = Depends(require_any_authenticated)):
    uid = None if user["role"].lower() in ["administrator", "faculty"] else user["id"]
    return DatasetService.list_datasets(uid)

@app.get("/api/datasets/{dataset_id}")
def api_get_dataset(dataset_id: str, user: dict = Depends(require_any_authenticated)):
    ds = DatasetService.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

@app.delete("/api/datasets/{dataset_id}")
def api_delete_dataset(dataset_id: str, user: dict = Depends(require_evaluator_or_admin)):
    is_admin = user["role"].lower() == "administrator"
    try:
        deleted = DatasetService.delete_dataset(dataset_id, user_id=user["id"], is_admin=is_admin)
        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": "Dataset deleted successfully"}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

class CompatibilityCheckRequest(BaseModel):
    model_id: str
    target_column: Optional[str] = None

@app.post("/api/datasets/{dataset_id}/compatibility")
def api_check_compatibility(dataset_id: str, req: CompatibilityCheckRequest, user: dict = Depends(require_any_authenticated)):
    try:
        return DatasetService.check_compatibility(req.model_id, dataset_id, req.target_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Evaluation Configuration & Execution (WME-F-007, WME-F-008, WME-F-009) -----------------
class CreateEvaluationRequest(BaseModel):
    name: str
    model_id: str
    dataset_id: str
    eval_type: str
    target_column: str
    feature_columns: List[str]
    selected_metrics: Optional[List[str]] = []

@app.post("/api/evaluations")
def api_create_evaluation(req: CreateEvaluationRequest, user: dict = Depends(require_evaluator_or_admin)):
    try:
        created = EvaluationEngine.create_evaluation(
            user_id=user["id"],
            name=req.name,
            model_id=req.model_id,
            dataset_id=req.dataset_id,
            eval_type=req.eval_type,
            target_column=req.target_column,
            feature_columns=req.feature_columns,
            selected_metrics=req.selected_metrics
        )
        return created
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/evaluations/{evaluation_id}/run")
def api_run_evaluation(evaluation_id: str, user: dict = Depends(require_evaluator_or_admin)):
    try:
        res = EvaluationEngine.run_evaluation(evaluation_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/evaluations/{evaluation_id}")
def api_get_evaluation(evaluation_id: str, user: dict = Depends(require_any_authenticated)):
    data = EvaluationEngine.get_evaluation(evaluation_id)
    if not data:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return data

# ----------------- Model Comparison (WME-F-013, TC-Result-01) -----------------
class ComparisonRequest(BaseModel):
    evaluation_ids: List[str]

@app.post("/api/evaluations/compare")
def api_compare_models(req: ComparisonRequest, user: dict = Depends(require_any_authenticated)):
    try:
        return ComparisonService.compare_evaluations(req.evaluation_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Reports & Download (WME-F-014, TC-Report-01) -----------------
@app.get("/api/reports/{evaluation_id}/html")
def api_download_html_report(evaluation_id: str, user: dict = Depends(require_any_authenticated)):
    try:
        html = ReportService.generate_html_report(evaluation_id)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/reports/{evaluation_id}/markdown")
def api_download_markdown_report(evaluation_id: str, user: dict = Depends(require_any_authenticated)):
    try:
        md = ReportService.generate_markdown_report(evaluation_id)
        return PlainTextResponse(content=md, media_type="text/markdown")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/reports/{evaluation_id}/json")
def api_download_json_report(evaluation_id: str, user: dict = Depends(require_any_authenticated)):
    try:
        data = ReportService.get_full_report_data(evaluation_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# ----------------- Evaluation History (WME-F-015, TC-History-01) -----------------
@app.get("/api/history")
def api_get_history(
    status: Optional[str] = None,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(require_any_authenticated)
):
    return HistoryService.list_history(
        user_id=user["id"],
        role=user["role"],
        status_filter=status,
        model_id=model_id,
        dataset_id=dataset_id,
        limit=limit,
        offset=offset
    )

# ----------------- Admin Management & Monitoring (WME-SR-001, WME-SR-005) -----------------
@app.get("/api/admin/users")
def api_admin_list_users(user: dict = Depends(require_admin)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

class UpdateRoleRequest(BaseModel):
    role: str

@app.put("/api/admin/users/{user_id}/role")
def api_admin_update_role(user_id: str, req: UpdateRoleRequest, user: dict = Depends(require_admin)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (req.role.lower(), user_id))
    conn.commit()
    conn.close()
    log_audit_event("ROLE_UPDATE", f"User:{user_id}", user_id=user["id"], details={"new_role": req.role})
    return {"message": "User role updated successfully"}

@app.get("/api/admin/audit-logs")
def api_admin_audit_logs(limit: int = 100, offset: int = 0, user: dict = Depends(require_admin)):
    return get_audit_logs(limit=limit, offset=offset)

@app.get("/api/admin/system-health")
def api_admin_health(user: dict = Depends(require_admin)):
    cpu_usage = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(".")

    return {
        "status": "healthy",
        "cpu_usage_percent": cpu_usage,
        "memory_used_mb": round((memory.total - memory.available) / (1024 * 1024), 1),
        "memory_total_mb": round(memory.total / (1024 * 1024), 1),
        "memory_usage_percent": memory.percent,
        "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
        "disk_total_gb": round(disk.total / (1024 * 1024 * 1024), 2)
    }

# ----------------- Demo Dataset & Model Generator -----------------
@app.post("/api/demo/seed-sample-data")
def api_seed_sample_data(user: dict = Depends(require_evaluator_or_admin)):
    """
    Seeds ready-to-evaluate classification (Iris Logistic Regression & Random Forest)
    and regression (Diabetes Ridge & Random Forest Regressor) models and test datasets!
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris, load_diabetes
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split

    # 1. Iris Classification Models & Dataset
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    # Map feature names to clean names
    df_iris.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df_iris.columns]
    feature_cols = [c for c in df_iris.columns if c != "target"]
    X_train, X_test, y_train, y_test = train_test_split(df_iris[feature_cols], df_iris["target"], test_size=0.3, random_state=42)

    # Train Logistic Regression
    clf_lr = LogisticRegression(max_iter=300)
    clf_lr.fit(X_train, y_train)

    # Train Random Forest
    clf_rf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf_rf.fit(X_train, y_train)

    # Save models
    import io, joblib
    buf_lr = io.BytesIO()
    joblib.dump(clf_lr, buf_lr)
    buf_lr.seek(0)
    m1 = ModelService.save_model_artifact(user["id"], "Iris Logistic Classifier", "iris_logistic.pkl", buf_lr.read())

    buf_rf = io.BytesIO()
    joblib.dump(clf_rf, buf_rf)
    buf_rf.seek(0)
    m2 = ModelService.save_model_artifact(user["id"], "Iris Random Forest", "iris_rf.pkl", buf_rf.read())

    # Save Iris Test Dataset
    test_df_iris = X_test.copy()
    test_df_iris["target"] = y_test
    csv_bytes = test_df_iris.to_csv(index=False).encode("utf-8")
    ds1 = DatasetService.save_dataset_artifact(user["id"], "Iris Test Dataset", "iris_test.csv", csv_bytes)

    # 2. Diabetes Regression Models & Dataset
    diab = load_diabetes(as_frame=True)
    df_diab = diab.frame
    feat_diab = [c for c in df_diab.columns if c != "target"]
    X_dtrain, X_dtest, y_dtrain, y_dtest = train_test_split(df_diab[feat_diab], df_diab["target"], test_size=0.25, random_state=42)

    reg_ridge = Ridge()
    reg_ridge.fit(X_dtrain, y_dtrain)

    reg_rf = RandomForestRegressor(n_estimators=30, random_state=42)
    reg_rf.fit(X_dtrain, y_dtrain)

    buf_ridge = io.BytesIO()
    joblib.dump(reg_ridge, buf_ridge)
    buf_ridge.seek(0)
    m3 = ModelService.save_model_artifact(user["id"], "Diabetes Ridge Regressor", "diabetes_ridge.pkl", buf_ridge.read())

    buf_reg_rf = io.BytesIO()
    joblib.dump(reg_rf, buf_reg_rf)
    buf_reg_rf.seek(0)
    m4 = ModelService.save_model_artifact(user["id"], "Diabetes RF Regressor", "diabetes_rf.pkl", buf_reg_rf.read())

    test_df_diab = X_dtest.copy()
    test_df_diab["target"] = y_dtest
    csv_diab_bytes = test_df_diab.to_csv(index=False).encode("utf-8")
    ds2 = DatasetService.save_dataset_artifact(user["id"], "Diabetes Test Dataset", "diabetes_test.csv", csv_diab_bytes)

    return {
        "message": "Demo sample models and datasets created successfully!",
        "classification_models": [m1["id"], m2["id"]],
        "classification_dataset": ds1["id"],
        "regression_models": [m3["id"], m4["id"]],
        "regression_dataset": ds2["id"]
    }

# ----------------- Serve Frontend Static UI -----------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Web Based ML Model Evaluator</h1><p>Frontend file index.html is being prepared.</p>"
