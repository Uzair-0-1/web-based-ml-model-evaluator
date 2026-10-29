# Web Based ML Model Evaluator (WME)

> **Software Requirements Specification (SRS) Problem Statement No. 25 &bull; Version 1.0**

A browser-based platform for evaluating trained machine-learning models against evaluation datasets, calculating performance metrics, generating visualizations, comparing models, and exporting reports.

---

## 🚀 Key Features (SRS Conformance)

| Requirement | Module | Feature Description |
| :--- | :--- | :--- |
| **WME-F-001** | `auth` | User registration, login, logout, session expiration. |
| **WME-F-002** | `rbac` | Role-based access control (`Evaluator`, `Administrator`, `Faculty`). |
| **WME-F-003** | `models` | Upload trained ML models (`.pkl`, `.joblib`, `.pt`, `.onnx`). |
| **WME-F-004** | `models` | Model validation, architecture inspection, error rejection. |
| **WME-F-005** | `datasets` | Upload evaluation datasets (`.csv`, `.json`, `.parquet`). |
| **WME-F-006** | `datasets` | Schema validation, missing value checking, model-dataset compatibility. |
| **WME-F-007** | `evaluator` | Evaluation configuration wizard (target column, feature columns, metrics). |
| **WME-F-008** | `evaluator` | Execution pipeline & model prediction generation in sandboxed environment. |
| **WME-F-009** | `evaluator` | Real-time job tracking (`queued`, `running`, `completed`, `failed`). |
| **WME-F-010** | `metrics` | Classification metrics: Accuracy, Precision, Recall, F1 (macro/micro/weighted), per-class. |
| **WME-F-011** | `metrics` | Regression metrics: MAE, MSE, RMSE, R²-score, Max Error, Explained Variance. |
| **WME-F-012** | `visualizations` | Confusion Matrix heatmaps, ROC curves, PR curves, Residual scatter plots. |
| **WME-F-013** | `comparison` | Multi-model comparative analysis, leaderboard, and side-by-side charts. |
| **WME-F-014** | `reports` | Downloadable reports in standalone HTML (with embedded plots), Markdown, and JSON. |
| **WME-F-015** | `history` | Complete historical archive with filtering, search, and audit trail. |
| **WME-SR-001..5** | `core` | PBKDF2 password hashing with salt, sandbox execution, parameterized SQL, audit logs. |

---

## 🛠️ Architecture & Deep Modules

- **`src/wme/core/`**: Database persistence (`SQLite` with parameterized queries), Cryptographic Authentication (`PBKDF2-HMAC-SHA256`), RBAC guards, and Execution Sandbox.
- **`src/wme/models/`**: Deep model adapter & inspector supporting Scikit-Learn, PyTorch, and ONNX.
- **`src/wme/datasets/`**: Dataset ingestion, validation, and automated feature-compatibility engine.
- **`src/wme/evaluator/`**: Metric calculation engine, execution pipeline, and headless matplotlib visualization generator.
- **`src/wme/comparison/`**: Comparative ranking, deltas, and cross-model performance aggregation.
- **`src/wme/reports/`**: Standalone HTML, Markdown, and JSON reporting engine.
- **`src/wme/frontend/`**: Responsive web application with interactive canvas charts (Chart.js) and intuitive workflows.

---

## 🏃 Quick Start

### 1. Launch Server
Using the Python environment:
```bash
python run.py
```
Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 2. Default Test Accounts
| Role | Username | Password |
| :--- | :--- | :--- |
| **Administrator** | `admin` | `Admin@123` |
| **User / Evaluator** | `evaluator` | `Eval@123` |
| **Project Guide / Faculty** | `faculty` | `Faculty@123` |

*(Fast 1-click login buttons are also provided in the login modal for quick switching during testing).*

### 3. Immediate 1-Click Evaluation Test
1. Click **"✨ 1-Click Load Demo Models & Datasets"** on the Dashboard.
2. Go to **"⚙️ New Evaluation"**, select a Model (e.g. *Iris Random Forest*) and Dataset (*Iris Test Dataset*).
3. Click **"🚀 Execute Model Evaluation"**.
4. View real-time Confusion Matrix, ROC curves, and download the HTML evaluation report!

---

## 🧪 Running Acceptance Tests

Run the complete test suite verifying all 15 functional requirements and security controls:
```bash
pytest tests/test_all_wme_srs.py -v
```
All 8 verification suites pass with 100% specification compliance.


## 📊 ML Evaluation Metrics

The evaluation engine calculates performance metrics for classification and regression models.

### Classification Metrics

- **Accuracy:** Proportion of predictions that are correct.
- **Precision:** Proportion of positive predictions that are correct. The default `precision` result uses macro averaging.
- **Recall:** Proportion of actual positive cases identified correctly. The default `recall` result uses macro averaging.
- **F1-score:** Harmonic mean of precision and recall. The default `f1_score` result uses macro averaging.
- **Micro and weighted averages:** Additional precision, recall, and F1-score values are provided using micro and weighted averaging.
- **Per-class metrics:** Precision, recall, F1-score, and support are reported for each class.
- **Confusion matrix:** Shows counts of actual versus predicted classes.
- **Sample count:** Number of evaluated observations.

### Regression Metrics

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target variable's units.
- **R² score:** Measures model performance relative to a baseline that predicts the target mean.
- **Maximum error:** Largest absolute prediction error.
- **Explained variance:** Measures how much variation in the target is explained by the predictions.
- **Residual statistics:** Includes the mean and standard deviation of prediction residuals.
- **Sample count:** Number of evaluated observations.

### Running Metric Tests

Run the dedicated metrics tests from the project root:

```bash
python -m pytest tests/test_metrics.py -v
```

The tests cover expected metric values, string classification labels, empty inputs, and mismatched input lengths.