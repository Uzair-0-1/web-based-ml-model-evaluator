# WME frontend

The browser interface is served from `index.html`. It uses Tailwind CSS (CDN) for utility styling and Chart.js (CDN) for comparison charts; application logic calls the FastAPI backend through relative `/api` endpoints.

## Dashboard areas

- **Dashboard:** summary counts and recent evaluation runs.
- **Models / Datasets:** upload, validate, inspect, and manage artifacts.
- **New Evaluation:** select a registered model and dataset and start an evaluation.
- **Results:** inspect metrics, plots, prediction samples, and downloadable reports.
- **Comparison:** compare completed evaluations.
- **History & Reports:** revisit prior runs and export reports.
- **Admin & Audit:** administrative telemetry and audit information for authorized users.

## Local development

Run the backend using the repository's documented setup, then open the frontend through the FastAPI application so relative `/api` requests use the same origin. Avoid opening `index.html` directly as a local file because API requests require the running backend.

When changing the frontend, preserve existing element IDs and handler names unless the corresponding JavaScript is updated. Check desktop and narrow viewport widths, and verify model upload, dataset upload, tab navigation, evaluation results, and comparison flows against a running backend.
