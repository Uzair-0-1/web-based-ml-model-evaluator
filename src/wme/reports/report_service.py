import json
from pathlib import Path
from src.wme.evaluator.engine import EvaluationEngine
from src.wme.models.model_service import ModelService
from src.wme.datasets.dataset_service import DatasetService
from src.wme.core.config import REPORTS_DIR

class ReportService:
    """
    Generates and exports evaluation reports in HTML, JSON, and Markdown formats (WME-F-014, TC-Report-01).
    """

    @staticmethod
    def get_full_report_data(evaluation_id: str) -> dict:
        eval_info = EvaluationEngine.get_evaluation(evaluation_id)
        if not eval_info:
            raise ValueError(f"Evaluation {evaluation_id} not found.")

        model = ModelService.get_model(eval_info["model_id"])
        dataset = DatasetService.get_dataset(eval_info["dataset_id"])

        return {
            "evaluation": eval_info,
            "model": model,
            "dataset": dataset,
            "result": eval_info.get("result")
        }

    @classmethod
    def generate_html_report(cls, evaluation_id: str) -> str:
        data = cls.get_full_report_data(evaluation_id)
        ev = data["evaluation"]
        mod = data["model"]
        ds = data["dataset"]
        res = data["result"]
        metrics = res["metrics"] if res else {}
        visualizations = res["visualizations"] if res else {}

        # Render metric cards/table
        metric_cards_html = ""
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_cards_html += f"""
                <div class="card metric-card">
                    <div class="metric-title">{k.replace('_', ' ').upper()}</div>
                    <div class="metric-val">{v}</div>
                </div>
                """

        # Render visual images
        plots_html = ""
        for name, b64 in visualizations.items():
            plots_html += f"""
            <div class="plot-box">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="{b64}" alt="{name}" style="max-width:100%; border-radius: 8px; border: 1px solid #e2e8f0;"/>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WME Evaluation Report - {ev['name']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #1e293b; background: #f8fafc; }}
        .container {{ max-width: 960px; margin: 0 auto; background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-top: 0; }}
        h2 {{ color: #1e293b; margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
        .badge {{ background: #10b981; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 13px; font-weight: 600; display: inline-block; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
        .metric-card {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 16px; border-radius: 8px; text-align: center; }}
        .metric-title {{ font-size: 13px; color: #1e40af; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .metric-val {{ font-size: 28px; font-weight: 700; color: #1d4ed8; margin-top: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
        th {{ background: #f1f5f9; color: #475569; }}
        .plot-box {{ margin: 24px 0; text-align: center; }}
        .footer {{ margin-top: 40px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 16px; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>Web Based ML Model Evaluator</h1>
            <span class="badge">STATUS: {ev['status'].upper()}</span>
        </div>
        <p><strong>Evaluation Run:</strong> {ev['name']} (ID: {ev['id']})</p>
        <p><strong>Generated At:</strong> {ev.get('completed_at', ev['created_at'])} | <strong>Execution Time:</strong> {res.get('execution_time_seconds', 0) if res else 0}s</p>

        <h2>Artifacts & Specifications</h2>
        <table>
            <tr><th>Attribute</th><th>Model Specification</th><th>Dataset Specification</th></tr>
            <tr>
                <td><strong>Name</strong></td>
                <td>{mod['name'] if mod else 'N/A'}</td>
                <td>{ds['name'] if ds else 'N/A'}</td>
            </tr>
            <tr>
                <td><strong>Type / Framework</strong></td>
                <td>{ev['eval_type'].title()} ({mod['framework'] if mod else 'N/A'})</td>
                <td>{ds['row_count'] if ds else 0} rows &bull; {ds['column_count'] if ds else 0} columns</td>
            </tr>
            <tr>
                <td><strong>Target Variable</strong></td>
                <td colspan="2"><span style="color:#2563eb; font-weight:600;">{ev['target_column']}</span></td>
            </tr>
            <tr>
                <td><strong>Evaluated Features</strong></td>
                <td colspan="2">{', '.join(ev['feature_columns'])}</td>
            </tr>
        </table>

        <h2>Performance Metrics</h2>
        <div class="grid">
            {metric_cards_html}
        </div>

        <h2>Evaluator Visualizations</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 24px;">
            {plots_html}
        </div>

        <div class="footer">
            Generated by Web Based ML Model Evaluator &bull; Standard Conformance SRS v1.0
        </div>
    </div>
</body>
</html>"""
        return html_content

    @classmethod
    def generate_markdown_report(cls, evaluation_id: str) -> str:
        data = cls.get_full_report_data(evaluation_id)
        ev = data["evaluation"]
        mod = data["model"]
        ds = data["dataset"]
        res = data["result"]
        metrics = res["metrics"] if res else {}

        md = f"""# Web Based ML Model Evaluator - Report

**Evaluation:** {ev['name']} (`{ev['id']}`)  
**Status:** {ev['status'].upper()}  
**Evaluation Type:** {ev['eval_type'].title()}  
**Target Column:** `{ev['target_column']}`  
**Model:** {mod['name'] if mod else 'N/A'} (`{mod['framework'] if mod else 'N/A'}`)  
**Dataset:** {ds['name'] if ds else 'N/A'} ({ds['row_count'] if ds else 0} rows, {ds['column_count'] if ds else 0} columns)  
**Execution Time:** {res.get('execution_time_seconds', 0) if res else 0} seconds  

---

## Performance Metrics

| Metric | Score |
| :--- | :--- |
"""
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                md += f"| {k.replace('_', ' ').upper()} | {v} |\n"

        md += "\n---\n*Report generated by Web Based ML Model Evaluator (SRS Problem Statement No. 25)*\n"
        return md
