"""
Web Based ML Model Evaluator (WME) - Server Entrypoint
Usage:
    python run.py
"""
import sys
import os
import uvicorn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.wme.core.database import init_db

if __name__ == "__main__":
    print("=" * 60)
    print("  Starting Web Based ML Model Evaluator (WME)")
    print("  SRS Problem Statement No. 25")
    print("=" * 60)
    print("Initializing Database...")
    init_db()
    print("Database ready.")
    print("Server running at: http://127.0.0.1:8000")
    print("Press Ctrl+C to terminate.")
    print("=" * 60)
    uvicorn.run("src.wme.api.app:app", host="127.0.0.1", port=8000, reload=False)
