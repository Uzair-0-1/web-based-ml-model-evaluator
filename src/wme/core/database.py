import sqlite3
import json
import uuid
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from src.wme.core.config import DATABASE_PATH

def get_db_connection():
    """Returns a SQLite connection with row_factory enabled and parameterized queries."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hashes a password with a 16-byte salt using PBKDF2-HMAC-SHA256 (WME-SR-003)."""
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    ).hex()
    return hashed, salt

def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verifies a password against hash and salt."""
    computed, _ = hash_password(password, salt)
    return computed == hashed

def init_db():
    """Initializes the database schema and seeds initial administrative and evaluator accounts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'evaluator',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        last_login TEXT
    )
    """)

    # Active Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)

    # Models Table (WME-F-003, WME-F-004)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        framework TEXT NOT NULL,
        model_type TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)

    # Datasets Table (WME-F-005, WME-F-006)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL,
        columns_json TEXT NOT NULL,
        preview_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)

    # Evaluations Table (WME-F-007, WME-F-008, WME-F-009)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        model_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        eval_type TEXT NOT NULL,
        target_column TEXT NOT NULL,
        feature_columns_json TEXT NOT NULL,
        selected_metrics_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        error_message TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
    )
    """)

    # Results Table (WME-F-010, WME-F-011, WME-F-012, WME-F-013)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id TEXT PRIMARY KEY,
        evaluation_id TEXT UNIQUE NOT NULL,
        metrics_json TEXT NOT NULL,
        visualizations_json TEXT NOT NULL,
        predictions_sample_json TEXT NOT NULL,
        execution_time_seconds REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (evaluation_id) REFERENCES evaluations (id) ON DELETE CASCADE
    )
    """)

    # Audit Logs Table (WME-SR-005)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        username TEXT,
        action TEXT NOT NULL,
        resource TEXT NOT NULL,
        details_json TEXT,
        ip_address TEXT,
        timestamp TEXT NOT NULL
    )
    """)

    conn.commit()

    # Seed Default Roles & Accounts if not already existing
    now = datetime.now(timezone.utc).isoformat()
    default_users = [
        ("admin-001", "admin", "admin@wme.local", "Admin@123", "administrator"),
        ("evaluator-001", "evaluator", "evaluator@wme.local", "Eval@123", "evaluator"),
        ("faculty-001", "faculty", "faculty@wme.local", "Faculty@123", "faculty")
    ]

    for uid, uname, email, pwd, role in default_users:
        cursor.execute("SELECT id FROM users WHERE username = ?", (uname,))
        if not cursor.fetchone():
            pwd_hash, salt = hash_password(pwd)
            cursor.execute(
                """
                INSERT INTO users (id, username, email, password_hash, salt, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (uid, uname, email, pwd_hash, salt, role, now)
            )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
