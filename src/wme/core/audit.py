import json
import uuid
from datetime import datetime, timezone
from src.wme.core.database import get_db_connection

def log_audit_event(action: str, resource: str, user_id: str = None, username: str = None, details: dict = None, ip_address: str = "127.0.0.1"):
    """
    Records security-sensitive evaluation and system events to the audit trail (WME-SR-005).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    audit_id = f"aud-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    details_str = json.dumps(details or {})

    cursor.execute(
        """
        INSERT INTO audit_logs (id, user_id, username, action, resource, details_json, ip_address, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (audit_id, user_id, username, action, resource, details_str, ip_address, now)
    )
    conn.commit()
    conn.close()
    return audit_id

def get_audit_logs(limit: int = 100, offset: int = 0):
    """Retrieves recent audit logs for administrative monitoring."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?
        """,
        (limit, offset)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
