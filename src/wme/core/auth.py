import uuid
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Header, Depends, status
from src.wme.core.database import get_db_connection, hash_password, verify_password
from src.wme.core.config import SESSION_TIMEOUT_MINUTES
from src.wme.core.audit import log_audit_event

def register_user(username: str, email: str, password: str, role: str = "evaluator") -> dict:
    """Registers a new user account with hashed password and role (WME-F-001)."""
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    
    valid_roles = ["evaluator", "administrator", "faculty"]
    if role.lower() not in valid_roles:
        role = "evaluator"
    else:
        role = role.lower()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username or email already exists.")

    user_id = f"usr-{uuid.uuid4().hex[:12]}"
    pwd_hash, salt = hash_password(password)
    now = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        """
        INSERT INTO users (id, username, email, password_hash, salt, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (user_id, username, email, pwd_hash, salt, role, now)
    )
    conn.commit()
    conn.close()

    log_audit_event("USER_REGISTER", f"User:{user_id}", user_id=user_id, username=username, details={"role": role})
    return {"id": user_id, "username": username, "email": email, "role": role}

def authenticate_user(username: str, password: str, ip_address: str = "127.0.0.1") -> dict:
    """Authenticates credentials, generates a time-bound session token (WME-F-001, WME-SR-002, WME-SR-003)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        log_audit_event("AUTH_FAILED", f"Username:{username}", details={"reason": "User not found"}, ip_address=ip_address)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not user["is_active"]:
        conn.close()
        raise HTTPException(status_code=403, detail="Account is disabled.")

    if not verify_password(password, user["password_hash"], user["salt"]):
        conn.close()
        log_audit_event("AUTH_FAILED", f"Username:{username}", user_id=user["id"], username=username, details={"reason": "Invalid password"}, ip_address=ip_address)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Generate session token
    token = f"tok-{uuid.uuid4().hex}"
    now_dt = datetime.now(timezone.utc)
    expires_dt = now_dt + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    now = now_dt.isoformat()
    expires = expires_dt.isoformat()

    cursor.execute(
        """
        INSERT INTO sessions (token, user_id, role, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (token, user["id"], user["role"], now, expires)
    )

    # Update last login
    cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user["id"]))
    conn.commit()
    conn.close()

    log_audit_event("AUTH_LOGIN_SUCCESS", f"User:{user['id']}", user_id=user["id"], username=username, details={"role": user["role"]}, ip_address=ip_address)

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        },
        "expires_at": expires
    }

def logout_user(token: str) -> bool:
    """Terminates active session (WME-F-001)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE token = ?", (token,))
    session = cursor.fetchone()
    if session:
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        log_audit_event("AUTH_LOGOUT", f"Session:{token}", user_id=session["user_id"])
    conn.close()
    return True

def get_current_user(authorization: str = Header(None)) -> dict:
    """FastAPI dependency to extract and validate the authenticated user from the Authorization header."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing.")

    token = authorization
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE token = ?", (token,))
    session = cursor.fetchone()

    if not session:
        conn.close()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token.")

    # Check expiry (WME-SR-002)
    expires_at = datetime.fromisoformat(session["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please log in again.")

    # Fetch user
    cursor.execute("SELECT id, username, email, role, is_active, created_at, last_login FROM users WHERE id = ?", (session["user_id"],))
    user = cursor.fetchone()
    conn.close()

    if not user or not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive or not found.")

    return dict(user)
