from fastapi import HTTPException, status, Depends
from src.wme.core.auth import get_current_user

class RoleChecker:
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = [r.lower() for r in allowed_roles]

    def __call__(self, user: dict = Depends(get_current_user)):
        user_role = user.get("role", "").lower()
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{user_role}' lacks required permissions."
            )
        return user

# Role access helpers
require_admin = RoleChecker(["administrator"])
require_evaluator_or_admin = RoleChecker(["administrator", "evaluator"])
require_any_authenticated = RoleChecker(["administrator", "evaluator", "faculty"])
require_faculty_or_admin = RoleChecker(["administrator", "faculty"])
