"""Authentication: hard-coded demo users + signed JWT bearer tokens.

Passwords are PBKDF2-hashed at import time (demo credentials only - swap to
Keycloak/OIDC: replace `authenticate()` and `decode_token()`; RBAC code is unchanged).
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.errors import AuthenticationError
from app.security.rbac import Role, UserContext

_SALT = b"enterprise-demo-salt"


def _hash(password: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), _SALT, 50_000)


@dataclass(frozen=True)
class UserRecord:
    username: str
    name: str
    role: Role
    department: str
    password_hash: bytes


DEMO_USERS: dict[str, UserRecord] = {
    u.username: u
    for u in (
        UserRecord("viewer", "Vera Viewer", Role.VIEWER, "customer-support", _hash("viewer123")),
        UserRecord("analyst", "Anil Analyst", Role.ANALYST, "payments", _hash("analyst123")),
        UserRecord("admin", "Asha Admin", Role.ADMIN, "platform-engineering", _hash("admin123")),
    )
}


def authenticate(username: str, password: str) -> UserContext:
    rec = DEMO_USERS.get(username)
    # Compare even for unknown users so response time does not reveal which usernames exist.
    candidate = _hash(password)
    if rec is None or not hmac.compare_digest(candidate, rec.password_hash):
        raise AuthenticationError("Invalid username or password")
    return UserContext(rec.username, rec.name, rec.role, rec.department)


def create_token(user: UserContext) -> str:
    s = get_settings()
    payload = {**user.to_dict(), "sub": user.user_id, "exp": int(time.time()) + s.jwt_ttl_minutes * 60}
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> UserContext:
    try:
        data = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])
        return UserContext.from_dict(data)
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired token") from exc


_bearer = HTTPBearer(auto_error=False)


async def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> UserContext:
    if creds is None:
        raise AuthenticationError("Missing bearer token")
    return decode_token(creds.credentials)
