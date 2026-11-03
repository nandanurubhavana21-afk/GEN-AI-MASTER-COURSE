"""
auth.py - Authentication & Authorization Module.
Handles password hashing, JWT token creation, and route decorators for Flask.
Supports PyJWT and python-jose.
"""

import os
import time
import functools
import logging
from typing import Optional, Dict, Any
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

from database import get_user_by_id

logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "resumex-super-secret-jwt-key-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 48


def hash_password(password: str) -> str:
    """Hashes a plaintext password using secure PBKDF2/sha256."""
    return generate_password_hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored hash."""
    return check_password_hash(hashed_password, plain_password)


def generate_auth_token(user_id: int, role: str = "student", expires_in: int = JWT_EXPIRATION_HOURS * 3600) -> str:
    """Generates a signed JWT authentication token."""
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in
    }

    try:
        import jwt
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token
    except ImportError:
        from jose import jwt as jose_jwt
        return jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_auth_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT token."""
    try:
        try:
            import jwt
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except ImportError:
            from jose import jwt as jose_jwt
            payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
    except Exception as e:
        logger.warning(f"Failed to decode token: {e}")
        return None


def get_current_user_from_request() -> Optional[Dict[str, Any]]:
    """Extracts and verifies user from Authorization header or cookie."""
    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    elif request.cookies.get("auth_token"):
        token = request.cookies.get("auth_token")
    elif request.args.get("token"):
        token = request.args.get("token")

    if not token:
        return None

    payload = decode_auth_token(token)
    if not payload:
        return None

    user_id = payload.get("user_id") or payload.get("sub")
    if not user_id:
        return None

    try:
        user = get_user_by_id(int(user_id))
        return user
    except Exception:
        return None


def token_required(f):
    """Decorator to enforce that the caller has a valid JWT token."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user_from_request()
        if not user:
            return jsonify({"error": "Authentication required. Please log in.", "code": "AUTH_REQUIRED"}), 401
        if not user.get("is_active"):
            return jsonify({"error": "User account has been deactivated. Please contact support."}), 403

        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator to enforce that the caller is an authenticated administrator."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user_from_request()
        if not user:
            return jsonify({"error": "Authentication required.", "code": "AUTH_REQUIRED"}), 401
        if user.get("role") != "admin":
            return jsonify({"error": "Administrator privilege required for this resource.", "code": "FORBIDDEN"}), 403
        if not user.get("is_active"):
            return jsonify({"error": "Administrator account has been deactivated."}), 403

        g.current_user = user
        return f(*args, **kwargs)
    return decorated

