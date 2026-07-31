"""
GestureMind — Authentication
===============================
Self-contained JWT auth — no external auth service needed.
Password hashing via bcrypt, tokens via python-jose.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

from database import get_db
from models_db import User

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM    = "HS256"
EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY not set in .env. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ─────────────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency — extracts and validates the Bearer token,
    returns the corresponding User row. Raises 401 if invalid/missing.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")

    token = authorization.replace("Bearer ", "").strip()
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")

    user_email = payload.get("sub")
    user = db.query(User).filter(User.email == user_email).first()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")

    return user
