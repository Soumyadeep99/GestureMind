"""
GestureMind — Database Connection
====================================
Connects to Supabase Postgres via SQLAlchemy.
Tables are auto-created on startup (no migrations needed for hackathon scope).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set in .env. "
        "Get it from Supabase: Project Settings → Database → Connection String → URI"
    )

# Supabase requires SSL; psycopg2 handles this automatically with the URI as given
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
