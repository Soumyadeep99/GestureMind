"""
GestureMind — Database Models
================================
User        — one row per registered account
EmergencyContact — one or more contacts per user, used for urgency email alerts
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    contacts = relationship(
        "EmergencyContact",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    name       = Column(String, nullable=False)
    email      = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="contacts")
