"""
GestureMind — FastAPI Backend (Model + Agent + Email Alerts + Auth)
======================================================================
Endpoints:
  GET    /                 — health check
  GET    /health            — detailed status
  POST   /auth/register     — create account
  POST   /auth/login        — login, returns JWT
  GET    /contacts          — list logged-in user's emergency contacts
  POST   /contacts          — add an emergency contact
  DELETE /contacts/{id}     — remove an emergency contact
  POST   /predict           — LSTM gesture prediction               [AUTH REQUIRED]
  POST   /agent             — Gemini agentic processing + auto-email [AUTH REQUIRED]
  POST   /agent/chat        — freeform chat with agent               [AUTH REQUIRED]
  DELETE /agent/session     — clear agent memory                     [AUTH REQUIRED]

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing  import List, Dict, Optional

import tensorflow as tf
from fastapi              import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic              import BaseModel, validator, EmailStr
from sqlalchemy.orm        import Session
from dotenv                 import load_dotenv

from database  import get_db, engine, Base
from models_db import User, EmergencyContact
from auth      import hash_password, verify_password, create_access_token, get_current_user
from agent       import SignSenseAgent
from email_alert import EmailAlertService

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
log = logging.getLogger("gesturemind")

# ── Create DB tables on startup (no migrations needed for this scope) ────────
Base.metadata.create_all(bind=engine)

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT_DIR  = Path(os.getenv("PROJECT_DIR", r"D:\Downloads\data_collection"))
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR    = PROJECT_DIR / "trained_model"
MODEL_PATH   = MODEL_DIR   / "signsense_model.keras"
LABELS_PATH  = MODEL_DIR   / "labels.json"
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

SEQUENCE_LENGTH   = 30
FEATURE_SIZE      = 1662
CONFIDENCE_THRESH = 0.70
UNKNOWN_LABEL     = "..."

app = FastAPI(title="GestureMind", description="Real-Time ASL Recognition + Agentic AI + Auth", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://gesture-mind-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    model       : tf.keras.Model      = None
    actions     : List[str]           = []
    agent       : SignSenseAgent      = None
    email_svc   : EmailAlertService   = None
    email_ready : bool                = False
    model_ready : bool                = False
    agent_ready : bool                = False
    load_time   : float               = 0.0

state = AppState()


@app.on_event("startup")
async def startup():
    t0 = time.time()

    if MODEL_PATH.exists() and LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            labels_data = json.load(f)
        state.actions = labels_data["actions"]
        state.model   = tf.keras.models.load_model(MODEL_PATH)
        state.model.predict(np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE)), verbose=0)
        state.model_ready = True
        log.info(f"LSTM model loaded. Classes: {state.actions}")
    else:
        log.warning(f"Model not found at {MODEL_PATH}")

    if GEMINI_KEY:
        try:
            state.agent = SignSenseAgent(api_key=GEMINI_KEY)
            state.agent_ready = True
            log.info("Gemini agent initialized.")
        except Exception as e:
            log.error(f"Agent init failed: {e}")
    else:
        log.warning("GEMINI_API_KEY not set. Agent disabled.")

    # Email service — sender identity only; recipients now come from DB per-user
    state.email_svc = EmailAlertService(
        sender_email=GMAIL_ADDRESS,
        app_password=GMAIL_APP_PASSWORD,
        recipients=[]  # placeholder — overridden per-call with user's DB contacts
    )
    state.email_ready = state.email_svc.enabled

    state.load_time = time.time() - t0
    log.info(f"GestureMind ready in {state.load_time:.2f}s")


# ═════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas
# ═════════════════════════════════════════════════════════════════════════════
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @validator("password")
    def password_min_length(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str

class ContactCreate(BaseModel):
    name: str
    email: EmailStr

class ContactResponse(BaseModel):
    id: int
    name: str
    email: str
    class Config:
        orm_mode = True

class LandmarkFrame(BaseModel):
    keypoints: List[float]
    @validator("keypoints")
    def v_kp(cls, v):
        if len(v) != FEATURE_SIZE: raise ValueError(f"Expected {FEATURE_SIZE} keypoints, got {len(v)}")
        return v

class PredictRequest(BaseModel):
    frames: List[LandmarkFrame]
    @validator("frames")
    def v_frames(cls, v):
        if len(v) != SEQUENCE_LENGTH: raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {len(v)}")
        return v

class PredictResponse(BaseModel):
    gesture: str; confidence: float; is_confident: bool
    all_probabilities: Dict[str, float]; inference_time_ms: float

class AgentRequest(BaseModel):
    words: List[str]; user_message: Optional[str] = None

class AgentChatRequest(BaseModel):
    message: str; current_words: Optional[List[str]] = []

class AgentResponse(BaseModel):
    sentence: str; intent: str; urgency: str
    urgency_message: Optional[str]; suggestions: List[str]
    agent_message: str; tools_used: List[str]
    latency_ms: float; signs_used: List[str]
    email_alert_sent: Optional[bool] = False
    email_alert_reason: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
#  Public Routes
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "service": "GestureMind", "version": "3.0.0", "status": "running",
        "model_ready": state.model_ready, "agent_ready": state.agent_ready,
        "classes": state.actions,
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if state.model_ready else "degraded",
        "model_loaded": state.model_ready,
        "agent_loaded": state.agent_ready,
        "email_alerts_ready": state.email_ready,   # <-- Add this line
        "model_load_time": f"{state.load_time:.2f}s",
        "classes": state.actions,
        "num_classes": len(state.actions),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_size": FEATURE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESH,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Auth Routes
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/auth/register", response_model=TokenResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(400, "An account with this email already exists.")

    user = User(email=request.email, hashed_password=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.email})
    log.info(f"New user registered: {user.email}")
    return TokenResponse(access_token=token, email=user.email)


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password.")

    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token, email=user.email)


# ═════════════════════════════════════════════════════════════════════════════
#  Emergency Contacts Routes  (all require login)
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/contacts", response_model=List[ContactResponse])
def list_contacts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(EmergencyContact).filter(EmergencyContact.user_id == current_user.id).all()


@app.post("/contacts", response_model=ContactResponse)
def add_contact(
    contact: ContactCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    existing_count = db.query(EmergencyContact).filter(EmergencyContact.user_id == current_user.id).count()
    if existing_count >= 5:
        raise HTTPException(400, "Maximum 5 emergency contacts allowed.")

    new_contact = EmergencyContact(
        user_id=current_user.id, name=contact.name, email=contact.email
    )
    db.add(new_contact)
    db.commit()
    db.refresh(new_contact)
    return new_contact


@app.delete("/contacts/{contact_id}")
def delete_contact(
    contact_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    contact = db.query(EmergencyContact).filter(
        EmergencyContact.id == contact_id,
        EmergencyContact.user_id == current_user.id
    ).first()
    if not contact:
        raise HTTPException(404, "Contact not found.")
    db.delete(contact)
    db.commit()
    return {"status": "deleted", "id": contact_id}


# ═════════════════════════════════════════════════════════════════════════════
#  Protected — Recognition & Agent Routes
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    current_user: User = Depends(get_current_user)
):
    if not state.model_ready:
        raise HTTPException(503, "Model not loaded.")
    t0 = time.time()
    sequence = np.expand_dims(np.array([f.keypoints for f in request.frames], dtype=np.float32), axis=0)
    probs = state.model.predict(sequence, verbose=0)[0]
    pred_idx = int(np.argmax(probs)); confidence = float(probs[pred_idx])
    is_conf = confidence >= CONFIDENCE_THRESH
    gesture = state.actions[pred_idx] if is_conf else UNKNOWN_LABEL
    all_probs = {a: round(float(p), 4) for a, p in zip(state.actions, probs)}
    return PredictResponse(
        gesture=gesture, confidence=round(confidence, 4), is_confident=is_conf,
        all_probabilities=all_probs, inference_time_ms=round((time.time()-t0)*1000, 2)
    )


@app.post("/agent", response_model=AgentResponse)
def agent_process(
    request: AgentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Agentic processing — auto-sends email to the LOGGED-IN USER's contacts if urgency is HIGH."""
    if not state.agent_ready:
        raise HTTPException(503, "Agent not available. Set GEMINI_API_KEY in .env")
    if not request.words:
        raise HTTPException(400, "No words provided.")

    result = state.agent.process_signs(words=request.words, user_message=request.user_message)

    email_sent   = False
    email_reason = None
    if result.get("urgency") == "HIGH":
        user_contacts = db.query(EmergencyContact).filter(
            EmergencyContact.user_id == current_user.id
        ).all()
        recipient_emails = [c.email for c in user_contacts]

        if not recipient_emails:
            email_reason = "No emergency contact configured. Add one in Settings."
        else:
            alert_result = state.email_svc.send_urgency_alert(
                urgency_message=result.get("urgency_message") or "User signed for HELP.",
                signs_detected=request.words,
                sentence=result.get("sentence"),
                recipients=recipient_emails
            )
            email_sent   = alert_result["sent"]
            email_reason = alert_result["reason"]
            if email_sent:
                log.warning(f"🚨 EMAIL ALERT SENT to {recipient_emails} for user {current_user.email}")

    result["email_alert_sent"]   = email_sent
    result["email_alert_reason"] = email_reason

    return AgentResponse(**{k: result[k] for k in AgentResponse.__fields__ if k in result})


@app.post("/agent/chat")
def agent_chat(
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user)
):
    if not state.agent_ready:
        raise HTTPException(503, "Agent not available. Set GEMINI_API_KEY in .env")
    return state.agent.process_signs(words=request.current_words or [], user_message=request.message)


@app.delete("/agent/session")
def clear_session(current_user: User = Depends(get_current_user)):
    if state.agent: state.agent.clear_session()
    return {"status": "cleared", "message": "Agent session memory reset."}