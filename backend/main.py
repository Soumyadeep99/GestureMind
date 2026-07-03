"""
GestureMind — FastAPI Backend (Model + Agent + Email Alerts)
================================================================
Endpoints:
  GET  /          — health check
  GET  /health    — detailed status
  POST /predict   — LSTM gesture prediction
  POST /agent     — Gemini agentic processing (auto-sends email on HIGH urgency)
  POST /agent/chat — freeform chat with agent
  DELETE /agent/session — clear agent memory
  GET  /alerts/status — email alert configuration status

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
from fastapi              import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic             import BaseModel, validator
from dotenv                import load_dotenv

from agent       import SignSenseAgent
from email_alert import EmailAlertService

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
log = logging.getLogger("gesturemind")

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(os.getenv("PROJECT_DIR", r"D:\Downloads\data_collection"))
MODEL_DIR    = PROJECT_DIR / "trained_model"
MODEL_PATH   = MODEL_DIR   / "signsense_model.keras"
LABELS_PATH  = MODEL_DIR   / "labels.json"
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
ALERT_RECIPIENTS   = os.getenv("ALERT_RECIPIENT_EMAILS", "").split(",")

SEQUENCE_LENGTH   = 30
FEATURE_SIZE      = 1662
CONFIDENCE_THRESH = 0.70
UNKNOWN_LABEL     = "..."

app = FastAPI(title="GestureMind", description="Real-Time ASL Recognition + Agentic AI + Alerts", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:5173","http://127.0.0.1:3000","http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class AppState:
    model       : tf.keras.Model      = None
    actions     : List[str]           = []
    agent       : SignSenseAgent      = None
    email_svc   : EmailAlertService   = None
    model_ready : bool                = False
    agent_ready : bool                = False
    email_ready : bool                = False
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

    state.email_svc = EmailAlertService(
        sender_email=GMAIL_ADDRESS,
        app_password=GMAIL_APP_PASSWORD,
        recipients=ALERT_RECIPIENTS
    )
    state.email_ready = state.email_svc.enabled

    state.load_time = time.time() - t0
    log.info(f"GestureMind ready in {state.load_time:.2f}s | Email alerts: {state.email_ready}")


# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "GestureMind", "version": "2.1.0", "status": "running",
        "model_ready": state.model_ready, "agent_ready": state.agent_ready,
        "email_alerts_ready": state.email_ready, "classes": state.actions,
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if state.model_ready else "degraded",
        "model_loaded": state.model_ready, "agent_loaded": state.agent_ready,
        "email_alerts_ready": state.email_ready,
        "model_load_time": f"{state.load_time:.2f}s",
        "classes": state.actions, "num_classes": len(state.actions),
        "sequence_length": SEQUENCE_LENGTH, "feature_size": FEATURE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESH,
    }

@app.get("/alerts/status")
def alerts_status():
    return {
        "enabled": state.email_ready,
        "recipients_configured": len(ALERT_RECIPIENTS) if state.email_ready else 0,
        "cooldown_seconds": state.email_svc.COOLDOWN_SECONDS if state.email_svc else None,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
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
def agent_process(request: AgentRequest):
    """Agentic processing — auto-sends email alert if urgency is HIGH."""
    if not state.agent_ready:
        raise HTTPException(503, "Agent not available. Set GEMINI_API_KEY in .env")
    if not request.words:
        raise HTTPException(400, "No words provided.")

    result = state.agent.process_signs(words=request.words, user_message=request.user_message)

    # ── Auto-trigger email alert on HIGH urgency ──────────────────────────────
    email_sent   = False
    email_reason = None
    if result.get("urgency") == "HIGH" and state.email_ready:
        alert_result = state.email_svc.send_urgency_alert(
            urgency_message=result.get("urgency_message") or "User signed for HELP.",
            signs_detected=request.words,
            sentence=result.get("sentence")
        )
        email_sent   = alert_result["sent"]
        email_reason = alert_result["reason"]
        if email_sent:
            log.warning(f"🚨 EMAIL ALERT SENT — urgency detected: {request.words}")

    result["email_alert_sent"]   = email_sent
    result["email_alert_reason"] = email_reason

    return AgentResponse(**{k: result[k] for k in AgentResponse.__fields__ if k in result})

@app.post("/agent/chat")
def agent_chat(request: AgentChatRequest):
    if not state.agent_ready:
        raise HTTPException(503, "Agent not available. Set GEMINI_API_KEY in .env")
    return state.agent.process_signs(words=request.current_words or [], user_message=request.message)

@app.delete("/agent/session")
def clear_session():
    if state.agent: state.agent.clear_session()
    return {"status": "cleared", "message": "Agent session memory reset."}
