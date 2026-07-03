"""
SignSense AI — Gemini Agentic Layer
=====================================
This module adds genuine agentic behavior on top of the LSTM classifier.

Agent capabilities (tools):
  1. form_sentence       — Takes raw recognized words → grammatically correct sentence
  2. detect_urgency      — Detects if user is in distress → triggers alert
  3. infer_intent        — Understands partial/ambiguous signs, infers user intent
  4. suggest_completion  — Suggests what the user might be trying to say next

Flow:
  Recognized words → Agent reasons → Selects tools → Returns structured response
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional
import google.generativeai as genai

log = logging.getLogger("signsense.agent")

# ─────────────────────────────────────────────────────────────────────────────
#  Agent Configuration
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"  # Fast + cheap — perfect for real-time
MAX_HISTORY     = 10                   # Conversation turns to keep in memory
URGENCY_SIGNS   = {"help", "stop", "sorry"}  # Signs that may indicate distress

# ─────────────────────────────────────────────────────────────────────────────
#  Tool Definitions — What the Agent Can Do
# ─────────────────────────────────────────────────────────────────────────────
AGENT_TOOLS = [
    {
        "name": "form_sentence",
        "description": (
            "Takes a list of raw ASL sign words and forms them into a natural, "
            "grammatically correct English sentence. Handles missing words, "
            "ASL grammar differences, and incomplete phrases."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of recognized ASL sign words in order"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context from previous conversation turns"
                }
            },
            "required": ["words"]
        }
    },
    {
        "name": "detect_urgency",
        "description": (
            "Analyzes recognized signs for urgency or distress signals. "
            "Returns urgency level and recommended action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of recognized ASL sign words"
                }
            },
            "required": ["words"]
        }
    },
    {
        "name": "infer_intent",
        "description": (
            "When recognized words are ambiguous or incomplete, infers the most "
            "likely intent of the user based on signs and conversation context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recognized signs so far"
                },
                "conversation_history": {
                    "type": "string",
                    "description": "Recent conversation context"
                }
            },
            "required": ["words"]
        }
    },
    {
        "name": "suggest_completion",
        "description": (
            "Suggests what the user might be trying to communicate next, "
            "based on current signs and conversational context. "
            "Returns 2-3 likely completions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signs recognized so far"
                }
            },
            "required": ["words"]
        }
    }
]

# ─────────────────────────────────────────────────────────────────────────────
#  Tool Execution — Local logic before calling Gemini
# ─────────────────────────────────────────────────────────────────────────────
def execute_tool(tool_name: str, params: dict) -> dict:
    """
    Execute a tool call requested by the Gemini agent.
    Returns structured result that goes back to the agent.
    """
    if tool_name == "detect_urgency":
        words_lower = [w.lower() for w in params.get("words", [])]
        urgent      = any(w in URGENCY_SIGNS for w in words_lower)
        level       = "HIGH" if "help" in words_lower else ("MEDIUM" if urgent else "LOW")
        return {
            "urgency_level"     : level,
            "urgent_signs_found": [w for w in words_lower if w in URGENCY_SIGNS],
            "recommended_action": "Alert caregiver immediately" if level == "HIGH" else
                                  "Monitor situation" if level == "MEDIUM" else
                                  "Normal communication — no action needed",
            "alert_message"     : f"⚠️ User may need help! Signs detected: {', '.join(words_lower)}" if urgent else None
        }

    # For other tools, return context for Gemini to process
    return {"words": params.get("words", []), "status": "ready_for_llm"}


# ─────────────────────────────────────────────────────────────────────────────
#  SignSense Agent Class
# ─────────────────────────────────────────────────────────────────────────────
class SignSenseAgent:
    """
    Agentic AI layer for SignSense AI.

    Wraps Gemini with:
    - Persistent conversation memory
    - Tool calling (form_sentence, detect_urgency, infer_intent, suggest_completion)
    - Structured responses
    - Urgency detection
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model            = genai.GenerativeModel(GEMINI_MODEL)
        self.conversation_history: List[Dict] = []
        self.session_signs    : List[str]     = []  # All signs in this session
        self.initialized      = True
        log.info(f"SignSense Agent initialized with model: {GEMINI_MODEL}")

    def _build_system_prompt(self) -> str:
        return """You are SignSense AI Agent — an intelligent assistant embedded in a 
real-time American Sign Language (ASL) recognition system.

Your role is to help deaf and mute users communicate effectively by:
1. Converting raw recognized ASL signs into natural English sentences
2. Inferring intent when signs are incomplete or ambiguous  
3. Detecting urgency or distress from sign patterns
4. Suggesting what the user might want to say next
5. Maintaining conversation context across multiple sign inputs

IMPORTANT CONTEXT:
- The user communicates ONLY through ASL signs detected by a camera
- Signs recognized may be: hello, thank_you, please, yes, no, sorry, help, good, bad, stop
- Signs come in sequence — order matters for meaning
- ASL grammar differs from English (often Subject-Object-Verb order)
- Be empathetic, clear, and concise in all responses
- If urgency is detected (especially "help"), prioritize that above all else

RESPONSE FORMAT:
Always respond with a JSON object:
{
  "sentence": "The natural English sentence formed from the signs",
  "intent": "Brief description of what the user is trying to communicate",
  "urgency": "LOW | MEDIUM | HIGH",
  "urgency_message": "Alert message if urgency is HIGH, null otherwise",
  "suggestions": ["possible next sign 1", "possible next sign 2"],
  "agent_message": "Your helpful response to the user",
  "tools_used": ["list of reasoning steps you applied"]
}

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

    def _format_history(self) -> str:
        if not self.conversation_history:
            return "No previous conversation."
        recent = self.conversation_history[-MAX_HISTORY:]
        return "\n".join([
            f"{'User signed' if h['role']=='user' else 'Agent'}: {h['content']}"
            for h in recent
        ])

    def process_signs(self, words: List[str], user_message: Optional[str] = None) -> Dict:
        """
        Main agent method — processes recognized signs and returns structured response.

        Args:
            words        : List of recognized ASL signs
            user_message : Optional text message from user (typed)

        Returns:
            Structured dict with sentence, intent, urgency, suggestions, agent_message
        """
        if not words and not user_message:
            return self._empty_response()

        # ── Quick urgency check (local, no API call needed) ───────────────────
        urgency_result = execute_tool("detect_urgency", {"words": words})

        # ── Build prompt ──────────────────────────────────────────────────────
        history_str  = self._format_history()
        signs_str    = ", ".join(words) if words else "none"
        session_str  = ", ".join(self.session_signs[-20:]) if self.session_signs else "none"

        prompt = f"""{self._build_system_prompt()}

CONVERSATION HISTORY:
{history_str}

CURRENT INPUT:
- Signs recognized in this request: [{signs_str}]
- All signs in this session: [{session_str}]
- User typed message: {user_message or 'none'}
- Local urgency pre-check: {urgency_result['urgency_level']}

TASK:
Analyze the signs above. Apply your agent reasoning:
1. Form a natural English sentence from the signs
2. Infer the user's intent
3. Assess urgency level
4. Generate 2 suggestions for what they might sign next
5. Write a helpful, empathetic agent_message responding to the user

Return ONLY a valid JSON object as specified."""

        t0 = time.time()
        try:
            response     = self.model.generate_content(prompt)
            raw_text     = response.text.strip()
            latency_ms   = (time.time() - t0) * 1000

            # ── Clean response (remove markdown if present) ───────────────────
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            result = json.loads(raw_text)

            # ── Override urgency with local check if higher ───────────────────
            if urgency_result["urgency_level"] == "HIGH" and result.get("urgency") != "HIGH":
                result["urgency"]         = "HIGH"
                result["urgency_message"] = urgency_result["alert_message"]

            result["latency_ms"] = round(latency_ms, 1)
            result["signs_used"] = words

            # ── Update memory ─────────────────────────────────────────────────
            self.conversation_history.append({"role": "user",  "content": f"Signed: {signs_str}"})
            self.conversation_history.append({"role": "agent", "content": result.get("agent_message", "")})
            self.session_signs.extend(words)

            # Keep history bounded
            if len(self.conversation_history) > MAX_HISTORY * 2:
                self.conversation_history = self.conversation_history[-MAX_HISTORY * 2:]

            log.info(f"Agent response | urgency={result.get('urgency')} | latency={latency_ms:.0f}ms")
            return result

        except json.JSONDecodeError as e:
            log.error(f"JSON parse error: {e} | raw: {raw_text[:200]}")
            return self._fallback_response(words, raw_text)
        except Exception as e:
            log.error(f"Agent error: {e}")
            return self._error_response(str(e))

    def clear_session(self):
        """Reset conversation memory for a new session."""
        self.conversation_history = []
        self.session_signs        = []
        log.info("Agent session cleared.")

    def _empty_response(self) -> Dict:
        return {
            "sentence"       : "",
            "intent"         : "No signs provided",
            "urgency"        : "LOW",
            "urgency_message": None,
            "suggestions"    : ["hello", "help"],
            "agent_message"  : "I'm ready to help! Please perform an ASL sign in front of the camera.",
            "tools_used"     : [],
            "latency_ms"     : 0,
            "signs_used"     : []
        }

    def _fallback_response(self, words: List[str], raw: str) -> Dict:
        """Used when Gemini returns non-JSON."""
        return {
            "sentence"       : " ".join(w.replace("_", " ") for w in words),
            "intent"         : "Direct sign translation",
            "urgency"        : "LOW",
            "urgency_message": None,
            "suggestions"    : [],
            "agent_message"  : raw[:300] if raw else "I understood your signs.",
            "tools_used"     : ["fallback"],
            "latency_ms"     : 0,
            "signs_used"     : words
        }

    def _error_response(self, error: str) -> Dict:
        return {
            "sentence"       : "",
            "intent"         : "Agent error",
            "urgency"        : "LOW",
            "urgency_message": None,
            "suggestions"    : [],
            "agent_message"  : f"I encountered an error: {error}. Please try again.",
            "tools_used"     : [],
            "latency_ms"     : 0,
            "signs_used"     : [],
            "error"          : error
        }
