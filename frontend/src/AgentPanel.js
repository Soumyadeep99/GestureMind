import React, { useState, useRef, useEffect } from 'react';
import './AgentPanel.css';
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://3.85.135.107:8000';

// ─────────────────────────────────────────────────────────────────────────────
//  Voice Response — Browser Web Speech API (free, no backend needed)
// ─────────────────────────────────────────────────────────────────────────────
function speakText(text, { rate = 1.0, pitch = 1.0 } = {}) {
  if (!('speechSynthesis' in window) || !text) return;

  // Cancel any speech currently in progress before starting new one
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = rate;
  utterance.pitch = pitch;
  utterance.lang = 'en-US';

  // Prefer an English voice if multiple are available
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v => v.lang.startsWith('en')) || voices[0];
  if (preferred) utterance.voice = preferred;

  window.speechSynthesis.speak(utterance);
}

function stopSpeaking() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
}

export default function AgentPanel({ recognizedWords, isRunning, token }) {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [urgentAlert, setUrgentAlert] = useState(null);
  const [agentReady, setAgentReady] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true); // master toggle
  const chatEndRef = useRef(null);
  const lastWordsRef = useRef([]);

  // Check agent availability — retry every 5s
  useEffect(() => {
    const check = () => {
      fetch(`${BACKEND_URL}/health`)
        .then(r => r.json())
        .then(d => setAgentReady(d.agent_loaded))
        .catch(() => setAgentReady(false));
    };
    check();
    const t = setInterval(check, 5000);
    return () => clearInterval(t);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Track speaking state for UI indicator
  useEffect(() => {
    if (!('speechSynthesis' in window)) return;
    const check = setInterval(() => {
      setIsSpeaking(window.speechSynthesis.speaking);
    }, 200);
    return () => clearInterval(check);
  }, []);

  // Load voices (some browsers load them async)
  useEffect(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }
  }, []);

  // ── Auto-process when new words recognized ──────────────────────────────
  useEffect(() => {
    if (!isRunning || !agentReady) return;
    if (recognizedWords.length === 0) return;
    if (recognizedWords.length % 8 !== 0) return;
    const words = recognizedWords.map(w => w.text);
    if (JSON.stringify(words) === JSON.stringify(lastWordsRef.current)) return;
    lastWordsRef.current = words;
    processSigns(words, true);
  }, [recognizedWords, isRunning, agentReady]);

  // ── Process signs via agent ─────────────────────────────────────────────
  const processSigns = async (words, isAuto = false) => {
    if (!agentReady || isLoading) return;
    setIsLoading(true);

    if (!isAuto) {
      setMessages(prev => [...prev, {
        role: 'user',
        type: 'signs',
        content: words.map(w => w.replace(/_/g, ' ').toUpperCase()).join(' → '),
        timestamp: new Date()
      }]);
    }

    try {
      const res = await fetch(`${BACKEND_URL}/agent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ words })
    });
      const data = await res.json();

      // ── URGENCY ALERT + AUTO-SPEAK ────────────────────────────────────────
      if (data.urgency === 'HIGH') {
        setUrgentAlert(data.urgency_message);
        setTimeout(() => setUrgentAlert(null), 8000);

        // Auto-speak urgent messages — no button click needed
        if (voiceEnabled) {
          speakText(data.urgency_message || data.agent_message, { rate: 1.05 });
        }
      }

      setMessages(prev => [...prev, {
        role: 'agent',
        type: 'response',
        sentence: data.sentence,
        intent: data.intent,
        urgency: data.urgency,
        suggestions: data.suggestions || [],
        content: data.agent_message,
        latency: data.latency_ms,
        timestamp: new Date(),
        isAuto
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'agent',
        type: 'error',
        content: 'Agent unavailable. Check backend connection.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Chat message (typed) ────────────────────────────────────────────────
  const sendChatMessage = async () => {
    if (!inputText.trim() || isLoading) return;
    const msg = inputText.trim();
    setInputText('');
    setMessages(prev => [...prev, {
      role: 'user', type: 'text', content: msg, timestamp: new Date()
    }]);
    setIsLoading(true);
    try {
     const res = await fetch(`${BACKEND_URL}/agent/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        message: msg,
        current_words: currentWords
      })
    });
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'agent', type: 'chat', content: data.agent_message,
        sentence: data.sentence, timestamp: new Date(), latency: data.latency_ms
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'agent', type: 'error',
        content: 'Failed to reach agent.', timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeNow = () => {
    const words = recognizedWords.map(w => w.text);
    if (words.length > 0) processSigns(words);
    else alert('No signs recognized yet. Perform some ASL signs first.');
  };

  const clearSession = async () => {
    await fetch(`${BACKEND_URL}/agent/session`, {
  method: 'DELETE',
  headers: {
    Authorization: `Bearer ${token}`,
  },
}).catch(() => {});
    setMessages([]);
    lastWordsRef.current = [];
    stopSpeaking();
  };

  const fmtTime = (d) => d.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  });

  return (
    <div className="agent-panel">
      {/* Header */}
      <div className="ap-header">
        <div className="ap-title-row">
          <span className="ap-title">AI ASSISTANT</span>
          <div className="ap-header-right">
            {/* Voice toggle */}
            <button
              className={`ap-voice-toggle ${voiceEnabled ? 'on' : 'off'}`}
              onClick={() => { setVoiceEnabled(v => !v); if (voiceEnabled) stopSpeaking(); }}
              title={voiceEnabled ? 'Voice responses ON — click to mute' : 'Voice responses OFF — click to unmute'}
            >
              {voiceEnabled ? '🔊' : '🔇'}
            </button>
            <span className={`ap-status ${agentReady ? 'ready' : 'offline'}`}>
              <span className="ap-dot" />{agentReady ? 'AGENT READY' : 'AGENT OFFLINE'}
            </span>
          </div>
        </div>
        <div className="ap-subtitle">Gemini-powered sign language understanding</div>
      </div>

      {/* Urgency Alert */}
      {urgentAlert && (
        <div className="urgency-alert">
          <span className="urgency-icon">⚠️</span>
          <span>{urgentAlert}</span>
          {isSpeaking && <span className="urgency-speaking">🔊 Speaking...</span>}
        </div>
      )}

      {/* Quick actions */}
      <div className="ap-quick">
        <button className="ap-btn-analyze" onClick={handleAnalyzeNow} disabled={!agentReady || isLoading}>
          {isLoading ? <span className="ap-spinner" /> : '🧠'} Analyze Signs
        </button>
        <button className="ap-btn-clear" onClick={clearSession}>
          🗑 Clear
        </button>
      </div>

      {/* Chat messages */}
      <div className="ap-chat">
        {messages.length === 0 && (
          <div className="ap-empty">
            <div className="ap-empty-icon">🤟</div>
            <div className="ap-empty-text">
              {agentReady
                ? 'Perform ASL signs and click "Analyze Signs" — or type a message below.'
                : 'Set GEMINI_API_KEY in your .env file to enable the AI agent.'}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`msg ${msg.role}`}>
            {msg.role === 'user' && (
              <div className="msg-bubble user-bubble">
                {msg.type === 'signs' && <div className="msg-signs-label">🤟 Signs detected</div>}
                <div className="msg-text">{msg.content}</div>
                <div className="msg-time">{fmtTime(msg.timestamp)}</div>
              </div>
            )}

            {msg.role === 'agent' && msg.type !== 'error' && (
              <div className="msg-bubble agent-bubble">
                {msg.sentence && (
                  <div className="msg-sentence">
                    <span className="sentence-label">Sentence</span>
                    <span className="sentence-text">"{msg.sentence}"</span>
                  </div>
                )}
                <div className="msg-text">{msg.content}</div>
                {msg.suggestions && msg.suggestions.length > 0 && (
                  <div className="msg-suggestions">
                    <span className="sug-label">Next signs:</span>
                    {msg.suggestions.map((s, j) => (
                      <span key={j} className="sug-chip">{s.replace(/_/g, ' ')}</span>
                    ))}
                  </div>
                )}
                {msg.urgency && msg.urgency !== 'LOW' && (
                  <div className={`msg-urgency ${msg.urgency.toLowerCase()}`}>
                    {msg.urgency === 'HIGH' ? '🚨' : '⚠️'} Urgency: {msg.urgency}
                  </div>
                )}
                <div className="msg-footer">
                  <span className="msg-time">{fmtTime(msg.timestamp)}</span>
                  {msg.latency && <span className="msg-latency">{msg.latency}ms</span>}
                  {msg.isAuto && <span className="msg-auto">auto</span>}
                  {/* Manual speak button — reads this specific response aloud */}
                  <button
                    className="msg-speak-btn"
                    onClick={() => speakText(msg.sentence || msg.content)}
                    title="Speak this message"
                  >
                    🔊
                  </button>
                </div>
              </div>
            )}

            {msg.role === 'agent' && msg.type === 'error' && (
              <div className="msg-bubble error-bubble">
                <span>⚠️ {msg.content}</span>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="msg agent">
            <div className="msg-bubble agent-bubble loading">
              <div className="typing-dots"><span /><span /><span /></div>
              <span className="typing-label">Agent thinking...</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <div className="ap-input-row">
        <input
          className="ap-input"
          type="text"
          placeholder={agentReady ? "Ask the agent anything..." : "Agent offline"}
          value={inputText}
          disabled={!agentReady || isLoading}
          onChange={e => setInputText(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendChatMessage()}
        />
        <button className="ap-send-btn" onClick={sendChatMessage} disabled={!agentReady || isLoading || !inputText.trim()}>
          <SendIcon />
        </button>
      </div>
    </div>
  );
}

// ── Export for use in App.js "Speak Output" quick action ───────────────────
export { speakText, stopSpeaking };

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);