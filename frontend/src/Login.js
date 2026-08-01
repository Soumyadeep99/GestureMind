import React, { useState } from 'react';
import './Login.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://3.85.135.107:8000';

export default function Login({ onLogin }) {
  const [mode, setMode]         = useState('login'); // 'login' | 'signup'
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!email || !password) {
      setError('Please fill in both fields.');
      return;
    }
    if (mode === 'signup' && password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }

    setLoading(true);
    try {
      const endpoint = mode === 'login' ? '/auth/login' : '/auth/register';
      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || 'Something went wrong. Please try again.');
        setLoading(false);
        return;
      }

      localStorage.setItem('gm_token', data.access_token);
      localStorage.setItem('gm_email', data.email);
      onLogin(data.access_token, data.email);
    } catch (err) {
      setError('Cannot reach the server. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">
          <svg width="44" height="44" viewBox="0 0 36 36" fill="none">
            <circle cx="18" cy="18" r="17" stroke="#00dfa2" strokeWidth="1.5"/>
            <path d="M12 24L12 14L15 14L15 18L18 18L18 14L21 14L21 24"
              stroke="#00dfa2" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            <circle cx="24" cy="16" r="2" fill="#00dfa2"/>
          </svg>
        </div>
        <h1 className="login-title">GestureMind</h1>
        <p className="login-subtitle">
          {mode === 'login' ? 'Sign in to continue' : 'Create your account'}
        </p>

        <form onSubmit={handleSubmit} className="login-form">
          <label className="login-label">Email</label>
          <input
            type="email"
            className="login-input"
            placeholder="you@example.com"
            value={email}
            onChange={e => setEmail(e.target.value)}
            autoComplete="email"
          />

          <label className="login-label">Password</label>
          <input
            type="password"
            className="login-input"
            placeholder={mode === 'signup' ? 'At least 6 characters' : '••••••••'}
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          />

          {error && <div className="login-error">⚠️ {error}</div>}

          <button type="submit" className="login-submit-btn" disabled={loading}>
            {loading ? 'Please wait...' : (mode === 'login' ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div className="login-switch">
          {mode === 'login' ? (
            <>Don't have an account?{' '}
              <button className="login-switch-btn" onClick={() => { setMode('signup'); setError(''); }}>
                Sign up
              </button>
            </>
          ) : (
            <>Already have an account?{' '}
              <button className="login-switch-btn" onClick={() => { setMode('login'); setError(''); }}>
                Sign in
              </button>
            </>
          )}
        </div>

        <p className="login-note">
          🚨 Your emergency contacts are tied to your account — set them up
          under Settings after logging in, so urgency alerts reach the right person.
        </p>
      </div>
    </div>
  );
}
