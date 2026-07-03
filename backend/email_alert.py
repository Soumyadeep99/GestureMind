"""
GestureMind — Emergency Email Alert Module
=============================================
Sends an email alert to a designated contact (parent/friend/caregiver)
when the agent detects HIGH urgency (e.g. "help" sign detected).

Uses Gmail SMTP with an App Password — no paid service required.
"""

import os
import smtplib
import logging
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

log = logging.getLogger("gesturemind.email")

# ─────────────────────────────────────────────────────────────────────────────
class EmailAlertService:
    """
    Handles sending urgency alert emails via Gmail SMTP.

    Includes cooldown logic to prevent spamming the recipient if
    urgency is detected repeatedly in a short time window.
    """

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT   = 587
    COOLDOWN_SECONDS = 120  # Don't send more than 1 alert per 2 minutes

    def __init__(self, sender_email: str, app_password: str, recipients: List[str]):
        self.sender_email  = sender_email
        self.app_password  = app_password
        self.recipients    = [r.strip() for r in recipients if r.strip()]
        self.last_sent_at  = 0
        self.enabled       = bool(sender_email and app_password and self.recipients)

        if self.enabled:
            log.info(f"Email alerts enabled. Recipients: {self.recipients}")
        else:
            log.warning("Email alerts disabled — missing sender, password, or recipients.")

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_sent_at) < self.COOLDOWN_SECONDS

    def send_urgency_alert(
        self,
        urgency_message: str,
        signs_detected: List[str],
        sentence: Optional[str] = None
    ) -> dict:
        """
        Sends an urgency alert email.

        Returns:
            dict with 'sent' (bool) and 'reason' (str) explaining outcome.
        """
        if not self.enabled:
            return {"sent": False, "reason": "Email alerts not configured."}

        if self._in_cooldown():
            remaining = int(self.COOLDOWN_SECONDS - (time.time() - self.last_sent_at))
            return {"sent": False, "reason": f"Cooldown active ({remaining}s remaining)."}

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "🚨 GestureMind Alert — User May Need Help"
            msg["From"]    = self.sender_email
            msg["To"]      = ", ".join(self.recipients)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            signs_str = ", ".join(signs_detected) if signs_detected else "unknown"

            text_body = f"""
GestureMind Emergency Alert

Time: {timestamp}
Urgency Message: {urgency_message}
Signs Detected: {signs_str}
Interpreted Sentence: {sentence or 'N/A'}

This is an automated alert from GestureMind — a real-time ASL
recognition system. The user may need assistance.

Please check on them as soon as possible.

---
This is an automated message from GestureMind AI.
"""

            html_body = f"""
<html>
  <body style="font-family: Arial, sans-serif; background:#0d0f1a; padding:24px;">
    <div style="max-width:480px; margin:0 auto; background:#0f1221; border:1px solid #f87171; border-radius:10px; overflow:hidden;">
      <div style="background:#f87171; padding:16px 20px;">
        <h2 style="margin:0; color:#0d0f1a; font-size:18px;">🚨 GestureMind Emergency Alert</h2>
      </div>
      <div style="padding:20px; color:#e8eaf0;">
        <p style="font-size:14px; line-height:1.6;">
          <strong>{urgency_message}</strong>
        </p>
        <table style="width:100%; margin-top:16px; font-size:13px; color:#8892aa;">
          <tr><td style="padding:6px 0;">Time</td><td style="color:#e8eaf0;">{timestamp}</td></tr>
          <tr><td style="padding:6px 0;">Signs Detected</td><td style="color:#e8eaf0;">{signs_str}</td></tr>
          <tr><td style="padding:6px 0;">Interpreted</td><td style="color:#e8eaf0;">{sentence or 'N/A'}</td></tr>
        </table>
        <p style="font-size:13px; color:#8892aa; margin-top:20px;">
          Please check on them as soon as possible.
        </p>
      </div>
      <div style="background:#080a16; padding:12px 20px; font-size:11px; color:#4a5268;">
        Automated message from GestureMind AI — Real-Time ASL Recognition
      </div>
    </div>
  </body>
</html>
"""

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, self.recipients, msg.as_string())

            self.last_sent_at = time.time()
            log.info(f"Urgency alert email sent to {self.recipients}")
            return {"sent": True, "reason": f"Alert sent to {len(self.recipients)} recipient(s)."}

        except smtplib.SMTPAuthenticationError:
            log.error("SMTP auth failed — check Gmail App Password.")
            return {"sent": False, "reason": "Authentication failed. Check GMAIL_APP_PASSWORD in .env"}
        except Exception as e:
            log.error(f"Email send failed: {e}")
            return {"sent": False, "reason": str(e)}
