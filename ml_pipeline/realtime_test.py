"""
SignSense AI — Real-Time Local Test
====================================
Tests the FULL pipeline locally WITHOUT needing the FastAPI backend or React.

Webcam → MediaPipe → LSTM → Live prediction on screen

Run:
    cd "D:/Downloads/data_collection"
    python realtime_test.py
"""

import os
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration — must match training exactly
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR     = r"D:\Downloads\data_collection"
MODEL_PATH      = os.path.join(PROJECT_DIR, "trained_model", "signsense_model.keras")
LABELS_PATH     = os.path.join(PROJECT_DIR, "trained_model", "labels.json")

SEQUENCE_LENGTH     = 30      # Frames per prediction window
FEATURE_SIZE        = 1662    # Landmark features per frame
CONFIDENCE_THRESH   = 0.70    # Min confidence to show prediction
STABILITY_FRAMES    = 10      # How many of last N frames must agree
CAMERA_INDEX        = 0       # 0 = built-in webcam

# ─────────────────────────────────────────────────────────────────────────────
#  Load model + labels
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SignSense AI — Real-Time Test")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"Run 02_train_model.ipynb first."
    )

print("  Loading model...", end=" ", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
model.predict(np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE)), verbose=0)  # warm-up
print("done")

with open(LABELS_PATH, "r") as f:
    labels_data = json.load(f)
ASL_ACTIONS = labels_data["actions"]

print(f"  Classes: {ASL_ACTIONS}")
print(f"  Confidence threshold: {CONFIDENCE_THRESH}")
print()
print("  Controls:")
print("    Q → quit")
print("    C → clear displayed text")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────
mp_hands         = mp.solutions.hands
mp_pose          = mp.solutions.pose
mp_drawing       = mp.solutions.drawing_utils

HAND_DOT  = mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=4)
HAND_LINE = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2)
POSE_DOT  = mp_drawing.DrawingSpec(color=(255, 120, 30), thickness=2, circle_radius=3)
POSE_LINE = mp_drawing.DrawingSpec(color=(200, 80, 0),  thickness=2, circle_radius=2)


def detect_landmarks(frame_bgr, hand_det, pose_det):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hand_res = hand_det.process(rgb)
    pose_res = pose_det.process(rgb)
    rgb.flags.writeable = True
    return hand_res, pose_res


def draw_landmarks(frame, hand_results, pose_results):
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS, POSE_DOT, POSE_LINE
        )
    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lms,
                mp_hands.HAND_CONNECTIONS, HAND_DOT, HAND_LINE
            )
    return frame


def extract_keypoints(hand_results, pose_results):
    # Pose
    if pose_results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)

    face = np.zeros(468 * 3)  # Not used for ASL

    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_lms, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
            ).flatten()
            label = handedness.classification[0].label
            if label == "Left":
                lh = coords
            else:
                rh = coords

    return np.concatenate([pose, face, lh, rh])  # (1662,)


def count_hands(hand_results):
    return len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0


# ─────────────────────────────────────────────────────────────────────────────
#  UI drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_confidence_bar(frame, confidence, x, y, width=200, height=16):
    """Draw a horizontal confidence bar."""
    filled = int(width * confidence)
    color  = (0, 220, 80) if confidence >= CONFIDENCE_THRESH else (0, 100, 220)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (x + width + 8, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_ui(frame, gesture, confidence, is_confident, recognized_words,
            n_hands, fps, sequence_len, all_probs):
    h, w = frame.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 105), (8, 10, 22), -1)

    cv2.putText(frame, "SIGNSENSE AI",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 220, 180), 2)
    cv2.putText(frame, "REAL-TIME ASL RECOGNITION",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 180), 1)

    # Live dot
    dot_color = (0, 220, 80) if n_hands > 0 else (0, 60, 200)
    cv2.circle(frame, (w - 30, 30), 8, dot_color, -1)
    cv2.putText(frame, "LIVE" if n_hands > 0 else "NO HAND",
                (w - 100, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 100, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    # Buffer fill bar (shows when prediction will fire)
    buf_pct = sequence_len / SEQUENCE_LENGTH
    bx, by, bw2, bh = 12, 88, w - 24, 10
    cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh), (30, 30, 50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw2 * buf_pct), by + bh), (0, 120, 220), -1)
    cv2.putText(frame, f"Buffer: {sequence_len}/{SEQUENCE_LENGTH}",
                (bx, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 160), 1)

    # ── Current gesture (large center label) ─────────────────────────────────
    if is_confident and gesture != "...":
        label_color = (0, 255, 200)
        display     = gesture.replace("_", " ").upper()
    elif sequence_len == SEQUENCE_LENGTH:
        label_color = (80, 80, 200)
        display     = "..."
    else:
        label_color = (60, 60, 100)
        display     = "COLLECTING..."

    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(frame, display,
                (w // 2 - tw // 2, h // 2 + th // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, label_color, 3)

    # Confidence bar
    if sequence_len == SEQUENCE_LENGTH:
        draw_confidence_bar(frame, confidence,
                            x=w // 2 - 100, y=h // 2 + 30)

    # ── Right panel — top probabilities ───────────────────────────────────────
    if all_probs:
        panel_x = w - 230
        cv2.rectangle(frame, (panel_x - 10, 115), (w - 5, 115 + len(ASL_ACTIONS) * 30 + 10),
                      (10, 12, 25), -1)
        cv2.putText(frame, "CONFIDENCE",
                    (panel_x, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 140), 1)

        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (action, prob) in enumerate(sorted_probs):
            y_pos = 155 + i * 28
            bar_w = int(180 * prob)
            bar_color = (0, 200, 120) if action == gesture else (40, 80, 120)
            cv2.rectangle(frame, (panel_x, y_pos - 12),
                          (panel_x + bar_w, y_pos + 4), bar_color, -1)
            txt_color = (0, 255, 180) if action == gesture else (160, 160, 160)
            cv2.putText(frame, f"{action.replace('_',' '):<12} {prob*100:4.1f}%",
                        (panel_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_color, 1)

    # ── Bottom — recognized text output ──────────────────────────────────────
    panel_h = 100
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (8, 10, 22), -1)
    cv2.putText(frame, "RECOGNIZED TEXT",
                (12, h - panel_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 140), 1)

    # Show last 8 words
    words      = recognized_words[-8:]
    words_text = "  ".join(w.replace("_", " ").upper() for w in words)
    cv2.putText(frame, words_text if words_text else "—",
                (12, h - panel_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

    cv2.putText(frame, "Q = quit   C = clear text",
                (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# State
sequence          = deque(maxlen=SEQUENCE_LENGTH)  # Rolling 30-frame window
prediction_buffer = deque(maxlen=STABILITY_FRAMES) # Stability check buffer
recognized_words  = []                             # Text output list
last_gesture      = ""
last_added_time   = 0
ADD_COOLDOWN      = 1.5   # Seconds before same word can be added again

current_gesture   = "..."
current_conf      = 0.0
is_conf           = False
all_probs         = {}

fps_clock, fps_cnt, fps_val = time.time(), 0, 0.0

print("Camera opened. Perform ASL signs in front of the camera.")
print("Press Q to quit, C to clear text.")
print()

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hand_det, mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as pose_det:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hand_res, pose_res = detect_landmarks(frame, hand_det, pose_det)
        frame = draw_landmarks(frame, hand_res, pose_res)

        # ── Extract keypoints and fill sequence buffer ─────────────────────────
        keypoints = extract_keypoints(hand_res, pose_res)
        sequence.append(keypoints)

        # ── Run prediction when buffer is full ────────────────────────────────
        if len(sequence) == SEQUENCE_LENGTH:
            X    = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 1662)
            probs = model.predict(X, verbose=0)[0]

            pred_idx   = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            gesture    = ASL_ACTIONS[pred_idx]

            all_probs = {a: round(float(p), 4) for a, p in zip(ASL_ACTIONS, probs)}

            prediction_buffer.append(pred_idx)

            # Stability check — last N predictions must all agree
            if (len(prediction_buffer) == STABILITY_FRAMES and
                    len(set(prediction_buffer)) == 1 and
                    confidence >= CONFIDENCE_THRESH):

                current_gesture = gesture
                current_conf    = confidence
                is_conf         = True

                # Add to text output with cooldown
                now = time.time()
                if (gesture != last_gesture or now - last_added_time > ADD_COOLDOWN * 3):
                    if now - last_added_time > ADD_COOLDOWN:
                        recognized_words.append(gesture)
                        last_gesture    = gesture
                        last_added_time = now
                        print(f"  → {gesture.replace('_',' ').upper():<15} ({confidence*100:.1f}%)")
            else:
                current_gesture = "..."
                current_conf    = confidence
                is_conf         = False
        else:
            all_probs = {}

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_cnt += 1
        if fps_cnt >= 20:
            fps_val   = fps_cnt / (time.time() - fps_clock)
            fps_clock = time.time()
            fps_cnt   = 0

        # ── Draw UI ───────────────────────────────────────────────────────────
        frame = draw_ui(frame, current_gesture, current_conf, is_conf,
                        recognized_words, count_hands(hand_res),
                        fps_val, len(sequence), all_probs)

        cv2.imshow("SignSense AI — Real-Time ASL", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            recognized_words.clear()
            prediction_buffer.clear()
            print("  [Text cleared]")

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("  Session ended.")
print(f"  Words recognized : {len(recognized_words)}")
if recognized_words:
    print(f"  Text output      : {' '.join(w.upper() for w in recognized_words)}")
print("=" * 60)
