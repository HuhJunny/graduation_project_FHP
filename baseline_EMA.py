import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# -----------------------------
# Calibration
# -----------------------------
CALIB_FRAMES = 60
calib_buffer = []
baseline = None

# -----------------------------
# EMA
# -----------------------------
ema = None

TARGET = 50
THRESHOLD = 5

# -----------------------------
# Landmark
# -----------------------------
def get_landmarks(frame_512):
    rgb = cv2.cvtColor(frame_512, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark

    def to_xyz(l):
        return np.array([l.x, l.y, l.z])

    return {
        "le": to_xyz(lm[mp_pose.PoseLandmark.LEFT_EAR]),
        "re": to_xyz(lm[mp_pose.PoseLandmark.RIGHT_EAR]),
        "ls": to_xyz(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]),
        "rs": to_xyz(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
    }

# -----------------------------
# Angle (Depth 제거 버전)
# -----------------------------
def compute_angle(ear_left, ear_right, shoulder_left, shoulder_right):
    M = (ear_left + ear_right) / 2
    S = shoulder_right - shoulder_left
    S0 = shoulder_left

    MS0 = M - S0
    dot_MS0_S = np.dot(MS0, S)
    S_len2 = np.dot(S, S)

    if S_len2 == 0:
        return None

    Mprime = S0 + (dot_MS0_S / S_len2) * S

    V = np.array([M[0], Mprime[1], M[2]])

    n = np.cross(S, V - Mprime)
    MMp = Mprime - M

    if np.linalg.norm(MMp) == 0 or np.linalg.norm(n) == 0:
        return None

    dot = np.dot(MMp, n)

    angle = np.pi / 2 - np.arccos(
        np.clip(abs(dot) / (np.linalg.norm(MMp)*np.linalg.norm(n)), -1, 1)
    )

    return float(np.degrees(angle))

# -----------------------------
# Process
# -----------------------------
def process(angle):
    global ema, baseline, calib_buffer

    if ema is None:
        ema = angle
    else:
        if abs(angle - ema) < 20:
            ema = 0.3 * angle + 0.7 * ema

    if baseline is None:
        calib_buffer.append(ema)

        progress = len(calib_buffer) / CALIB_FRAMES

        if len(calib_buffer) >= CALIB_FRAMES:
            baseline = np.mean(calib_buffer)

        return None, None, progress

    delta = ema - baseline
    corrected = TARGET + delta * 1.5

    is_turtle = corrected < (TARGET - THRESHOLD)

    return corrected, is_turtle, None

# -----------------------------
# Main
# -----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_512 = cv2.resize(frame, (512, 512))

    lm = get_landmarks(frame_512)
    if lm is None:
        continue

    angle = compute_angle(
        lm["le"], lm["re"],
        lm["ls"], lm["rs"]
    )

    if angle is None:
        continue
    

    corrected, is_turtle, progress = process(angle)
    

    # -----------------------------
    # UI
    # -----------------------------
    if baseline is None:
        percent = int(progress * 100)
        cv2.putText(frame_512, f"Calibrating... {percent}%",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

    else:
        if corrected is None:
            cv2.putText(frame_512, "Initializing...",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
        else:
            text = "TURTLE" if is_turtle else "NORMAL"
            color = (0,0,255) if is_turtle else (0,255,0)

            cv2.putText(frame_512, f"{text} ({corrected:.1f})",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Webcam", frame_512)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()