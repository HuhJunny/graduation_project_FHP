import cv2
import numpy as np
import torch
import depth_pro
from PIL import Image
import mediapipe as mp
import threading

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Depth Pro
# -----------------------------
model, transform = depth_pro.create_model_and_transforms()
model.eval()
model.to(device)

# -----------------------------
# MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# -----------------------------
# Shared
# -----------------------------
depth_map = None
latest_frame = None
latest_landmarks = None
depth_ready = False

lock = threading.Lock()

# -----------------------------
# Calibration
# -----------------------------
CALIB_FRAMES = 30
calib_buffer = []
baseline = None

# -----------------------------
# EMA
# -----------------------------
ema = None

TARGET = 50
THRESHOLD = 5

# -----------------------------
# Depth
# -----------------------------
def get_depth_map(frame_1536):
    image = Image.fromarray(cv2.cvtColor(frame_1536, cv2.COLOR_BGR2RGB))
    inputs = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(inputs)

    depth = prediction[0] if not isinstance(prediction, dict) else prediction["depth"]
    return depth.squeeze().cpu().numpy()

def depth_worker():
    global depth_map, latest_frame, latest_landmarks, depth_ready

    while True:
        with lock:
            frame_copy = None if latest_frame is None else latest_frame.copy()
            lm_copy = None if latest_landmarks is None else latest_landmarks.copy()

        if frame_copy is None or lm_copy is None:
            continue

        frame_1536 = cv2.resize(frame_copy, (1536, 1536))
        new_depth = get_depth_map(frame_1536)

        with lock:
            depth_map = new_depth
            latest_landmarks = lm_copy
            depth_ready = True 

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
# Depth Sampling
# -----------------------------
def sample_depth(pt, depth_map, scale):
    x = int(pt[0] * 512 * scale)
    y = int(pt[1] * 512 * scale)

    h, w = depth_map.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    return depth_map[y, x]

# -----------------------------
# Alignment
# -----------------------------
def align_depth_to_blaze(lm, depth_map, scale):
    keys = ["ls", "rs", "le", "re"]

    blaze_z = []
    depth_z = []

    for k in keys:
        pt = lm[k]
        d = sample_depth(pt[:2], depth_map, scale)

        blaze_z.append(pt[2])
        depth_z.append(d)

    blaze_z = np.array(blaze_z)
    depth_z = np.array(depth_z)

    d_mean = depth_z.mean()
    b_mean = blaze_z.mean()

    d_std = depth_z.std() + 1e-6
    b_std = blaze_z.std() + 1e-6

    a = b_std / d_std
    b = b_mean - a * d_mean

    return a, b

# -----------------------------
# Angle (pseudo-3D)
# -----------------------------
def compute_angle(lm, depth_map, scale):
    a, b = align_depth_to_blaze(lm, depth_map, scale)

    def fuse(pt):
        d = sample_depth(pt[:2], depth_map, scale)
        z = a * d + b
        return np.array([pt[0], pt[1], 0.7 * pt[2] + 0.3 * z])

    le = fuse(lm["le"])
    re = fuse(lm["re"])
    ls = fuse(lm["ls"])
    rs = fuse(lm["rs"])

    M = (le + re) / 2
    S = rs - ls
    S0 = ls

    if np.dot(S, S) == 0:
        return None

    Mprime = S0 + (np.dot(M - S0, S) / np.dot(S, S)) * S
    V = np.array([M[0], Mprime[1], 1.0])

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
        ema = 0.3 * angle + 0.7 * ema

    if baseline is None:
        calib_buffer.append(ema)
        if len(calib_buffer) >= CALIB_FRAMES:
            baseline = np.mean(calib_buffer)
        return None, None

    delta = ema - baseline
    corrected = TARGET + delta * 1.5
    is_turtle = corrected > (TARGET + THRESHOLD)

    return corrected, is_turtle

# -----------------------------
# Thread
# -----------------------------
threading.Thread(target=depth_worker, daemon=True).start()

# -----------------------------
# Main
# -----------------------------
cap = cv2.VideoCapture(0)
scale = 1536 / 512

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_512 = cv2.resize(frame, (512, 512))
    lm = get_landmarks(frame_512)

    with lock:
        latest_frame = frame.copy()
        if lm is not None:
            latest_landmarks = lm

        current_depth = None if depth_map is None else depth_map.copy()
        ready = depth_ready

    if ready and current_depth is not None and latest_landmarks is not None:
        with lock:
            depth_ready = False

        angle = compute_angle(latest_landmarks, current_depth, scale)
        corrected, is_turtle = process(angle)

        print(f"[DEPTH SYNC] angle: {angle:.2f} | corrected: {corrected}")

    cv2.imshow("Webcam", frame_512)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()