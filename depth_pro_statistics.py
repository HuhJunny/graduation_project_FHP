import cv2
import numpy as np
import torch
import depth_pro
from PIL import Image
import mediapipe as mp
import time

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
# Config
# -----------------------------
BASELINE_SECONDS = 3
DEPTH_INTERVAL = 5

# -----------------------------
# Statistics
# -----------------------------
baseline_values = []
delta_history = []

baseline_offset = None

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

# -----------------------------
# Landmark
# -----------------------------
def get_landmarks(frame_512):
    rgb = cv2.cvtColor(frame_512, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark

    def to_xy(l):
        return np.array([l.x, l.y])

    return {
        "le": to_xy(lm[mp_pose.PoseLandmark.LEFT_EAR]),
        "re": to_xy(lm[mp_pose.PoseLandmark.RIGHT_EAR]),
        "ls": to_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]),
        "rs": to_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
        "nose": to_xy(lm[mp_pose.PoseLandmark.NOSE]),
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
# Compute Offset
# -----------------------------
def compute_forward_offset(lm, depth_map, scale):

    le = sample_depth(lm["le"], depth_map, scale)
    re = sample_depth(lm["re"], depth_map, scale)

    ls = sample_depth(lm["ls"], depth_map, scale)
    rs = sample_depth(lm["rs"], depth_map, scale)

    ear_depth = (le + re) / 2
    shoulder_depth = (ls + rs) / 2

    # 귀가 앞으로 올수록 증가
    offset = shoulder_depth - ear_depth

    return float(offset)

# -----------------------------
# Main
# -----------------------------
cap = cv2.VideoCapture(0)

scale = 1536 / 512

start_time = time.time()
last_depth_time = 0

print("정상 자세를 유지해주세요(Baseline 측정 중)")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_512 = cv2.resize(frame, (512, 512))

    lm = get_landmarks(frame_512)

    if lm is None:
        continue

    current_time = time.time()

    # 5초마다 실행
    if current_time - last_depth_time >= DEPTH_INTERVAL:

        last_depth_time = current_time

        frame_1536 = cv2.resize(frame, (1536, 1536))

        depth_map = get_depth_map(frame_1536)

        offset = compute_forward_offset(lm, depth_map, scale)

        elapsed = current_time - start_time

        # -----------------------------
        # Baseline
        # -----------------------------
        if elapsed <= BASELINE_SECONDS:

            baseline_values.append(offset)

            print(f"[BASELINE] offset: {offset:.4f}")

        else:

            # baseline 생성
            if baseline_offset is None:
                baseline_offset = np.mean(baseline_values)

                print("\n===== BASELINE FIXED =====")
                print(f"Baseline offset: {baseline_offset:.4f}")
                print("==========================\n")

            delta = offset - baseline_offset

            delta_history.append(delta)

            print(f"[MEASURE]")
            print(f"Current offset : {offset:.4f}")
            print(f"Delta           : {delta:.4f}")

    cv2.imshow("Webcam", frame_512)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Final Statistics
# -----------------------------
print("\n==============================")
print("SESSION RESULT")
print("==============================")

if len(delta_history) > 0:

    avg_delta = np.mean(delta_history)
    max_delta = np.max(delta_history)
    min_delta = np.min(delta_history)

    print(f"Average Forward Delta : {avg_delta:.4f}")
    print(f"Maximum Forward Delta : {max_delta:.4f}")
    print(f"Minimum Forward Delta : {min_delta:.4f}")

else:
    print("No measurement data.")

cap.release()
cv2.destroyAllWindows()