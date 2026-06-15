import argparse
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# =====================================================
# Settings
# =====================================================

VIDEO_PATH = "data/seunghyun.mp4"

RF_2D_MODEL_PATH = "rf_model.pkl"
RF_3D_MODEL_PATH = "rf_model_z.pkl"
XGB_3D_MODEL_PATH = "xgb_model_z.pkl"

OUTPUT_TXT = "evaluation_result_combined_seunghyun.txt"

THRESHOLDS = [0.4, 0.5, 0.6]

# (start_frame, end_frame, label)
# label 0: normal / label 1: turtle neck / label -1: ambiguous
LABEL_RANGES = [
    (0, 1008, 0),
    (1308, 2127, 1),
]


# =====================================================
# Ground Truth
# =====================================================

def parse_label_ranges(raw_ranges):
    label_ranges = []

    for raw_range in raw_ranges:
        parts = raw_range.split(",")

        if len(parts) != 3:
            raise ValueError(
                "Label ranges must use start,end,label format. "
                f"Invalid value: {raw_range}"
            )

        start, end, label = (int(part.strip()) for part in parts)
        label_ranges.append((start, end, label))

    return label_ranges


def get_gt_label(frame_idx, label_ranges):
    for start, end, label in label_ranges:
        if start <= frame_idx <= end:
            return label

    return -1


# =====================================================
# MediaPipe
# =====================================================

mp_pose = mp.solutions.pose


def create_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
    )


def get_landmarks(frame, pose):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark

    def to_xy(landmark):
        return np.array([landmark.x, landmark.y])

    def to_xyz(landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    return {
        "le_2d": to_xy(lm[mp_pose.PoseLandmark.LEFT_EAR]),
        "re_2d": to_xy(lm[mp_pose.PoseLandmark.RIGHT_EAR]),
        "ls_2d": to_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]),
        "rs_2d": to_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
        "nose_2d": to_xy(lm[mp_pose.PoseLandmark.NOSE]),
        "le_3d": to_xyz(lm[mp_pose.PoseLandmark.LEFT_EAR]),
        "re_3d": to_xyz(lm[mp_pose.PoseLandmark.RIGHT_EAR]),
        "ls_3d": to_xyz(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]),
        "rs_3d": to_xyz(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
        "nose_3d": to_xyz(lm[mp_pose.PoseLandmark.NOSE]),
    }


# =====================================================
# Feature Extraction
# =====================================================

def extract_2d_features(le, re, ls, rs, nose):
    ear = (le + re) / 2
    sh = (ls + rs) / 2
    vec = ear - sh

    dist_2d = np.linalg.norm(vec)

    vertical = np.array([0, -1])
    cos = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
    cos = np.clip(cos, -1, 1)
    angle = np.degrees(np.arccos(cos))

    nose_offset = nose - ear

    return np.array([
        vec[0],
        vec[1],
        dist_2d,
        angle,
        nose_offset[0],
        nose_offset[1],
    ], dtype=np.float32)


def extract_3d_features(le, re, ls, rs, nose):
    ear = (le + re) / 2
    sh = (ls + rs) / 2
    vec = ear - sh

    dist_2d = np.linalg.norm(vec[:2])
    dist_3d = np.linalg.norm(vec)

    vertical = np.array([0, -1])
    cos = np.dot(vec[:2], vertical) / (np.linalg.norm(vec[:2]) + 1e-6)
    cos = np.clip(cos, -1, 1)
    angle = np.degrees(np.arccos(cos))

    nose_offset = nose - ear
    depth_diff = ear[2] - sh[2]

    return np.array([
        vec[0],
        vec[1],
        dist_2d,
        dist_3d,
        angle,
        nose_offset[0],
        nose_offset[1],
        nose_offset[2],
        ear[2],
        sh[2],
        depth_diff,
    ], dtype=np.float32)


def extract_features(lm):
    features_2d = extract_2d_features(
        lm["le_2d"],
        lm["re_2d"],
        lm["ls_2d"],
        lm["rs_2d"],
        lm["nose_2d"],
    )

    features_3d = extract_3d_features(
        lm["le_3d"],
        lm["re_3d"],
        lm["ls_3d"],
        lm["rs_3d"],
        lm["nose_3d"],
    )

    return features_2d, features_3d


# =====================================================
# Evaluation
# =====================================================

def collect_video_features(video_path, label_ranges):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    y_true = []
    features_2d = []
    features_3d = []
    skipped_ambiguous = 0
    skipped_no_landmark = 0
    frame_idx = 0

    with create_pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            gt = get_gt_label(frame_idx, label_ranges)

            if gt == -1:
                skipped_ambiguous += 1
                frame_idx += 1
                continue

            frame_512 = cv2.resize(frame, (512, 512))
            lm = get_landmarks(frame_512, pose)

            if lm is None:
                skipped_no_landmark += 1
                frame_idx += 1
                continue

            feature_2d, feature_3d = extract_features(lm)

            y_true.append(gt)
            features_2d.append(feature_2d)
            features_3d.append(feature_3d)
            frame_idx += 1

    cap.release()

    return {
        "y_true": np.array(y_true),
        "features_2d": np.array(features_2d),
        "features_3d": np.array(features_3d),
        "total_frames": frame_idx,
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_no_landmark": skipped_no_landmark,
    }


def predict_with_threshold(model, features, threshold):
    probs = model.predict_proba(features)[:, 1]
    preds = (probs > threshold).astype(int)

    return probs, preds


def build_result_text(model_name, threshold, y_true, preds):
    lines = [
        "==============================",
        f"{model_name} RESULT - threshold {threshold:.1f}",
        "==============================",
        "",
        "Accuracy",
        f"{accuracy_score(y_true, preds)}",
        "",
        "Classification Report",
        classification_report(y_true, preds),
        "Confusion Matrix",
        str(confusion_matrix(y_true, preds)),
        "",
    ]

    return "\n".join(lines)


def evaluate_models(video_data, model_configs, thresholds):
    y_true = video_data["y_true"]
    output_blocks = []

    if len(y_true) == 0:
        raise ValueError("No evaluable frames found. Check video path and label ranges.")

    for config in model_configs:
        print(f"\nLoading {config['name']}: {config['path']}")
        model = joblib.load(config["path"])
        features = video_data[config["feature_key"]]

        for threshold in thresholds:
            _, preds = predict_with_threshold(model, features, threshold)
            result_text = build_result_text(
                config["name"],
                threshold,
                y_true,
                preds,
            )

            print("\n" + result_text)
            output_blocks.append(result_text)

    return "\n".join(output_blocks)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 2D RF, 3D RF, and 3D XGBoost posture models "
            "on one video with thresholds 0.4, 0.5, and 0.6."
        )
    )

    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--rf-2d-model", default=RF_2D_MODEL_PATH)
    parser.add_argument("--rf-3d-model", default=RF_3D_MODEL_PATH)
    parser.add_argument("--xgb-3d-model", default=XGB_3D_MODEL_PATH)
    parser.add_argument("--output", default=OUTPUT_TXT)
    parser.add_argument(
        "--label-range",
        action="append",
        dest="label_ranges",
        help=(
            "Frame label range in start,end,label format. "
            "Use multiple times for multiple ranges."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    label_ranges = (
        parse_label_ranges(args.label_ranges)
        if args.label_ranges
        else LABEL_RANGES
    )

    model_configs = [
        {
            "name": "2D RF",
            "path": args.rf_2d_model,
            "feature_key": "features_2d",
        },
        {
            "name": "3D RF",
            "path": args.rf_3d_model,
            "feature_key": "features_3d",
        },
        {
            "name": "3D XGBoost",
            "path": args.xgb_3d_model,
            "feature_key": "features_3d",
        },
    ]

    print("==============================")
    print("COMBINED EVALUATION")
    print("==============================")
    print(f"Video: {video_path}")
    print(f"Label ranges: {label_ranges}")
    print(f"Thresholds: {THRESHOLDS}")

    video_data = collect_video_features(video_path, label_ranges)

    print("\nFrame Summary")
    print(f"Total frames: {video_data['total_frames']}")
    print(f"Evaluated frames: {len(video_data['y_true'])}")
    print(f"Skipped ambiguous frames: {video_data['skipped_ambiguous']}")
    print(f"Skipped no-landmark frames: {video_data['skipped_no_landmark']}")

    result_text = evaluate_models(video_data, model_configs, THRESHOLDS)

    header_text = "\n".join([
        "==============================",
        "COMBINED EVALUATION",
        "==============================",
        "",
        f"Video: {video_path}",
        f"Label ranges: {label_ranges}",
        f"Thresholds: {THRESHOLDS}",
        "",
        "Frame Summary",
        f"Total frames: {video_data['total_frames']}",
        f"Evaluated frames: {len(video_data['y_true'])}",
        f"Skipped ambiguous frames: {video_data['skipped_ambiguous']}",
        f"Skipped no-landmark frames: {video_data['skipped_no_landmark']}",
        "",
    ])

    output_path.write_text(
        header_text + result_text,
        encoding="utf-8",
    )

    print(f"\nTXT saved: {output_path}")


if __name__ == "__main__":
    main()
