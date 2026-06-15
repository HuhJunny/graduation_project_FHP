import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1. CSV 병합
# -----------------------------
DATA_DIR = "data"

files = [
    "labeled_landmarks_1.csv",
    "labeled_landmarks_2.csv",
    "labeled_landmarks_3.csv",
    "labeled_landmarks_4.csv",
    "labeled_landmarks_5.csv",
    "labeled_landmarks_6.csv",
    "labeled_landmarks_7.csv",
    "labeled_landmarks_8.csv",
    "labeled_landmarks_9.csv",
    "labeled_landmarks_10.csv",
    "labeled_landmarks_11.csv",
    "labeled_landmarks_12.csv",
    "labeled_landmarks_ha_1.csv",
    "labeled_landmarks_ha_2.csv",
    "labeled_landmarks_seungmin_0.csv",
    "labeled_landmarks_seungmin_1.csv",
    "labeled_landmarks_mingyu_0.csv",
    "labeled_landmarks_mingyu_1.csv"
]

dfs = []
for f in files:
    path = os.path.join(DATA_DIR, f)
    print(f"Loading: {path}")
    dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True)
print("총 데이터 수:", len(df))


# -----------------------------
# 2. Feature 추출 (2D)
# -----------------------------
def get_xy(row, idx):
    return np.array([row[f"x{idx}"], row[f"y{idx}"]])


X = []
y = []

for _, row in df.iterrows():

    le = get_xy(row, 7)
    re = get_xy(row, 8)
    ls = get_xy(row, 11)
    rs = get_xy(row, 12)
    nose = get_xy(row, 0)

    ear = (le + re) / 2
    sh = (ls + rs) / 2

    vec = ear - sh

    dist_2d = np.linalg.norm(vec)

    vertical = np.array([0, -1])
    cos = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
    cos = np.clip(cos, -1, 1)
    angle = np.degrees(np.arccos(cos))

    nose_offset = nose - ear

    features = [
        vec[0], vec[1],
        dist_2d,
        angle,
        nose_offset[0],
        nose_offset[1]
    ]

    X.append(features)
    y.append(row["label"])


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# 3. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. RF 학습
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

# -----------------------------
# 5. 평가
# -----------------------------
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_names = [
    "vec_x",
    "vec_y",
    "dist_2d",
    "angle",
    "nose_offset_x",
    "nose_offset_y"
]

importance = rf.feature_importances_

print("\n=== Feature Importance ===")

for name, score in sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {score:.4f}")


# -----------------------------
# 6. 모델 저장
# -----------------------------
joblib.dump(rf, "rf_model.pkl")

print("모델 저장 완료: rf_model.pkl")