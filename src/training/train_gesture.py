import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import json


GESTURE_LABELS = {
    "OPEN_PALM": 0, "FIST": 1, "THUMBS_UP": 2,
    "POINT": 3, "PEACE": 4,
}
ID_TO_GESTURE = {v: k for k, v in GESTURE_LABELS.items()}
WINDOW_SIZE = (64, 64)


def extract_hog_features(img_list, hog):
    features = []
    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        resized = cv2.resize(gray, WINDOW_SIZE)
        feat = hog.compute(resized).ravel()
        features.append(feat)
    return np.array(features)


def load_gesture_data(data_dir: str):
    X, y = [], []
    hog = cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    for gesture_name, label in GESTURE_LABELS.items():
        gesture_dir = os.path.join(data_dir, gesture_name)
        if not os.path.isdir(gesture_dir):
            print(f"  Warning: {gesture_dir} not found, skipping")
            continue

        images = []
        for f in os.listdir(gesture_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(gesture_dir, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)

        if not images:
            print(f"  Warning: no images found for {gesture_name}")
            continue

        features = extract_hog_features(images, hog)
        X.append(features)
        y.extend([label] * len(images))
        print(f"  {gesture_name}: {len(images)} samples")

    if not X:
        raise ValueError("No gesture data found")

    return np.vstack(X), np.array(y)


def train(data_dir: str, output_dir: str = "models"):
    print(f"Loading gesture data from {data_dir}...")
    X, y = load_gesture_data(data_dir)
    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training SVM with GridSearchCV...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf'],
    }
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    best_svm = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    val_pred = best_svm.predict(X_val_scaled)
    val_acc = np.mean(val_pred == y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_val, val_pred, target_names=list(GESTURE_LABELS.keys())
    ))

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_svm, os.path.join(output_dir, "gesture_svm.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "gesture_scaler.pkl"))
    print(f"Model saved to {output_dir}/")

    results = {
        "best_params": grid.best_params_,
        "cv_accuracy": float(grid.best_score_),
        "validation_accuracy": float(val_acc),
        "samples": len(X),
    }
    with open(os.path.join(output_dir, "gesture_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gesture/hand_crops")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    train(args.data, args.output)
