import os
import cv2
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json


WINDOW_SIZE = (64, 64)


def extract_hog_features(img_list, hog):
    features = []
    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        resized = cv2.resize(gray, WINDOW_SIZE)
        feat = hog.compute(resized).ravel()
        features.append(feat)
    return np.array(features)


def load_face_images(data_dir: str) -> list:
    images = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(root, f)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
    print(f"Loaded {len(images)} face images from {data_dir}")
    return images


def generate_negatives(neg_dir: str, num_patches: int,
                       img_size: tuple = (200, 200)) -> list:
    negatives = []
    if os.path.isdir(neg_dir):
        for root, dirs, files in os.walk(neg_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    path = os.path.join(root, f)
                    img = cv2.imread(path)
                    if img is not None:
                        negatives.append(img)

    crops = []
    for img in negatives:
        h, w = img.shape[:2]
        if h < img_size[0] or w < img_size[1]:
            continue
        for _ in range(max(1, num_patches // len(negatives))):
            x = np.random.randint(0, w - img_size[1])
            y = np.random.randint(0, h - img_size[0])
            crop = img[y:y + img_size[0], x:x + img_size[1]]
            crops.append(crop)

    print(f"Generated {len(crops)} negative patches from {neg_dir}")
    return crops


def train(face_dir: str, neg_dir: str, output_dir: str = "models",
          num_neg_per_image: int = 5, test_size: float = 0.2):
    hog = cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    pos_images = load_face_images(face_dir)
    neg_patches = generate_negatives(neg_dir, num_neg_per_image)

    if not pos_images or not neg_patches:
        print("Need at least one face and one non-face image directory")
        return

    print("Extracting HOG features from positive samples...")
    X_pos = extract_hog_features(pos_images, hog)
    y_pos = np.ones(len(X_pos))

    print("Extracting HOG features from negative samples...")
    X_neg = extract_hog_features(neg_patches, hog)
    y_neg = np.zeros(len(X_neg))

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])

    print(f"Feature matrix: {X.shape}")
    print(f"Positive: {len(X_pos)}, Negative: {len(X_neg)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training LinearSVM...")
    svm = LinearSVC(C=1.0, class_weight='balanced',
                    max_iter=10000, random_state=42)
    svm.fit(X_train_scaled, y_train)

    val_pred = svm.predict(X_val_scaled)
    val_acc = np.mean(val_pred == y_val)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred,
                                target_names=['non-face', 'face']))

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(svm, os.path.join(output_dir, "face_svm.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "face_scaler.pkl"))
    print(f"Models saved to {output_dir}/")

    results = {
        "validation_accuracy": float(val_acc),
        "positive_samples": len(X_pos),
        "negative_samples": len(X_neg),
        "feature_dim": X.shape[1],
    }
    with open(os.path.join(output_dir, "face_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-dir", required=True,
                        help="Directory of face images (positive)")
    parser.add_argument("--neg-dir", required=True,
                        help="Directory of non-face images (negative)")
    parser.add_argument("--output", default="models")
    parser.add_argument("--num-neg", type=int, default=5,
                        help="Negative patches per image")
    args = parser.parse_args()
    train(args.face_dir, args.neg_dir, args.output, args.num_neg)
