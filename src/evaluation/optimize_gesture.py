import os
import sys
import cv2
import json
import csv
import numpy as np
import joblib
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.train_gesture_hagrid import (
    GESTURE_LABELS, ID_TO_GESTURE, HOG_FEATURE_DIM, WINDOW_SIZE,
)

HOG_DEFAULT = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

HOG_FINER = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(4, 4),
    _nbins=9,
)

HAGRID_ROOT = "/home/hazedawn/Documents/CV Project Rev2/Data/hagrid-sample-30k-384p"


def extract_hog_from_roi(roi_bgr, hog):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, WINDOW_SIZE)
    return hog.compute(resized).ravel()


def load_hagrid_data_with_aug(hagrid_root, hog=HOG_DEFAULT, add_margin=0.15,
                              augment_flip=True):
    ann_dir = os.path.join(hagrid_root, "ann_train_val")
    img_base = os.path.join(hagrid_root, "hagrid_30k")

    hagrid_map = {
        "palm": "OPEN_PALM", "fist": "FIST", "like": "THUMBS_UP",
        "one": "POINT", "peace": "PEACE",
    }

    all_features = []
    all_labels = []

    for hagrid_name, our_name in sorted(hagrid_map.items()):
        ann_path = os.path.join(ann_dir, f"{hagrid_name}.json")
        img_dir = os.path.join(img_base, f"train_val_{hagrid_name}")
        if not os.path.exists(ann_path) or not os.path.isdir(img_dir):
            continue

        with open(ann_path) as f:
            annotations = json.load(f)

        label = GESTURE_LABELS[our_name]
        count = 0

        for key, entry in annotations.items():
            img_path = os.path.join(img_dir, f"{key}.jpg")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            for bbox, bbox_label in zip(entry["bboxes"], entry["labels"]):
                if bbox_label == "no_gesture" or bbox_label != hagrid_name:
                    continue

                xc, yc, bw_n, bh_n = bbox
                box_w = int(bw_n * w)
                box_h = int(bh_n * h)
                margin_x = int(box_w * add_margin)
                margin_y = int(box_h * add_margin)

                x1 = max(0, int((xc - bw_n / 2) * w) - margin_x)
                y1 = max(0, int((yc - bh_n / 2) * h) - margin_y)
                x2 = min(w, int((xc + bw_n / 2) * w) + margin_x)
                y2 = min(h, int((yc + bh_n / 2) * h) + margin_y)

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                try:
                    feat = extract_hog_from_roi(roi, hog)
                    if len(feat) > 0:
                        all_features.append(feat)
                        all_labels.append(label)
                        count += 1

                        if augment_flip:
                            flipped = cv2.flip(roi, 1)
                            feat_f = extract_hog_from_roi(flipped, hog)
                            if len(feat_f) > 0:
                                all_features.append(feat_f)
                                all_labels.append(label)
                                count += 1
                except Exception:
                    continue

        print(f"  {hagrid_name} -> {our_name}: {count} samples (with aug)")

    return np.array(all_features), np.array(all_labels)


def load_existing_csv(csv_path: str):
    X, y = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gesture = row["gesture"]
            if gesture not in GESTURE_LABELS:
                continue
            feat = np.array([float(row[f"f{i}"]) for i in range(HOG_FEATURE_DIM)])
            X.append(feat)
            y.append(GESTURE_LABELS[gesture])
    return np.array(X), np.array(y)


def try_config(name, features_fn, pca_components, svm_params=None, model_type="svm",
               **train_kwargs):
    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    print(f"{'='*60}")

    X_h, y_h = load_hagrid_data_with_aug(HAGRID_ROOT, **train_kwargs)
    csv_path = train_kwargs.get("csv_path",
        "/home/hazedawn/Documents/CV Project Rev2/NeuraCam Repo/data/gesture/raw/features_2500.csv")
    if os.path.exists(csv_path):
        X_ex, y_ex = load_existing_csv(csv_path)
        X = np.vstack([X_h, X_ex])
        y = np.concatenate([y_h, y_ex])
    else:
        X, y = X_h, y_h

    print(f"Total: {len(X)} samples")
    print(f"Class distribution: {dict(sorted(Counter(y).items()))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n_comp = min(pca_components, X_train.shape[1], len(X_train) - 1)
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    var_retained = float(pca.explained_variance_ratio_.sum())
    print(f"PCA: {X.shape[1]} -> {n_comp} components (variance: {var_retained:.3f})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_pca)
    X_test_s = scaler.transform(X_test_pca)

    if svm_params:
        svm = SVC(probability=True, random_state=42, **svm_params)
        svm.fit(X_train_s, y_train)
        best_params = svm_params
        cv_acc = 0.0
    else:
        grid = GridSearchCV(
            SVC(probability=True, random_state=42),
            {'C': [1, 10, 100], 'gamma': ['scale', 0.01], 'kernel': ['rbf']},
            cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid.fit(X_train_s, y_train)
        svm = grid.best_estimator_
        best_params = grid.best_params_
        cv_acc = float(grid.best_score_)

    pred = svm.predict(X_test_s)
    acc = float(np.mean(pred == y_test))

    print(f"Params: {best_params}")
    if cv_acc:
        print(f"CV accuracy: {cv_acc:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Support vectors: {len(svm.support_)}")

    report = classification_report(y_test, pred, target_names=list(GESTURE_LABELS.keys()),
                                   output_dict=True)
    for cls_name in GESTURE_LABELS:
        r = report[cls_name]
        print(f"  {cls_name:<15} prec={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f}")

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)

    return {
        "name": name,
        "accuracy": acc,
        "cv_accuracy": cv_acc,
        "best_params": best_params,
        "support_vectors": len(svm.support_),
        "pca_components": n_comp,
        "pca_variance": var_retained,
        "samples": len(X),
        "model": svm,
        "scaler": scaler,
        "pca": pca,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run only quick configs")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)

    results = []

    # Baseline: current config
    r = try_config(
        "baseline (PCA=80, HOG default, no margin, no aug)",
        load_hagrid_data_with_aug, pca_components=80,
        add_margin=0.0, augment_flip=False,
    )
    results.append(r)

    # Test 1: More PCA components
    for pca_n in [200, 400, 800]:
        r = try_config(
            f"PCA={pca_n}",
            load_hagrid_data_with_aug, pca_components=pca_n,
            add_margin=0.0, augment_flip=False,
        )
        results.append(r)
        if args.fast:
            break

    # Test 2: ROI margin
    r = try_config(
        "margin=0.15, PCA=200",
        load_hagrid_data_with_aug, pca_components=200,
        add_margin=0.15, augment_flip=False,
    )
    results.append(r)

    # Test 3: Horizontal flip augmentation
    r = try_config(
        "flip aug, PCA=200, margin=0.15",
        load_hagrid_data_with_aug, pca_components=200,
        add_margin=0.15, augment_flip=True,
    )
    results.append(r)

    # Test 4: Finer HOG cells
    r = try_config(
        "finer HOG (4x4 cells), PCA=200, margin=0.15, flip",
        load_hagrid_data_with_aug, pca_components=200,
        add_margin=0.15, augment_flip=True, hog=HOG_FINER,
    )
    results.append(r)

    if not args.fast:
        # Test 5: Higher PCA with flip + margin
        r = try_config(
            "PCA=400, margin=0.15, flip aug",
            load_hagrid_data_with_aug, pca_components=400,
            add_margin=0.15, augment_flip=True,
        )
        results.append(r)

        # Test 6: Custom SVM params
        r = try_config(
            "C=100, gamma=0.01, PCA=400, margin=0.15, flip",
            load_hagrid_data_with_aug, pca_components=400,
            add_margin=0.15, augment_flip=True,
            svm_params={'C': 100, 'gamma': 0.01, 'kernel': 'rbf'},
        )
        results.append(r)

    print(f"\n{'='*70}")
    print(f"{'CONFIG':<50} {'ACCURACY':<10}")
    print(f"{'='*70}")
    results.sort(key=lambda r: r['accuracy'], reverse=True)
    for r in results:
        print(f"{r['name']:<50} {r['accuracy']:.4f}")

    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best['name']} — {best['accuracy']:.4f}")
    print(f"{'='*70}")

    # Save best model
    os.makedirs(output_dir, exist_ok=True)
    label = best['name'].split('(')[0].strip().replace(' ', '_')
    joblib.dump(best['model'], os.path.join(output_dir, "gesture_svm_best.pkl"))
    joblib.dump(best['scaler'], os.path.join(output_dir, "gesture_scaler_best.pkl"))
    joblib.dump(best['pca'], os.path.join(output_dir, "gesture_pca_best.pkl"))
    print(f"Best model saved to {output_dir}/gesture_svm_best.pkl")


if __name__ == "__main__":
    main()
