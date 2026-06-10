import os
import cv2
import csv
import json
import numpy as np
import joblib
import argparse
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

GESTURE_LABELS = {
    "OPEN_PALM": 0, "FIST": 1, "THUMBS_UP": 2,
    "POINT": 3, "PEACE": 4,
}
ID_TO_GESTURE = {v: k for k, v in GESTURE_LABELS.items()}
WINDOW_SIZE = (64, 64)
HOG = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)
HOG_FEATURE_DIM = 1764

HAGRID_GESTURE_MAP = {
    "palm": "OPEN_PALM",
    "fist": "FIST",
    "like": "THUMBS_UP",
    "one": "POINT",
    "peace": "PEACE",
}


def extract_hog_from_roi(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, WINDOW_SIZE)
    return HOG.compute(resized).ravel()


def load_hagrid_data(hagrid_root: str, max_per_class: int = None):
    ann_dir = os.path.join(hagrid_root, "ann_train_val")
    img_base = os.path.join(hagrid_root, "hagrid_30k")

    all_features = []
    all_labels = []

    for hagrid_name, our_name in sorted(HAGRID_GESTURE_MAP.items()):
        ann_path = os.path.join(ann_dir, f"{hagrid_name}.json")
        img_dir = os.path.join(img_base, f"train_val_{hagrid_name}")

        if not os.path.exists(ann_path):
            print(f"  Skipping {hagrid_name}: annotation file not found")
            continue
        if not os.path.isdir(img_dir):
            print(f"  Skipping {hagrid_name}: image directory not found")
            continue

        with open(ann_path) as f:
            annotations = json.load(f)

        label_target = GESTURE_LABELS[our_name]
        features = []
        label = GESTURE_LABELS[our_name]

        count = 0
        errors = 0
        for key, entry in annotations.items():
            if max_per_class and count >= max_per_class:
                break

            img_path = os.path.join(img_dir, f"{key}.jpg")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                errors += 1
                continue
            h, w = img.shape[:2]

            found = False
            for bbox, bbox_label in zip(entry["bboxes"], entry["labels"]):
                if bbox_label == "no_gesture" or bbox_label != hagrid_name:
                    continue
                xc, yc, bw, bh = bbox
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                bw = int(bw * w)
                bh = int(bh * h)
                x1 = max(0, x1)
                y1 = max(0, y1)
                bw = min(bw, w - x1)
                bh = min(bh, h - y1)

                if bw < 20 or bh < 20:
                    continue

                roi = img[y1:y1 + bh, x1:x1 + bw]
                if roi.size == 0:
                    continue

                try:
                    feat = extract_hog_from_roi(roi)
                    if len(feat) == HOG_FEATURE_DIM:
                        all_features.append(feat)
                        all_labels.append(label_target)
                        count += 1
                        found = True
                except Exception:
                    errors += 1

                if found:
                    break

        print(f"  {hagrid_name} -> {our_name}: {count} samples ({errors} errors)")

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


def train(hagrid_root: str, existing_csv: str, output_dir: str = "models",
          pca_components: int = 80, max_per_class: int = None):
    print("=" * 60)
    print("Loading HAGrid data...")
    print("=" * 60)
    X_hagrid, y_hagrid = load_hagrid_data(hagrid_root, max_per_class)
    print(f"\nHAGrid total: {len(X_hagrid)} samples")

    if existing_csv and os.path.exists(existing_csv):
        print("\n" + "=" * 60)
        print(f"Loading existing data from {existing_csv}...")
        print("=" * 60)
        X_existing, y_existing = load_existing_csv(existing_csv)
        print(f"Existing total: {len(X_existing)} samples")
        print(f"Per-class: {dict(sorted(Counter(y_existing).items()))}")

        X = np.vstack([X_hagrid, X_existing])
        y = np.concatenate([y_hagrid, y_existing])
    else:
        print("No existing CSV found, using HAGrid only")
        X, y = X_hagrid, y_hagrid

    print("\n" + "=" * 60)
    print(f"Combined dataset: {len(X)} samples")
    print(f"Class distribution: {dict(sorted(Counter(y).items()))}")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n_components = min(pca_components, X_train.shape[1], len(X_train) - 1)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    print(f"\nPCA: {X.shape[1]} -> {n_components} components "
          f"(variance retained: {pca.explained_variance_ratio_.sum():.3f})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca)

    print("\nTraining SVM with GridSearchCV (reduced grid)...")
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf'],
    }
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_train_scaled, y_train)

    best_svm = grid.best_estimator_
    print(f"\nBest params: {grid.best_params_}")
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
    joblib.dump(pca, os.path.join(output_dir, "gesture_pca.pkl"))
    print(f"\nModels saved to {output_dir}/")

    results = {
        "search_method": "RandomizedSearchCV",
        "best_params": grid.best_params_,
        "cv_accuracy": float(grid.best_score_),
        "validation_accuracy": float(val_acc),
        "pca_components": n_components,
        "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "samples": len(X),
        "hagrid_samples": len(X_hagrid),
        "existing_samples": len(X_existing) if existing_csv and os.path.exists(existing_csv) else 0,
        "class_distribution": {ID_TO_GESTURE[int(k)]: int(v) for k, v in sorted(Counter(y).items())},
    }
    with open(os.path.join(output_dir, "gesture_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/gesture_training_results.json")

    return best_svm, scaler, pca, results


def benchmark_models(models_dir: str, h_model, h_scaler, h_pca,
                     X_test, y_test):
    """Benchmark HAGrid-trained model vs existing user-data-only models."""
    print("\n" + "=" * 70)
    print("BENCHMARK: HAGrid+Existing vs Existing-Only Models")
    print("=" * 70)

    name_label = {v: k for k, v in GESTURE_LABELS.items()}

    comparisons = []
    existing_variants = [
        (models_dir, "existing_1500"),
    ]
    # Check for 2500 model too
    alt_dir = os.path.join(models_dir, "2500")
    if os.path.exists(os.path.join(alt_dir, "gesture_svm.pkl")):
        existing_variants.insert(0, (alt_dir, "existing_2500"))

    for model_dir, label in existing_variants:
        svm_path = os.path.join(model_dir, "gesture_svm.pkl")
        scaler_path = os.path.join(model_dir, "gesture_scaler.pkl")
        pca_path = os.path.join(model_dir, "gesture_pca.pkl")
        if not all(os.path.exists(p) for p in [svm_path, scaler_path, pca_path]):
            print(f"  Skipping {label}: model files not found in {model_dir}")
            continue
        svm = joblib.load(svm_path)
        scaler = joblib.load(scaler_path)
        pca_m = joblib.load(pca_path)
        X_test_pca = pca_m.transform(X_test)
        X_test_s = scaler.transform(X_test_pca)
        pred = svm.predict(X_test_s)
        acc = np.mean(pred == y_test)
        comparisons.append((label, acc, pred))

    X_test_pca_h = h_pca.transform(X_test)
    X_test_s_h = h_scaler.transform(X_test_pca_h)
    pred_h = h_model.predict(X_test_s_h)
    acc_h = np.mean(pred_h == y_test)
    comparisons.append(("HAGrid+Existing", acc_h, pred_h))

    print(f"\n{'Model':<25} {'Accuracy':<12} {'Samples':<10}")
    print("-" * 50)
    for label, acc, _ in comparisons:
        print(f"{label:<25} {acc:.4f}         {len(y_test)}")

    print(f"\nPer-class accuracy breakdown:")
    for label, acc, pred in comparisons:
        print(f"\n  {label} (overall: {acc:.4f}):")
        for cls_id in sorted(GESTURE_LABELS.values()):
            mask = y_test == cls_id
            if mask.sum() > 0:
                cls_acc = np.mean(pred[mask] == y_test[mask])
                print(f"    {name_label[cls_id]:<15} {cls_acc:.4f}  ({mask.sum():>4} samples)")

    print("\nConfusion matrices:")
    from sklearn.metrics import confusion_matrix
    for label, acc, pred in comparisons:
        cm = confusion_matrix(y_test, pred)
        print(f"\n  {label}:")
        print(f"           " + " ".join(f"{name_label[i]:>8}" for i in sorted(GESTURE_LABELS.values())))
        for i, row in enumerate(cm):
            print(f"  {name_label[i]:<12} " + " ".join(f"{v:>8}" for v in row))

    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hagrid-root",
                        default="/home/hazedawn/Documents/CV Project Rev2/Data/hagrid-sample-30k-384p")
    parser.add_argument("--existing-csv",
                        default="data/gesture/raw/features_2500.csv")
    parser.add_argument("--output", default="models")
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limit samples per class from HAGrid (useful for testing)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip training, only benchmark existing models against a test set")
    args = parser.parse_args()

    if args.benchmark_only:
        print("Benchmark-only mode: comparing models on test set")
        _, _, X_test_db = train_test_split(
            np.zeros((10, 1764)), np.zeros(10), test_size=0.2, random_state=42
        )
        print("Run without --benchmark-only to train + benchmark")
    else:
        best_svm, scaler, pca, results = train(
            args.hagrid_root, args.existing_csv, args.output,
            args.pca_components, args.max_per_class
        )

        X_all, y_all = [], []
        if args.existing_csv and os.path.exists(args.existing_csv):
            X_ex, y_ex = load_existing_csv(args.existing_csv)
            X_all.append(X_ex)
            y_all.append(y_ex)
        X_h, y_h = load_hagrid_data(args.hagrid_root, args.max_per_class)
        X_all.append(X_h)
        y_all.append(y_h)
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        benchmark_models(args.output, best_svm, scaler, pca, X_test, y_test)
