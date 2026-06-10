import os, sys, json, csv, cv2
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.train_gesture_hagrid import (
    GESTURE_LABELS, ID_TO_GESTURE, HOG_FEATURE_DIM, WINDOW_SIZE,
    load_existing_csv,
)

HAGRID_ROOT = "/home/hazedawn/Documents/CV Project Rev2/Data/hagrid-sample-30k-384p"
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(REPO_DIR, "models")

HOG = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE, _blockSize=(16, 16),
    _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
)


def evaluate(svm, scaler, pca, X_test, y_test):
    X_pca = pca.transform(X_test)
    X_s = scaler.transform(X_pca)
    pred = svm.predict(X_s)
    return float(np.mean(pred == y_test)), pred


def load_hagrid_test():
    import json
    hagrid_map = {"palm": "OPEN_PALM", "fist": "FIST", "like": "THUMBS_UP",
                   "one": "POINT", "peace": "PEACE"}
    ann_dir = os.path.join(HAGRID_ROOT, "ann_train_val")
    img_base = os.path.join(HAGRID_ROOT, "hagrid_30k")
    X, y = [], []
    for hname, oname in sorted(hagrid_map.items()):
        with open(os.path.join(ann_dir, f"{hname}.json")) as f:
            ann = json.load(f)
        img_dir = os.path.join(img_base, f"train_val_{hname}")
        label = GESTURE_LABELS[oname]
        count = 0
        for key, entry in ann.items():
            img_path = os.path.join(img_dir, f"{key}.jpg")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            for bbox, blabel in zip(entry["bboxes"], entry["labels"]):
                if blabel == "no_gesture" or blabel != hname:
                    continue
                xc, yc, bw_n, bh_n = bbox
                x1 = max(0, int((xc - bw_n/2) * w))
                y1 = max(0, int((yc - bh_n/2) * h))
                bw = min(int(bw_n * w), w - x1)
                bh = min(int(bh_n * h), h - y1)
                if bw < 20 or bh < 20:
                    continue
                roi = img[y1:y1+bh, x1:x1+bw]
                if roi.size == 0:
                    continue
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, WINDOW_SIZE)
                feat = HOG.compute(resized).ravel()
                if len(feat) == HOG_FEATURE_DIM:
                    X.append(feat)
                    y.append(label)
                    count += 1
                break
    return np.array(X), np.array(y)


def main():
    args_out = "/home/hazedawn/Documents/CV Project Rev2/Source of truth/GESTURE_MODEL_REPORT.md"
    
    report = []
    def log(s=""):
        print(s)
        report.append(s)

    log("# Gesture Model Benchmark Report")
    log(f"Date: 2026-05-20")
    log(f"")
    
    csv_path = os.path.join(REPO_DIR, "data/gesture/raw/features_2500.csv")
    X_user, y_user = load_existing_csv(csv_path)
    log(f"## Data Overview")
    log(f"")
    log(f"| Dataset | Samples | Source |")
    log(f"|---------|---------|--------|")
    log(f"| User-collected | {len(X_user)} | 500 per class, webcam captures |")
    X_h, y_h = load_hagrid_test()
    log(f"| HAGrid 30k | {len(X_h)} | ~1,700 per class, diverse backgrounds |")
    log(f"| Combined | {len(X_h) + len(X_user)} | Full training pool |")
    log(f"")

    # Split user data into train/test (80/20)
    Xu_train, Xu_test, yu_train, yu_test = train_test_split(
        X_user, y_user, test_size=0.2, stratify=y_user, random_state=42
    )
    
    # Split HAGrid data into train/test (80/20)
    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
        X_h, y_h, test_size=0.2, stratify=y_h, random_state=42
    )

    X_combined_train = np.vstack([Xu_train, Xh_train])
    y_combined_train = np.concatenate([yu_train, yh_train])

    X_combined_test = np.vstack([Xu_test, Xh_test])
    y_combined_test = np.concatenate([yu_test, yh_test])

    log(f"## Test Set Definitions")
    log(f"")
    log(f"| Test Set | Samples | Composition |")
    log(f"|----------|---------|-------------|")
    log(f"| User-only | {len(Xu_test)} | 100 per class from user data |")
    log(f"| HAGrid-only | {len(Xh_test)} | ~340 per class from HAGrid |")
    log(f"| Combined | {len(X_combined_test)} | Full mix |")
    log(f"")

    models = [
        ("User-Only 1500 (old)",
         os.path.join(MODELS_DIR, "gesture_svm_1500.pkl"),
         os.path.join(MODELS_DIR, "gesture_scaler_1500.pkl"),
         os.path.join(MODELS_DIR, "gesture_pca_1500.pkl")),
        ("HAGrid+Existing (new)",
         os.path.join(MODELS_DIR, "gesture_svm.pkl"),
         os.path.join(MODELS_DIR, "gesture_scaler.pkl"),
         os.path.join(MODELS_DIR, "gesture_pca.pkl")),
    ]
    # Verify model files exist
    for name, svm_p, sc_p, pca_p in models:
        for p in [svm_p, sc_p, pca_p]:
            if not os.path.exists(p):
                print(f"ERROR: Missing model file: {p}")
                sys.exit(1)

    test_sets = [
        ("User-only test set", Xu_test, yu_test),
        ("HAGrid-only test set", Xh_test, yh_test),
        ("Combined test set", X_combined_test, y_combined_test),
    ]

    log(f"## Model Comparison")
    log(f"")
    log(f"| Model | Params | Support Vectors | Size |")
    log(f"|-------|--------|----------------|------|")

    model_info = []
    for name, svm_p, sc_p, pca_p in models:
        svm = joblib.load(svm_p)
        n_sv = len(svm.support_)
        size_kb = os.path.getsize(svm_p) / 1024
        
        # Load results JSON for metadata
        results_path = svm_p.replace("gesture_svm", "gesture_training_results").replace(".pkl", ".json")
        params_str = "?"
        samples_str = "?"
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    rd = json.load(f)
                bp = rd.get("best_params", rd.get("params", {}))
                params_str = str(bp)
                samples_str = str(rd.get("samples", "?"))
            except:
                pass
        
        log(f"| {name} | {params_str} | {n_sv} | {size_kb:.0f} KB |")
        model_info.append((name, svm, sc_p, pca_p))

    log(f"")

    # Full benchmark matrix
    log(f"## Benchmark Matrix (All Models × All Test Sets)")
    log(f"")
    header = "| Model | " + " | ".join([ts[0] for ts in test_sets]) + " |"
    sep = "|-------|" + "|".join(["--------" for _ in test_sets]) + "|"
    log(header)
    log(sep)

    all_results = []
    for name, svm, sc_p, pca_p in model_info:
        scaler = joblib.load(sc_p)
        pca = joblib.load(pca_p)
        accs = []
        row = f"| {name} "
        for ts_name, Xt, yt in test_sets:
            acc, pred = evaluate(svm, scaler, pca, Xt, yt)
            accs.append((ts_name, acc, pred, Xt, yt))
            row += f"| {acc*100:.1f}% "
        row += "|"
        log(row)
        all_results.append((name, accs))

    log(f"")

    # Per-class breakdown
    log(f"## Per-Class Accuracy Breakdown")
    log(f"")
    for name, accs in all_results:
        log(f"### {name}")
        log(f"")
        cls_names = list(GESTURE_LABELS.keys())
        header = "| Class | " + " | ".join([a[0] for a in accs]) + " |"
        sep = "|-------|" + "|".join(["--------" for _ in accs]) + "|"
        log(header)
        log(sep)
        for cls_id in sorted(GESTURE_LABELS.values()):
            cname = ID_TO_GESTURE[cls_id]
            row = f"| {cname} "
            for ts_name, acc, pred, Xt, yt in accs:
                mask = yt == cls_id
                if mask.sum() > 0:
                    ca = float(np.mean(pred[mask] == yt[mask]))
                    row += f"| {ca*100:.1f}% "
                else:
                    row += "| N/A "
            row += "|"
            log(row)
        log(f"")

    # Confusion matrices
    log(f"## Confusion Matrices")
    log(f"")
    for name, accs in all_results:
        for ts_name, acc, pred, Xt, yt in accs:
            log(f"### {name} on {ts_name}")
            log(f"")
            cm = confusion_matrix(yt, pred)
            cls_names = list(GESTURE_LABELS.keys())
            header = "| True \\\\ Pred | " + " | ".join(f"{c}" for c in cls_names) + " |"
            log(header)
            log("|" + "|".join(["---" for _ in range(len(cls_names)+1)]) + "|")
            for i, row in enumerate(cm):
                log(f"| {cls_names[i]} | " + " | ".join(str(v) for v in row) + " |")
            log(f"")

    # Write report
    with open(args_out, "w") as f:
        f.write("\n".join(report))
    log(f"\nReport saved to {args_out}")


if __name__ == "__main__":
    main()
