"""
NeuraCam — v5 6-Class Gesture SVM Training
====================================================================
Unified hand detection + gesture classification using HOG+PCA+SVM.

Pipeline:
  1. Re-extract 5 gesture classes at 0% margin (full 118K, no subsample)
  2. Extract NO_HAND background negatives from same HAGrid images (~97K)
  3. Combine → PCA → StandardScaler → 6-class SVM RBF
  4. Calibrate NO_HAND confidence threshold (F1-maximization)
  5. Validate against acceptance criteria

Key differences from v4 (5-class, 40% margin):
  - 6 classes: OPEN_PALM, FIST, THUMBS_UP, POINT, PEACE, NO_HAND
  - 0% margin for ALL classes (tight crops for detection precision)
  - class_weight='balanced' (NO_HAND is ~45% of data)
  - SVM C=10 (tighter boundaries with overlapping NO_HAND class)
  - After detection, bbox is expanded 40% for refined 5-class gesture pass
"""

import os, sys, json, time, argparse, random
import numpy as np, cv2, joblib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAGRID_ROOT = "/home/hazedawn/Documents/CV Project Rev2/Data/hagrid_dataset_512"
ANN_DIR = os.path.join(HAGRID_ROOT, "annotations")
CACHE_DIR = os.path.join(REPO_DIR, "src", "data", "cache")
ARCHIVE_DIR = os.path.join(REPO_DIR, "models", "archive")
GESTURE_LABELS_6 = {
    "OPEN_PALM": 0, "FIST": 1, "THUMBS_UP": 2,
    "POINT": 3, "PEACE": 4, "NO_HAND": 5,
}
ID_TO_G = {v: k for k, v in GESTURE_LABELS_6.items()}
GESTURE_MAP = {"palm": "OPEN_PALM", "fist": "FIST", "like": "THUMBS_UP",
               "one": "POINT", "peace": "PEACE"}
MARGIN = 0.00
WIN = (64, 64)
HOG = cv2.HOGDescriptor(_winSize=WIN, _blockSize=(16, 16),
                         _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
HOG_DIM = 1764
NW = 8


def hog_feat(roi):
    return HOG.compute(cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), WIN)).ravel()


# ─── Gesture feature extraction (0% margin) ───────────────────────────────

def proc_gesture(args):
    ipath, entry, hname, label, margin = args
    if not os.path.exists(ipath):
        return None
    img = cv2.imread(ipath)
    if img is None:
        return None
    h, w = img.shape[:2]
    for bbox, blabel in zip(entry["bboxes"], entry["labels"]):
        if blabel == "no_gesture" or blabel != hname:
            continue
        xc, yc, bn, hn = bbox
        bw, bh = int(bn * w), int(hn * h)
        mx, my = int(bw * margin), int(bh * margin)
        x1 = max(0, int((xc - bn/2)*w) - mx)
        y1 = max(0, int((yc - hn/2)*h) - my)
        x2 = min(w, int((xc + bn/2)*w) + mx)
        y2 = min(h, int((yc + hn/2)*h) + my)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        try:
            feat = hog_feat(roi)
            if len(feat) == HOG_DIM:
                return (feat, label)
        except Exception:
            return None
        break
    return None


def extract_gestures(split, margin=MARGIN):
    cbase = os.path.join(CACHE_DIR, f"512px_{split}_m{margin:.2f}".replace(".", "_"))
    if os.path.exists(f"{cbase}.npy"):
        X, y = np.load(f"{cbase}.npy"), np.load(f"{cbase}_labels.npy")
        print(f"    Loaded cached {cbase}.npy ({len(X)} samples)")
        return X, y

    t0 = time.time()
    print(f"    Extracting {split} gesture features (margin={margin:.2f})...")
    feats, labs = [], []
    for hname, oname in sorted(GESTURE_MAP.items()):
        with open(os.path.join(ANN_DIR, split, f"{hname}.json")) as f:
            ann = json.load(f)
        label = GESTURE_LABELS_6[oname]
        tasks = [(os.path.join(HAGRID_ROOT, hname, f"{k}.jpg"), v, hname, label, margin)
                 for k, v in ann.items()]
        res = []
        with ThreadPoolExecutor(max_workers=NW) as ex:
            for fut in as_completed([ex.submit(proc_gesture, t) for t in tasks]):
                r = fut.result()
                if r is not None:
                    res.append(r)
        feats.extend([r[0] for r in res])
        labs.extend([r[1] for r in res])
        print(f"      {split}/{hname:>10}: {len(res)}/{len(tasks)}")
    X, y = np.array(feats), np.array(labs)
    np.save(f"{cbase}.npy", X)
    np.save(f"{cbase}_labels.npy", y)
    print(f"    Cached {len(X)} samples in {time.time()-t0:.1f}s")
    return X, y


# ─── NO_HAND background crop extraction ──────────────────────────────────

def proc_no_hand(args):
    ipath, entry, margin = args
    if not os.path.exists(ipath):
        return None
    img = cv2.imread(ipath)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Get all hand bboxes from this image
    hand_bboxes = []
    for bbox, blabel in zip(entry["bboxes"], entry["labels"]):
        if blabel == "no_gesture":
            continue
        xc, yc, bn, hn = bbox
        x1 = int((xc - bn/2) * w)
        y1 = int((yc - hn/2) * h)
        x2 = int((xc + bn/2) * w)
        y2 = int((yc + hn/2) * h)
        hand_bboxes.append((x1, y1, x2, y2))

    if not hand_bboxes:
        return None

    # Hand area distribution for size matching
    hand_areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in hand_bboxes]
    median_area = np.median(hand_areas) if hand_areas else 40000
    median_ar = np.median([(y2-y1)/max(x2-x1,1) for x1,y1,x2,y2 in hand_bboxes]) if hand_bboxes else 1.0

    # Generate negative crops that don't overlap hands
    features = []
    attempts = 0
    max_attempts = 20
    target_n = max(1, len(hand_bboxes))
    target_n = min(target_n, 3)

    while len(features) < target_n and attempts < max_attempts:
        attempts += 1
        # Random crop size matching hand distribution
        crop_area = np.random.uniform(0.5, 2.0) * median_area
        crop_ar = np.random.uniform(0.5, 2.0) * median_ar
        crop_w = int(np.sqrt(crop_area / crop_ar))
        crop_h = int(crop_area / crop_w)
        if crop_w < 20 or crop_h < 20:
            continue

        x1 = random.randint(0, max(0, w - crop_w))
        y1 = random.randint(0, max(0, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Check no overlap with any hand bbox
        overlaps = False
        hx1, hy1, hx2, hy2 = hand_bboxes[0]  # only need to avoid the main gesture hand
        for hx1, hy1, hx2, hy2 in hand_bboxes:
            ox = max(0, min(x2, hx2) - max(x1, hx1))
            oy = max(0, min(y2, hy2) - max(y1, hy1))
            if ox * oy > 0.1 * max(crop_w*crop_h, 1):
                overlaps = True
                break

        if overlaps:
            continue

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Apply margin (same 0% for consistency)
        if margin > 0:
            mx, my = int(crop_w * margin), int(crop_h * margin)
            x1e = max(0, x1 - mx)
            y1e = max(0, y1 - my)
            x2e = min(w, x2 + mx)
            y2e = min(h, y2 + my)
            roi = img[y1e:y2e, x1e:x2e]

        try:
            feat = hog_feat(roi)
            if len(feat) == HOG_DIM:
                features.append(feat)
        except Exception:
            continue

    return features if features else None


def extract_no_hand(split, margin=MARGIN, max_samples=None):
    cbase = os.path.join(CACHE_DIR, f"512px_no_hand_{split}_m{margin:.2f}".replace(".", "_"))
    if os.path.exists(f"{cbase}.npy"):
        X = np.load(f"{cbase}.npy")
        y = np.load(f"{cbase}_labels.npy")
        print(f"    Loaded cached {cbase}.npy ({len(X)} samples)")
        return X, y

    t0 = time.time()
    print(f"    Extracting {split} NO_HAND negatives (margin={margin:.2f})...")

    all_features = []
    total_images = 0

    for hname in sorted(GESTURE_MAP.keys()):
        ann_path = os.path.join(ANN_DIR, split, f"{hname}.json")
        img_dir = os.path.join(HAGRID_ROOT, hname)
        if not os.path.exists(ann_path) or not os.path.isdir(img_dir):
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        label = GESTURE_LABELS_6["NO_HAND"]
        tasks = [(os.path.join(img_dir, f"{k}.jpg"), v, margin)
                 for k, v in ann.items()]
        total_images += len(tasks)

        res = []
        with ThreadPoolExecutor(max_workers=NW) as ex:
            for fut in as_completed([ex.submit(proc_no_hand, t) for t in tasks]):
                r = fut.result()
                if r is not None:
                    res.extend(r)

        all_features.extend(res)
        print(f"      {split}/{hname:>10}: {len(res)} negatives from {len(tasks)} images")

        if max_samples and len(all_features) >= max_samples:
            all_features = all_features[:max_samples]
            break

    X = np.array(all_features)
    y = np.full(len(X), GESTURE_LABELS_6["NO_HAND"], dtype=np.int32)
    np.save(f"{cbase}.npy", X)
    np.save(f"{cbase}_labels.npy", y)
    print(f"    Cached {len(X)} NO_HAND samples in {time.time()-t0:.1f}s")
    return X, y


# ─── Training ────────────────────────────────────────────────────────────

def train_6class(X_train, y_train, X_val, y_val, C=10, gamma='scale',
                 pca_n=200, class_weight='balanced'):
    t0 = time.time()

    n = min(pca_n, X_train.shape[1], len(X_train)-1)
    pca = PCA(n_components=n, svd_solver='randomized')
    X_tr_p = pca.fit_transform(X_train)
    X_va_p = pca.transform(X_val)
    var = float(pca.explained_variance_ratio_.sum())

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_p)
    X_va_s = scaler.transform(X_va_p)

    print(f"  PCA: {X_train.shape[1]} -> {n} (var={var:.3f})")
    print(f"  Training SVM: C={C}, gamma={gamma}, class_weight={class_weight}...")

    svm = SVC(C=C, gamma=gamma, kernel='rbf', class_weight=class_weight,
              probability=True, random_state=42)
    svm.fit(X_tr_s, y_train)

    train_time = time.time() - t0
    val_pred = svm.predict(X_va_s)
    val_acc = float(np.mean(val_pred == y_val))

    print(f"  Trained in {train_time:.1f}s, {len(svm.support_)} SVs")
    print(f"  Validation accuracy: {val_acc:.4f}")

    return svm, scaler, pca, val_pred, val_acc


def calibrate_threshold(svm, scaler, pca, X_val, y_val):
    """Find confidence threshold that maximizes hand detection F1."""
    X_p = pca.transform(X_val)
    X_s = scaler.transform(X_p)
    y_pred = svm.predict(X_s)
    conf = svm.predict_proba(X_s).max(axis=1)

    best_f1 = 0.0
    best_th = 0.6
    results = []

    for th in np.linspace(0.1, 0.95, 18):
        y_adj = np.where(conf >= th, y_pred, GESTURE_LABELS_6["NO_HAND"])
        hand_true = (y_val != GESTURE_LABELS_6["NO_HAND"]).astype(int)
        hand_pred = (y_adj != GESTURE_LABELS_6["NO_HAND"]).astype(int)

        tp = np.sum((hand_pred == 1) & (hand_true == 1))
        fp = np.sum((hand_pred == 1) & (hand_true == 0))
        fn = np.sum((hand_pred == 0) & (hand_true == 1))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        results.append((th, prec, rec, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\n  Confidence threshold calibration:")
    print(f"  {'thresh':<8} {'precision':<12} {'recall':<12} {'F1':<12}")
    for th, prec, rec, f1 in results:
        flag = " <<<" if abs(th - best_th) < 0.01 else ""
        print(f"  {th:<8.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}{flag}")
    print(f"  >>> Optimal threshold: {best_th:.2f} (F1={best_f1:.4f})")
    return best_th, best_f1


def validate_acceptance(svm, scaler, pca, X_test, y_test, threshold):
    """Validate against v5 acceptance criteria."""
    X_p = pca.transform(X_test)
    X_s = scaler.transform(X_p)
    y_pred = svm.predict(X_s)
    conf = svm.predict_proba(X_s).max(axis=1)
    y_adj = np.where(conf >= threshold, y_pred, GESTURE_LABELS_6["NO_HAND"])

    results = {}

    # Gesture accuracy (among gesture-class samples only)
    gmask = y_test != GESTURE_LABELS_6["NO_HAND"]
    if gmask.sum() > 0:
        gesture_acc = float(np.mean(y_adj[gmask] == y_test[gmask]))
    else:
        gesture_acc = 0.0
    results["gesture_accuracy"] = gesture_acc

    # Hand detection precision/recall
    hand_true = (y_test != GESTURE_LABELS_6["NO_HAND"]).astype(int)
    hand_pred = (y_adj != GESTURE_LABELS_6["NO_HAND"]).astype(int)

    tp = np.sum((hand_pred == 1) & (hand_true == 1))
    fp = np.sum((hand_pred == 1) & (hand_true == 0))
    fn = np.sum((hand_pred == 0) & (hand_true == 1))
    tn = np.sum((hand_pred == 0) & (hand_true == 0))

    results["detection_precision"] = tp / max(tp + fp, 1)
    results["detection_recall"] = tp / max(tp + fn, 1)
    results["detection_f1"] = 2 * results["detection_precision"] * results["detection_recall"] / \
        max(results["detection_precision"] + results["detection_recall"], 1e-10)

    # NO_HAND rejection rate
    nmask = y_test == GESTURE_LABELS_6["NO_HAND"]
    if nmask.sum() > 0:
        results["no_hand_rejection"] = float(np.mean(y_adj[nmask] == y_test[nmask]))
    else:
        results["no_hand_rejection"] = 0.0

    # Per-class F1
    results["per_class_f1"] = {}
    for cls_id in sorted(GESTURE_LABELS_6.values()):
        cls_name = ID_TO_G[cls_id]
        c_mask = y_test == cls_id
        if c_mask.sum() == 0:
            continue
        c_acc = float(np.mean(y_adj[c_mask] == y_test[c_mask]))
        results["per_class_f1"][cls_name] = c_acc

    # Overall accuracy
    results["overall_accuracy"] = float(np.mean(y_adj == y_test))

    # Print
    criteria = [
        ("Gesture accuracy (≥95.0%)", results["gesture_accuracy"], 0.95),
        ("Detection precision (≥94.0%)", results["detection_precision"], 0.94),
        ("Detection recall (≥93.0%)", results["detection_recall"], 0.93),
        ("NO_HAND rejection (≥94.0%)", results["no_hand_rejection"], 0.94),
        ("Overall accuracy", results["overall_accuracy"], None),
    ]
    print(f"\n  {'Criterion':<40} {'Value':<10} {'Pass?':<8}")
    print(f"  " + "-" * 62)
    all_pass = True
    for name, val, target in criteria:
        if target is not None:
            passed = val >= target
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
        else:
            status = "N/A"
        print(f"  {name:<40} {val:<10.4f} {status:<8}")

    print(f"\n  Per-class accuracy:")
    for cls_name, acc in results["per_class_f1"].items():
        target = 0.90 if cls_name != "NO_HAND" else None
        if target is not None:
            passed = acc >= target
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
        else:
            status = "N/A"
        print(f"    {cls_name:<15} {acc:.4f}  {status}")

    results["all_criteria_pass"] = all_pass
    results["confusion_matrix"] = confusion_matrix(y_test, y_adj).tolist()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "models"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--no-hand-only", action="store_true",
                        help="Only extract NO_HAND negatives")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip extraction, only train from cached")
    args = parser.parse_args()

    output = args.output
    use_cache = not args.no_cache
    os.makedirs(output, exist_ok=True)

    print("=" * 70)
    print("  NEURACAM — v5 6-CLASS GESTURE SVM TRAINING")
    print(f"  Classes: {', '.join(GESTURE_LABELS_6.keys())}")
    print(f"  ROI margin: {MARGIN:.2f}")
    print("=" * 70)
    t_total = time.time()

    # ── Step 1: Extract gesture features at 0% margin ──────────────────────
    if not args.no_hand_only:
        print("\n--- Step 1: Gesture features (0% margin) ---")
        X_tr_g, y_tr_g = extract_gestures("train", MARGIN)
        X_va_g, y_va_g = extract_gestures("val", MARGIN)
        X_te_g, y_te_g = extract_gestures("test", MARGIN)
        print(f"  Gesture samples — train: {len(X_tr_g)}, val: {len(X_va_g)}, test: {len(X_te_g)}")

    # ── Step 2: Extract NO_HAND negatives ──────────────────────────────────
    print("\n--- Step 2: NO_HAND background negatives ---")
    X_tr_n, y_tr_n = extract_no_hand("train", MARGIN)
    X_va_n, y_va_n = extract_no_hand("val", MARGIN)
    X_te_n, y_te_n = extract_no_hand("test", MARGIN)

    if args.no_hand_only:
        print("\nNO_HAND extraction complete. Exiting (--no-hand-only).")
        return

    # ── Step 3: Combine into 6-class dataset ──────────────────────────────
    print("\n--- Step 3: Combine into 6-class dataset ---")
    X_train = np.vstack([X_tr_g, X_tr_n])
    y_train = np.concatenate([y_tr_g, y_tr_n])
    X_val = np.vstack([X_va_g, X_va_n])
    y_val = np.concatenate([y_va_g, y_va_n])
    X_test = np.vstack([X_te_g, X_te_n])
    y_test = np.concatenate([y_te_g, y_te_n])

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Class distribution (train): "
          f"{dict(sorted(Counter(y_train).items()))}")

    # Subsample for practical SVM training (RBF kernel is O(n²))
    # Target: ~60K total (8K/gesture, 24K NO_HAND), estimated ~35 min train time
    NO_HAND_ID = GESTURE_LABELS_6["NO_HAND"]
    n_per_gesture = min(8000, min(Counter(y_train)[c] for c in range(5) if Counter(y_train)[c] > 0))
    n_no_hand = n_per_gesture * 3  # 3:1 NO_HAND:gesture for balanced training

    X_tr_sub, y_tr_sub = [], []
    for c in range(5):
        mask = y_train == c
        idx = np.where(mask)[0]
        np.random.RandomState(42).shuffle(idx)
        selected = idx[:n_per_gesture]
        X_tr_sub.append(X_train[selected])
        y_tr_sub.append(y_train[selected])
    # NO_HAND
    mask_n = y_train == NO_HAND_ID
    idx_n = np.where(mask_n)[0]
    np.random.RandomState(42).shuffle(idx_n)
    selected_n = idx_n[:n_no_hand]
    X_tr_sub.append(X_train[selected_n])
    y_tr_sub.append(y_train[selected_n])

    X_train = np.vstack(X_tr_sub)
    y_train = np.concatenate(y_tr_sub)
    # Shuffle
    shuf = np.random.RandomState(42).permutation(len(X_train))
    X_train, y_train = X_train[shuf], y_train[shuf]
    print(f"  Subsampled train: {len(X_train)} samples "
          f"({dict(sorted(Counter(y_train).items()))})")

    # ── Step 4: Train 6-class SVM ─────────────────────────────────────────
    print("\n--- Step 4: Train 6-class SVM ---")
    svm, scaler, pca, val_pred, val_acc = train_6class(
        X_train, y_train, X_val, y_val,
        C=10, gamma='scale', pca_n=200, class_weight='balanced'
    )

    # ── Step 5: Calibrate confidence threshold ───────────────────────────
    print("\n--- Step 5: Confidence calibration ---")
    best_th, best_f1 = calibrate_threshold(svm, scaler, pca, X_val, y_val)

    # ── Step 6: Validate against acceptance criteria on test set ──────────
    print("\n--- Step 6: Acceptance criteria validation ---")
    val_results = validate_acceptance(svm, scaler, pca, X_test, y_test, best_th)

    # ── Step 7: Save model ───────────────────────────────────────────────
    print("\n--- Step 7: Save model ---")
    joblib.dump(svm, os.path.join(output, "gesture_svm.pkl"))
    joblib.dump(scaler, os.path.join(output, "gesture_scaler.pkl"))
    joblib.dump(pca, os.path.join(output, "gesture_pca.pkl"))
    print(f"  Saved to {output}/gesture_svm.pkl")

    # Save results
    results = {
        "model": "6-class HOG + PCA + SVM RBF",
        "dataset": "HAGrid 512px (5 gesture classes + NO_HAND)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "classes": list(GESTURE_LABELS_6.keys()),
            "roi_margin": MARGIN,
            "pca_components": pca.n_components_,
            "pca_variance": float(pca.explained_variance_ratio_.sum()),
            "svm_C": 10, "svm_gamma": "scale", "svm_kernel": "rbf",
            "class_weight": "balanced",
            "hog_window": "64x64", "hog_bins": 9, "hog_dim": HOG_DIM,
            "optimal_confidence_threshold": round(float(best_th), 4),
        },
        "data": {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "class_distribution_train": {
                ID_TO_G[int(k)]: int(v) for k, v in sorted(Counter(y_train).items())
            },
        },
        "acceptance_validation": val_results,
        "timing_sec": {"total": round(time.time() - t_total, 1)},
    }

    with open(os.path.join(output, "gesture_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved")

    # ── Summary ──────────────────────────────────────────────────────────
    total_t = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  v5 TRAINING {'PASSED' if val_results.get('all_criteria_pass') else 'FAILED'}")
    print("=" * 70)
    print(f"  Gesture accuracy:  {val_results.get('gesture_accuracy', 0):.4f}")
    print(f"  Detection prec:    {val_results.get('detection_precision', 0):.4f}")
    print(f"  Detection recall:  {val_results.get('detection_recall', 0):.4f}")
    print(f"  NO_HAND rejection: {val_results.get('no_hand_rejection', 0):.4f}")
    print(f"  Optimal threshold: {best_th:.2f}")
    print(f"  Total time:        {total_t:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
