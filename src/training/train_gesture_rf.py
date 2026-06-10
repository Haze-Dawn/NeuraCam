"""
NeuraCam — v6 Gesture Model: Random Forest
====================================================================
Replaces HOG+SVM (v4/v5) with Random Forest for unified 6-class
hand detection + gesture classification.

Why Random Forest over SVM:
  - RF builds LOCAL decision boundaries. Adding NO_HAND does NOT shift
    gesture-vs-gesture boundaries (unlike SVM's global boundaries).
  - Native multi-class (no O(C²) pairwise explosion like SVM).
  - No PCA required — trains on full 1764 HOG dims directly.
  - Training is O(n·log n) vs SVM's O(n²) — 236K samples in ~5 min vs hours.
  - Built-in feature importance, no Platt scaling needed.
  - class_weight='balanced_subsample' handles NO_HAND's majority naturally.

Full rationale in GESTURE_MODEL_RF_REPORT.md
"""

import os, sys, json, time, argparse
import numpy as np, joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(REPO_DIR, "src", "data", "cache")

GESTURE_LABELS_6 = {
    "OPEN_PALM": 0, "FIST": 1, "THUMBS_UP": 2,
    "POINT": 3, "PEACE": 4, "NO_HAND": 5,
}
ID_TO_G = {v: k for k, v in GESTURE_LABELS_6.items()}


def load_split(split, margin=0.40):
    """Load gesture + NO_HAND features for a split."""
    X_g = np.load(os.path.join(CACHE_DIR, f"512px_{split}_m{margin:.2f}".replace(".", "_")) + ".npy")
    y_g = np.load(os.path.join(CACHE_DIR, f"512px_{split}_m{margin:.2f}".replace(".", "_")) + "_labels.npy")
    X_n = np.load(os.path.join(CACHE_DIR, f"512px_no_hand_{split}_m{margin:.2f}".replace(".", "_")) + ".npy")
    y_n = np.full(len(X_n), GESTURE_LABELS_6["NO_HAND"], dtype=np.int32)

    X = np.vstack([X_g, X_n])
    y = np.concatenate([y_g, y_n])
    shuf = np.random.RandomState(42).permutation(len(X))
    return X[shuf], y[shuf]


def train_rf(X_train, y_train, n_estimators=300, max_depth=None,
             min_samples_split=5, min_samples_leaf=2, class_weight='balanced_subsample',
             verbose=2):
    """Train a Random Forest classifier. No PCA needed.
    verbose=2 shows progress per-chunk for timing visibility."""


def calibrate_threshold(rf, X_val, y_val):
    """Find confidence threshold maximizing hand detection F1."""
    proba = rf.predict_proba(X_val)
    conf = proba.max(axis=1)
    pred = rf.predict(X_val)

    best_f1, best_th = 0.0, 0.5
    print(f"\n  Confidence calibration:")
    print(f"  {'thresh':<8} {'precision':<12} {'recall':<12} {'F1':<12}")
    for th in np.linspace(0.1, 0.95, 18):
        y_adj = np.where(conf >= th, pred, GESTURE_LABELS_6["NO_HAND"])
        ht = (y_val != GESTURE_LABELS_6["NO_HAND"]).astype(int)
        hp = (y_adj != GESTURE_LABELS_6["NO_HAND"]).astype(int)
        tp, fp, fn = (hp==1)&(ht==1), (hp==1)&(ht==0), (hp==0)&(ht==1)
        prec = tp.sum() / max(tp.sum() + fp.sum(), 1)
        rec = tp.sum() / max(tp.sum() + fn.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        print(f"  {th:<8.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
        if f1 > best_f1:
            best_f1, best_th = f1, th
    print(f"  >>> Optimal threshold: {best_th:.2f} (F1={best_f1:.4f})")
    return best_th, best_f1


def validate(rf, X_test, y_test, threshold):
    """Validate against acceptance criteria."""
    proba = rf.predict_proba(X_test)
    conf = proba.max(axis=1)
    pred = rf.predict(X_test)
    y_adj = np.where(conf >= threshold, pred, GESTURE_LABELS_6["NO_HAND"])

    # Gesture accuracy (among non-NO_HAND)
    gmask = y_test != GESTURE_LABELS_6["NO_HAND"]
    gesture_acc = float(np.mean(y_adj[gmask] == y_test[gmask])) if gmask.sum() > 0 else 0

    # Detection metrics
    ht = (y_test != GESTURE_LABELS_6["NO_HAND"]).astype(int)
    hp = (y_adj != GESTURE_LABELS_6["NO_HAND"]).astype(int)
    tp, fp, fn = (hp==1)&(ht==1), (hp==1)&(ht==0), (hp==0)&(ht==1)
    prec = tp.sum() / max(tp.sum() + fp.sum(), 1)
    rec = tp.sum() / max(tp.sum() + fn.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)

    # NO_HAND rejection
    nmask = y_test == GESTURE_LABELS_6["NO_HAND"]
    nh_rej = float(np.mean(y_adj[nmask] == y_test[nmask])) if nmask.sum() > 0 else 0

    # Per-class F1
    per_class = {}
    for cid in sorted(GESTURE_LABELS_6.values()):
        cname = ID_TO_G[cid]
        cmask = y_test == cid
        if cmask.sum() > 0:
            per_class[cname] = float(np.mean(y_adj[cmask] == y_test[cmask]))

    overall = float(np.mean(y_adj == y_test))

    print(f"\n  === ACCEPTANCE CRITERIA ===")
    results = {}
    for name, val, target in [
        ("Gesture accuracy (>=95.0%)", gesture_acc, 0.95),
        ("Detection precision (>=94.0%)", prec, 0.94),
        ("Detection recall (>=93.0%)", rec, 0.93),
        ("NO_HAND rejection (>=94.0%)", nh_rej, 0.94),
        ("Overall accuracy", overall, None),
    ]:
        status = "PASS" if target is None or val >= target else "FAIL"
        print(f"  {name:<40} {val:.4f}  {status}")
        results[name.split("(")[0].strip()] = float(val)

    print(f"\n  Per-class accuracy:")
    all_f1_pass = True
    for cname, acc in per_class.items():
        target = 0.90 if cname != "NO_HAND" else None
        st = "PASS" if target is None or acc >= target else "FAIL"
        if target is not None and acc < target:
            all_f1_pass = False
        print(f"    {cname:<15} {acc:.4f}  {st}")

    results["per_class"] = per_class
    results["overall"] = overall
    results["threshold"] = float(threshold)
    results["confusion_matrix"] = confusion_matrix(y_test, y_adj).tolist()
    results["all_pass"] = (
        gesture_acc >= 0.95 and prec >= 0.94 and rec >= 0.93 and
        nh_rej >= 0.94 and all_f1_pass
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "models"))
    parser.add_argument("--grid-search", action="store_true",
                        help="Run hyperparameter grid search (WARNING: takes hours)")
    parser.add_argument("--margin", type=float, default=0.40,
                        help="ROI margin for features (default: 0.40)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max tree depth (default: None = full)")
    parser.add_argument("--min-samples-split", type=int, default=5,
                        help="Min samples to split")
    parser.add_argument("--min-samples-leaf", type=int, default=2,
                        help="Min samples per leaf")
    args = parser.parse_args()

    output = args.output
    margin = args.margin
    os.makedirs(output, exist_ok=True)

    print("=" * 70)
    print("  NEURACAM — v6 GESTURE MODEL: RANDOM FOREST")
    print(f"  Classes: {', '.join(GESTURE_LABELS_6.keys())}")
    print(f"  ROI margin: {margin:.2f}")
    print(f"  Trees: {args.n_estimators}")
    print("  No PCA — training on full 1764 HOG dims")
    print("=" * 70)
    t_total = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("\n--- Loading data ---")
    X_tr, y_tr = load_split("train", margin)
    X_va, y_va = load_split("val", margin)
    X_te, y_te = load_split("test", margin)
    print(f"  Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
    print(f"  Train distribution: {dict(sorted(Counter(y_tr).items()))}")
    print(f"  Features: {X_tr.shape[1]} HOG dims (no PCA)")

    # ── Train RF ─────────────────────────────────────────────────────────
    # RF is robust to hyperparameters. Defaults (300 trees, full depth)
    # are known-good. verbose=2 shows per-chunk progress every ~2-3s.
    est_trees_sec = min(args.n_estimators / (177382 * 1764 / 1e9), 60)
    est_sec = est_trees_sec if est_trees_sec > 1 else 180  # rough: ~3 min for 300 trees
    print(f"\n--- Training Random Forest ({args.n_estimators} trees) ---")
    print(f"  Estimated: ~{est_sec:.0f}s on {len(X_tr)} samples × {X_tr.shape[1]} dims")
    print(f"  Progress shown in chunks (16 parallel workers)")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
        verbose=2,  # Per-chunk progress
    )
    rf.fit(X_tr, y_tr)
    train_time = time.time() - t0
    best_rf = rf
    print(f"  Trained in {train_time:.1f}s")
    print(f"  Model: {best_rf.n_estimators} trees")

    # ── Calibrate threshold ──────────────────────────────────────────────
    threshold, best_f1 = calibrate_threshold(best_rf, X_va, y_va)

    # ── Validate on test set ─────────────────────────────────────────────
    print("\n--- Test set validation ---")
    results = validate(best_rf, X_te, y_te, threshold)

    # ── Feature importance (top 20) ──────────────────────────────────────
    print("\n  Top 20 features by importance:")
    imp = best_rf.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    for i, idx in enumerate(top_idx):
        hog_cell = idx // 9  # convert flat HOG index to spatial position
        hog_bin = idx % 9
        row = hog_cell // 8  # 8 cells per row in 64×64 with 8×8 cells
        col = hog_cell % 8
        print(f"    {i+1:>2}. HOG cell=({row},{col}) bin={hog_bin}  importance={imp[idx]:.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    print("\n--- Saving model ---")
    model_path = os.path.join(output, "gesture_rf.pkl")
    joblib.dump(best_rf, model_path)
    print(f"  Saved to {model_path} ({os.path.getsize(model_path)/1024/1024:.0f} MB)")

    results_json = {
        "model": "Random Forest (6-class, no PCA)",
        "dataset": f"HAGrid 512px (6 classes, margin={margin:.2f})",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "why_rf_over_svm": (
            "RF builds local decision boundaries — adding NO_HAND does not shift "
            "gesture-vs-gesture boundaries (SVM's global boundaries lost 2.91% "
            "accuracy). RF is native multi-class (no O(C²) pairwise explosion). "
            "No PCA needed — trains on full 1764 HOG dims. Training is O(n·log n) "
            "vs SVM's O(n²). Built-in feature importance, true probabilistic output."
        ),
        "config": {
            "model_type": "RandomForestClassifier",
            "n_estimators": best_rf.n_estimators,
            "max_depth": str(best_rf.max_depth),
            "min_samples_split": best_rf.min_samples_split,
            "min_samples_leaf": best_rf.min_samples_leaf,
            "class_weight": "balanced_subsample",
            "features": "full HOG (1764 dims, no PCA)",
            "roi_margin": margin,
        },
        "data": {
            "train_samples": int(len(X_tr)),
            "val_samples": int(len(X_va)),
            "test_samples": int(len(X_te)),
            "class_distribution": {ID_TO_G[int(k)]: int(v) for k, v in sorted(Counter(y_te).items())},
        },
        "acceptance": {
            "gesture_accuracy": results.get("Gesture accuracy", 0),
            "detection_precision": results.get("Detection precision", 0),
            "detection_recall": results.get("Detection recall", 0),
            "no_hand_rejection": results.get("NO_HAND rejection", 0),
            "overall_accuracy": results["overall"],
            "threshold": results["threshold"],
            "all_pass": results["all_pass"],
        },
        "per_class_accuracy": results["per_class"],
        "confusion_matrix": results["confusion_matrix"],
        "top_features": [
            {"index": int(idx), "importance": float(imp[idx])}
            for idx in top_idx
        ],
        "timing_sec": {"total": round(time.time() - t_total, 1)},
    }

    json_path = os.path.join(output, "gesture_training_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results saved to {json_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    total_t = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  v6 RANDOM FOREST — {'PASS' if results['all_pass'] else 'FAIL'}")
    print("=" * 70)
    print(f"  Gesture accuracy:  {results.get('Gesture accuracy', 0):.4f}")
    print(f"  Detection prec:    {results.get('Detection precision', 0):.4f}")
    print(f"  Detection recall:  {results.get('Detection recall', 0):.4f}")
    print(f"  NO_HAND rejection: {results.get('NO_HAND rejection', 0):.4f}")
    print(f"  Training time:     {total_t:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
