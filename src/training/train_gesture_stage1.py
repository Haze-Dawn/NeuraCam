"""
NeuraCam v6 — Stage 1: Random Forest Hand Detector
====================================================================
Binary classifier: hand vs background (NO_HAND rejection).
Trained on 40% margin HOG features to match the motion proposer's
expanded ROIs and the Stage 2 v4 SVM's input format.

Output: models/gesture_detector_rf.pkl
"""

import os, sys, json, time, argparse
import numpy as np, joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(REPO_DIR, "src", "data", "cache")

def load_40(split):
    """Load gesture + NO_HAND features at 40% margin."""
    X_g = np.load(f"{CACHE_DIR}/512px_{split}_m0_40.npy")
    y_g = np.load(f"{CACHE_DIR}/512px_{split}_m0_40_labels.npy")
    X_n = np.load(f"{CACHE_DIR}/512px_no_hand_{split}_m0_40.npy")
    y_n = np.full(len(X_n), 1, dtype=int)
    return X_g, y_g, X_n, y_n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "models"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--no-hand-ratio", type=float, default=0.5,
                        help="Fraction of NO_HAND samples relative to hand samples (default: 0.5)")
    args = parser.parse_args()

    output = args.output
    os.makedirs(output, exist_ok=True)

    print("=" * 70)
    print("  NEURACAM v6 — STAGE 1: RF BINARY HAND DETECTOR")
    print(f"  Features: HOG (1764 dims) at 40% margin")
    print(f"  Trees: {args.n_estimators}")
    print("=" * 70)

    # ── Load 40% margin data ───────────────────────────────────────────
    print("\n--- Loading data (40% margin) ---")
    X_tr_g, y_tr_g, X_tr_n, y_tr_n = load_40("train")
    X_va_g, y_va_g, X_va_n, y_va_n = load_40("val")
    X_te_g, y_te_g, X_te_n, y_te_n = load_40("test")

    # Binary labels: 0 = hand (gesture), 1 = background (NO_HAND)
    y_tr_g[:], y_va_g[:], y_te_g[:] = 0, 0, 0

    # Limit NO_HAND samples for balanced training
    n_neg = int(len(X_tr_g) * args.no_hand_ratio)
    X_tr = np.vstack([X_tr_g, X_tr_n[:n_neg]])
    y_tr = np.concatenate([y_tr_g, np.full(n_neg, 1, dtype=int)])
    shuf = np.random.RandomState(42).permutation(len(X_tr))
    X_tr, y_tr = X_tr[shuf], y_tr[shuf]

    X_va = np.vstack([X_va_g, X_va_n])
    y_va = np.concatenate([y_va_g, np.full(len(X_va_n), 1, dtype=int)])

    X_te = np.vstack([X_te_g, X_te_n])
    y_te = np.concatenate([y_te_g, np.full(len(X_te_n), 1, dtype=int)])

    print(f"  Train: {len(X_tr)}  (hand: {np.sum(y_tr==0)}, bg: {np.sum(y_tr==1)})")
    print(f"  Val:   {len(X_va)}  (hand: {np.sum(y_va==0)}, bg: {np.sum(y_va==1)})")
    print(f"  Test:  {len(X_te)}  (hand: {np.sum(y_te==0)}, bg: {np.sum(y_te==1)})")

    # ── Train RF ───────────────────────────────────────────────────────
    print(f"\n--- Training RF ({args.n_estimators} trees) ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )
    rf.fit(X_tr, y_tr)
    train_t = time.time() - t0
    print(f"  Trained in {train_t:.1f}s")

    # ── Threshold calibration ──────────────────────────────────────────
    print("\n--- Threshold calibration (F1-maximization) ---")
    proba_va = rf.predict_proba(X_va)
    conf_va = proba_va[:, 0]  # confidence of 'hand' class
    pred_va = rf.predict(X_va)
    y_va_hand = (y_va == 0).astype(int)

    best = {}
    for th in np.linspace(0.05, 0.95, 19):
        y_adj = (conf_va >= th).astype(int)
        tp = ((y_adj==1)&(y_va_hand==1)).sum()
        fp = ((y_adj==1)&(y_va_hand==0)).sum()
        fn = ((y_adj==0)&(y_va_hand==1)).sum()
        prec = tp / max(tp+fp, 1)
        rec = tp / max(tp+fn, 1)
        f1 = 2*prec*rec/max(prec+rec, 1e-10)
        flag = " <<<" if not best or f1 > best.get('f1', 0) else ""
        print(f"  th={th:.2f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}{flag}")
        if not best or f1 > best['f1']:
            best = {'th': th, 'prec': prec, 'rec': rec, 'f1': f1}

    print(f"\n  >>> Optimal: threshold={best['th']:.2f}, F1={best['f1']:.4f}")

    # ── Test evaluation ────────────────────────────────────────────────
    print("\n--- Test set evaluation ---")
    proba_te = rf.predict_proba(X_te)
    conf_te = proba_te[:, 0]
    y_adj = (conf_te >= best['th']).astype(int)
    y_te_hand = (y_te == 0).astype(int)

    tp = ((y_adj==1)&(y_te_hand==1)).sum()
    fp = ((y_adj==1)&(y_te_hand==0)).sum()
    fn = ((y_adj==0)&(y_te_hand==1)).sum()
    tn = ((y_adj==0)&(y_te_hand==0)).sum()
    prec = tp / max(tp+fp, 1)
    rec = tp / max(tp+fn, 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-10)
    acc = (tp+tn) / max(tp+fp+fn+tn, 1)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (target: ≥0.94)")
    print(f"  Recall:    {rec:.4f}  (target: ≥0.93)")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    detected = bool(prec >= 0.94 and rec >= 0.93)
    print(f"  Detection criteria: {'PASS' if detected else 'FAIL'}")

    # ── Save model ─────────────────────────────────────────────────────
    print(f"\n--- Saving model ---")
    path = os.path.join(output, "gesture_detector_rf.pkl")
    joblib.dump(rf, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.0f} MB)")

    # Save metadata
    results = {
        "model": "Random Forest binary hand detector (v6 Stage 1)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_estimators": rf.n_estimators,
            "max_depth": str(rf.max_depth),
            "min_samples_split": rf.min_samples_split,
            "min_samples_leaf": rf.min_samples_leaf,
            "class_weight": "balanced_subsample",
            "features": "HOG (1764 dims) at 40% margin",
            "no_pca": True,
            "optimal_threshold": round(float(best['th']), 4),
        },
        "data": {
            "train_samples": int(len(X_tr)),
            "val_samples": int(len(X_va)),
            "test_samples": int(len(X_te)),
            "train_hand": int(np.sum(y_tr==0)),
            "train_background": int(np.sum(y_tr==1)),
        },
        "results": {
            "test_accuracy": round(float(acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "test_f1": round(float(f1), 4),
            "confusion_matrix": [[int(tp), int(fp)], [int(fn), int(tn)]],
        },
        "detection_criteria_pass": detected,
        "timing_sec": round(train_t, 1),
    }

    jpath = os.path.join(output, "gesture_training_results.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {jpath}")

    print("\n" + "=" * 70)
    print(f"  STAGE 1 — {'PASS' if detected else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
