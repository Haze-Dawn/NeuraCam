"""Fast SVM optimization: load cached features, test configs quickly."""
import os, sys, json, csv, cv2
import numpy as np
import joblib
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.train_gesture_hagrid import (
    GESTURE_LABELS, ID_TO_GESTURE, HOG_FEATURE_DIM, WINDOW_SIZE,
    load_existing_csv,
)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAGRID_ROOT = "/home/hazedawn/Documents/CV Project Rev2/Data/hagrid-sample-30k-384p"
CACHE_DIR = os.path.join(REPO_DIR, "src", "data", "cache")
csv_path = os.path.join(REPO_DIR, "data/gesture/raw/features_2500.csv")


def load_cached(name):
    return (
        np.load(os.path.join(CACHE_DIR, f"{name}.npy")),
        np.load(os.path.join(CACHE_DIR, f"{name}_labels.npy")),
    )


def evaluate(name, X_h, y_h, X_ex, y_ex, pca_n=80, C=10, gamma='scale'):
    X = np.vstack([X_h, X_ex])
    y = np.concatenate([y_h, y_ex])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n = min(pca_n, X_tr.shape[1], len(X_tr) - 1)
    pca = PCA(n_components=n)
    X_tr_p = pca.fit_transform(X_tr)
    X_te_p = pca.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_p)
    X_te_s = sc.transform(X_te_p)

    svm = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42)
    svm.fit(X_tr_s, y_tr)

    pred = svm.predict(X_te_s)
    acc = float(np.mean(pred == y_te))
    cm = confusion_matrix(y_te, pred)

    return {
        "name": name, "acc": acc, "sv": len(svm.support_),
        "pca_n": n, "samples": len(X),
        "cm": cm, "y_te": y_te, "pred": pred,
        "model": svm, "scaler": sc, "pca": pca,
    }


def main():
    print("=" * 70)
    print("SVM OPTIMIZATION SWEEP (using cached features)")
    print("=" * 70)

    X_ex, y_ex = load_existing_csv(csv_path)
    print(f"\nUser data: {len(X_ex)} samples")

    # Find cached HAGrid feature variants
    import glob
    base_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("hagrid_m0.00_hog") and f.endswith(".npy") and "_labels" not in f]
    margin_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("hagrid_m0.20_hog") and f.endswith(".npy") and "_labels" not in f]
    aug_file = [f for f in os.listdir(CACHE_DIR) if f.startswith("hagrid_aug_m0.20") and f.endswith(".npy") and "_labels" not in f]

    if not base_files or not margin_files or not aug_file:
        print("ERROR: Cache files not found. Run pre-extraction first.")
        sys.exit(1)

    base_name = base_files[0].replace(".npy", "")
    margin_name = margin_files[0].replace(".npy", "")
    aug_name = aug_file[0].replace(".npy", "")

    X_h_base, y_h_base = load_cached(base_name)
    X_h_margin, y_h_margin = load_cached(margin_name)
    X_h_aug, y_h_aug = load_cached(aug_name)

    print(f"HAGrid baseline: {len(X_h_base)}")
    print(f"HAGrid margin=0.20: {len(X_h_margin)}")
    print(f"HAGrid flip+margin: {len(X_h_aug)}")

    configs = [
        # (name, X_h, y_h, pca_n, C, gamma)
        ("Baseline PCA=80", X_h_base, y_h_base, 80, 10, 'scale'),
        ("PCA=200", X_h_base, y_h_base, 200, 10, 'scale'),
        ("PCA=400", X_h_base, y_h_base, 400, 10, 'scale'),
        ("Margin PCA=80", X_h_margin, y_h_margin, 80, 10, 'scale'),
        ("Margin PCA=200", X_h_margin, y_h_margin, 200, 10, 'scale'),
        ("Margin PCA=400", X_h_margin, y_h_margin, 400, 10, 'scale'),
        ("Margin+Flip PCA=80", X_h_aug, y_h_aug, 80, 10, 'scale'),
        ("Margin+Flip PCA=200", X_h_aug, y_h_aug, 200, 10, 'scale'),
        ("Margin+Flip PCA=400", X_h_aug, y_h_aug, 400, 10, 'scale'),
        ("Margin+Flip PCA=200 C=100", X_h_aug, y_h_aug, 200, 100, 'scale'),
        ("Margin+Flip PCA=200 gamma=0.01", X_h_aug, y_h_aug, 200, 10, 0.01),
    ]

    results = []
    for name, X_h, y_h, pca_n, C, gamma in configs:
        r = evaluate(name, X_h, y_h, X_ex, y_ex, pca_n, C, gamma)
        results.append(r)
        print(f"  {name:<35} {r['acc']:.4f}  sv={r['sv']:<5}  pca={r['pca_n']:<3}")

    # Summary
    results.sort(key=lambda r: r['acc'], reverse=True)
    print(f"\n{'='*80}")
    print(f"{'RANK':<6} {'CONFIG':<40} {'ACC':<8} {'PCA':<6} {'SV':<6}")
    print(f"{'='*80}")
    for i, r in enumerate(results):
        print(f"{i+1:<6} {r['name']:<40} {r['acc']:.4f}   {r['pca_n']:<4} {r['sv']:<5}")

    best = results[0]
    print(f"\n{'='*80}")
    print(f"BEST: {best['name']} — {best['acc']:.4f}")
    print(f"{'='*80}")

    # Per-class for best and second-best
    for rank, r in enumerate([results[0], results[1]]):
        print(f"\n--- {r['name']} (rank {rank+1}, {r['acc']:.4f}) ---")
        for cls_id in sorted(GESTURE_LABELS.values()):
            mask = r['y_te'] == cls_id
            if mask.sum() > 0:
                ca = float(np.mean(r['pred'][mask] == r['y_te'][mask]))
                print(f"  {ID_TO_GESTURE[cls_id]:<15} {ca:.4f}  ({mask.sum():>4} samples)")
        print(f"  Confusion matrix:")
        cm = r['cm']
        header = " " * 14 + " ".join(f"{ID_TO_GESTURE[i]:>10}" for i in sorted(GESTURE_LABELS.values()))
        print(header)
        for i, row in enumerate(cm):
            print(f"  {ID_TO_GESTURE[i]:<12} " + " ".join(f"{v:>10}" for v in row))

    # Save best model
    out_dir = os.path.join(REPO_DIR, "models")
    joblib.dump(best['model'], os.path.join(out_dir, "gesture_svm.pkl"))
    joblib.dump(best['scaler'], os.path.join(out_dir, "gesture_scaler.pkl"))
    joblib.dump(best['pca'], os.path.join(out_dir, "gesture_pca.pkl"))
    print(f"\nBest model saved to {out_dir}/gesture_svm.pkl")

    # Also save tier-2 best if different
    if len(results) > 1 and results[1]['name'] != best['name']:
        r2 = results[1]
        joblib.dump(r2['model'], os.path.join(out_dir, "gesture_svm_tier2.pkl"))
        joblib.dump(r2['scaler'], os.path.join(out_dir, "gesture_scaler_tier2.pkl"))
        joblib.dump(r2['pca'], os.path.join(out_dir, "gesture_pca_tier2.pkl"))
        print(f"Tier-2 model saved to {out_dir}/gesture_svm_tier2.pkl")


if __name__ == "__main__":
    main()
