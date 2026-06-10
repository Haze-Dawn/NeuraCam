"""Minimal timed SVM optimization - 4 key configs only."""
import os, sys, time, cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.train_gesture_hagrid import (
    GESTURE_LABELS, ID_TO_GESTURE, load_existing_csv,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(REPO, "src", "data", "cache")
csv_path = os.path.join(REPO, "data/gesture/raw/features_2500.csv")


def load_cached(name):
    return (
        np.load(os.path.join(CACHE, f"{name}.npy")),
        np.load(os.path.join(CACHE, f"{name}_labels.npy")),
    )


def run_cfg(name, X_h, y_h, X_ex, y_ex, pca_n, C=10, gamma='scale'):
    t0 = time.time()
    X = np.vstack([X_h, X_ex])
    y = np.concatenate([y_h, y_ex])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n = min(pca_n, X_tr.shape[1], len(X_tr) - 1)
    pca = PCA(n_components=n)
    X_tr_p = pca.fit_transform(X_tr)
    X_te_p = pca.transform(X_te)
    t1 = time.time()

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_p)
    X_te_s = sc.transform(X_te_p)

    svm = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42)
    svm.fit(X_tr_s, y_tr)
    t2 = time.time()

    pred = svm.predict(X_te_s)
    acc = float(np.mean(pred == y_te))
    cm = confusion_matrix(y_te, pred)

    print(f"\n{name}")
    print(f"  PCA({n}): {t1-t0:.1f}s | SVM: {t2-t1:.1f}s | Total: {t2-t0:.1f}s")
    print(f"  Accuracy: {acc:.4f} | Support vectors: {len(svm.support_)}")
    for cls_id in sorted(GESTURE_LABELS.values()):
        mask = y_te == cls_id
        ca = float(np.mean(pred[mask] == y_te[mask]))
        print(f"    {ID_TO_GESTURE[cls_id]:<15} {ca:.4f}")
    header = " " * 14 + " ".join(f"{ID_TO_GESTURE[i]:>10}" for i in sorted(GESTURE_LABELS.values()))
    print(f"  {header}")
    for i, row in enumerate(cm):
        print(f"  {ID_TO_GESTURE[i]:<12} " + " ".join(f"{v:>10}" for v in row))

    return acc, svm, sc, pca


def main():
    print("=" * 70)
    print("TIMED SVM OPTIMIZATION (4 key configs)")
    print("=" * 70)

    X_ex, y_ex = load_existing_csv(csv_path)
    print(f"User data: {len(X_ex)} samples")

    # Find cache files
    base_file = [f for f in os.listdir(CACHE) if f.startswith("hagrid_m0.00_hog") and f.endswith(".npy") and "_labels" not in f][0]
    margin_file = [f for f in os.listdir(CACHE) if f.startswith("hagrid_m0.20_hog") and f.endswith(".npy") and "_labels" not in f][0]
    aug_file = [f for f in os.listdir(CACHE) if f.startswith("hagrid_aug_m0.20") and f.endswith(".npy") and "_labels" not in f][0]

    X_h_b, y_h_b = load_cached(base_file.replace(".npy", ""))
    X_h_m, y_h_m = load_cached(margin_file.replace(".npy", ""))
    X_h_a, y_h_a = load_cached(aug_file.replace(".npy", ""))

    print(f"HAGrid baseline: {len(X_h_b)}")
    print(f"HAGrid margin=0.20: {len(X_h_m)}")
    print(f"HAGrid augment: {len(X_h_a)}")

    configs = [
        ("1. Baseline PCA=80", X_h_b, y_h_b, 80),
        ("2. Baseline PCA=200", X_h_b, y_h_b, 200),
        ("3. Margin PCA=200", X_h_m, y_h_m, 200),
        ("4. Margin+Flip PCA=200", X_h_a, y_h_a, 200),
        ("5. Margin+Flip PCA=400", X_h_a, y_h_a, 400),
    ]

    all_results = []
    for name, X_h, y_h, pca_n in configs:
        acc, svm, sc, pca = run_cfg(name, X_h, y_h, X_ex, y_ex, pca_n)
        all_results.append((name, acc, svm, sc, pca))

    print(f"\n{'='*70}")
    all_results.sort(key=lambda r: r[1], reverse=True)
    for name, acc, _, _, _ in all_results:
        print(f"  {name:<30} {acc:.4f}")

    best = all_results[0]
    out = os.path.join(REPO, "models")
    joblib.dump(best[2], os.path.join(out, "gesture_svm.pkl"))
    joblib.dump(best[3], os.path.join(out, "gesture_scaler.pkl"))
    joblib.dump(best[4], os.path.join(out, "gesture_pca.pkl"))
    print(f"\nBest: {best[0]} — {best[1]:.4f}")
    print(f"Saved to {out}/gesture_svm.pkl")


if __name__ == "__main__":
    main()
