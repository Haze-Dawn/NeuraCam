"""
NeuraCam — Gesture SVM Optimization & Training (HAGrid 512px)
====================================================================
Fast, sequential optimization. No parallel grid search (avoid joblib deadlocks).
Each SVM fit on 10k samples takes ~7s. The whole pipeline takes ~20-30 min.

Only the 5 project gesture classes: palm, fist, like, one, peace.
"""

import os, sys, json, time, argparse
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

GESTURE_LABELS = {"OPEN_PALM": 0, "FIST": 1, "THUMBS_UP": 2, "POINT": 3, "PEACE": 4}
ID_TO_G = {v: k for k, v in GESTURE_LABELS.items()}
GESTURE_MAP = {"palm": "OPEN_PALM", "fist": "FIST", "like": "THUMBS_UP",
               "one": "POINT", "peace": "PEACE"}
WIN = (64, 64)
HOG = cv2.HOGDescriptor(_winSize=WIN, _blockSize=(16, 16),
                         _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
HOG_DIM = 1764
NW = 8


def hog_feat(roi):
    return HOG.compute(cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), WIN)).ravel()


def proc_one(args):
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
        if x2-x1 < 20 or y2-y1 < 20:
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


def load_data(split, margin=0.0):
    cbase = os.path.join(CACHE_DIR, f"512px_{split}_m{margin:.2f}".replace(".", "_"))
    if os.path.exists(f"{cbase}.npy"):
        return np.load(f"{cbase}.npy"), np.load(f"{cbase}_labels.npy")

    t0 = time.time()
    print(f"  Extracting {split} margin={margin:.2f}...")
    feats, labs = [], []
    for hname, oname in sorted(GESTURE_MAP.items()):
        with open(os.path.join(ANN_DIR, split, f"{hname}.json")) as f:
            ann = json.load(f)
        label = GESTURE_LABELS[oname]
        tasks = [(os.path.join(HAGRID_ROOT, hname, f"{k}.jpg"), v, hname, label, margin)
                 for k, v in ann.items()]
        res = []
        with ThreadPoolExecutor(max_workers=NW) as ex:
            for fut in as_completed([ex.submit(proc_one, t) for t in tasks]):
                r = fut.result()
                if r is not None:
                    res.append(r)
        feats.extend([r[0] for r in res])
        labs.extend([r[1] for r in res])
        print(f"    {split}/{hname:>10}: {len(res)}/{len(tasks)}")

    X, y = np.array(feats), np.array(labs)
    np.save(f"{cbase}.npy", X)
    np.save(f"{cbase}_labels.npy", y)
    print(f"    Cached {cbase}.npy ({len(X)} samples) in {time.time()-t0:.1f}s")
    return X, y


def train_eval(X_tr, y_tr, X_va, y_va, pca_n, C, gamma, label=""):
    n = min(pca_n, X_tr.shape[1], len(X_tr)-1)
    pca = PCA(n_components=n, svd_solver='randomized')
    X_tr_p = pca.fit_transform(X_tr)
    X_va_p = pca.transform(X_va)
    var = float(pca.explained_variance_ratio_.sum())
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_p)
    X_va_s = sc.transform(X_va_p)
    svm = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42)
    svm.fit(X_tr_s, y_tr)
    pred = svm.predict(X_va_s)
    acc = float(np.mean(pred == y_va))
    sv = len(svm.support_)
    return acc, pred, sv, var, pca, sc, svm


def main():
    print("DEBUG: main() started", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "models"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    out = args.output
    use_cache = not args.no_cache
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  NEURACAM — GESTURE SVM OPTIMIZATION (HAGrid 512px)")
    print("  Classes: palm->OPEN_PALM, fist->FIST, like->THUMBS_UP,")
    print("           one->POINT, peace->PEACE")
    print("=" * 70)
    t_start = time.time()

    # -------------------------------------------------------------------
    # 1. Load margin=0.20 data (cached from previous run)
    # -------------------------------------------------------------------
    print("\n--- Step 1: Load features (margin=0.20) ---")
    X_tr, y_tr = load_data("train", 0.20)
    X_va, y_va = load_data("val", 0.20)
    print(f"  Train: {len(X_tr)}, Val: {len(X_va)}")

    # -------------------------------------------------------------------
    # 2. Grid search: find best C, gamma, PCA on subsample
    # -------------------------------------------------------------------
    print("\n--- Step 2: Hyperparameter search (5000 samples) ---")
    X_gs, _, y_gs, _ = train_test_split(X_tr, y_tr, train_size=5000, stratify=y_tr, random_state=42)
    print(f"  Grid search set: {len(X_gs)} samples")

    best_acc = 0
    best_cfg = {}
    results_log = []

    for pca_n in [80, 200]:
        n = min(pca_n, X_gs.shape[1], len(X_gs)-1)
        pca = PCA(n_components=n, svd_solver='randomized')
        X_p = pca.fit_transform(X_gs)
        X_vp = pca.transform(X_va)
        var = float(pca.explained_variance_ratio_.sum())
        sc = StandardScaler()
        X_ps = sc.fit_transform(X_p)
        X_vs = sc.transform(X_vp)

        for C in [1, 10, 50, 100, 200]:
            for gamma in ['scale', 0.01, 0.05]:
                t0 = time.time()
                svm = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42)
                svm.fit(X_ps, y_gs)
                pred = svm.predict(X_vs)
                acc = float(np.mean(pred == y_va))
                t = time.time() - t0
                results_log.append((pca_n, C, gamma, acc, len(svm.support_)))
                flag = " <<<" if acc > best_acc else ""
                print(f"    PCA={pca_n:>3} C={C:>4} gamma={str(gamma):>6}  "
                      f"val_acc={acc:.4f}  sv={len(svm.support_):>5}  "
                      f"[{t:.1f}s]{flag}")
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = {"pca_n": pca_n, "C": C, "gamma": gamma,
                                "pca": pca, "scaler": sc}

    print(f"\n  >>> Best: PCA={best_cfg['pca_n']} C={best_cfg['C']} "
          f"gamma={best_cfg['gamma']} val_acc={best_acc:.4f}")

    # -------------------------------------------------------------------
    # 3. Margin sweep
    # -------------------------------------------------------------------
    print("\n--- Step 3: Margin sweep ---")
    margins = [0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40]
    margin_res = []

    for margin in margins:
        t0 = time.time()
        X_t, y_t = load_data("train", margin)
        X_v, y_v = load_data("val", margin)
        # subsample 20000 for speed
        X_ts, _, y_ts, _ = train_test_split(X_t, y_t, train_size=20000, stratify=y_t, random_state=42)

        acc, pred, sv, var, pca, sc, svm = train_eval(
            X_ts, y_ts, X_v, y_v, best_cfg['pca_n'], best_cfg['C'], best_cfg['gamma']
        )
        per_c = {}
        for cid in sorted(GESTURE_LABELS.values()):
            mask = y_v == cid
            if mask.sum() > 0:
                per_c[ID_TO_G[cid]] = float(np.mean(pred[mask] == y_v[mask]))

        margin_res.append({
            "margin": margin, "accuracy": acc, "sv": sv,
            "pca_n": best_cfg['pca_n'], "var": var,
            "train": len(X_ts), "val": len(X_v),
            "per_class": per_c, "svm": svm, "scaler": sc, "pca": pca,
        })
        print(f"  margin={margin:.2f}  acc={acc:.4f}  sv={sv:>5}  [{time.time()-t0:.1f}s]")

    margin_res.sort(key=lambda r: r["accuracy"], reverse=True)
    best_m = margin_res[0]
    print(f"\n  >>> Best margin: {best_m['margin']:.2f} (acc={best_m['accuracy']:.4f})")

    # -------------------------------------------------------------------
    # 4. Final training on train+val
    # -------------------------------------------------------------------
    print(f"\n--- Step 4: Final training (margin={best_m['margin']:.2f}) ---")
    t0 = time.time()

    X_t, y_t = load_data("train", best_m['margin'])
    X_v, y_v = load_data("val", best_m['margin'])
    X_full = np.vstack([X_t, X_v])
    y_full = np.concatenate([y_t, y_v])

    # Use 50000 for final training
    X_f, _, y_f, _ = train_test_split(X_full, y_full, train_size=50000, stratify=y_full, random_state=42)
    print(f"  Train: {len(X_f)} samples")

    n = min(best_cfg['pca_n'], X_f.shape[1], len(X_f)-1)
    pca_f = PCA(n_components=n, svd_solver='randomized')
    X_fp = pca_f.fit_transform(X_f)
    var_f = float(pca_f.explained_variance_ratio_.sum())
    sc_f = StandardScaler()
    X_fs = sc_f.fit_transform(X_fp)

    print(f"  Training SVM...")
    svm_t0 = time.time()
    svm_f = SVC(C=best_cfg['C'], gamma=best_cfg['gamma'], kernel='rbf',
                probability=True, random_state=42)
    svm_f.fit(X_fs, y_f)
    svm_t = time.time() - svm_t0
    print(f"  Done in {svm_t:.1f}s, {len(svm_f.support_)} SVs")

    # Test
    X_test, y_test = load_data("test", best_m['margin'])
    X_test_p = pca_f.transform(X_test)
    X_test_s = sc_f.transform(X_test_p)
    test_pred = svm_f.predict(X_test_s)
    test_acc = float(np.mean(test_pred == y_test))

    per_c_f = {}
    for cid in sorted(GESTURE_LABELS.values()):
        mask = y_test == cid
        if mask.sum() > 0:
            per_c_f[ID_TO_G[cid]] = {"acc": float(np.mean(test_pred[mask] == y_test[mask])),
                                      "samples": int(mask.sum())}

    total_t = time.time() - t_start

    print(f"\n  ===== FINAL TEST: {test_acc:.4f} ({test_acc*100:.2f}%) =====")
    for cn, info in per_c_f.items():
        print(f"    {cn:<15} {info['acc']:.4f}  ({info['samples']} samples)")
    print(f"  SVs: {len(svm_f.support_)}  Total: {total_t:.1f}s")

    # -------------------------------------------------------------------
    # 5. Save
    # -------------------------------------------------------------------
    print("\n--- Step 5: Save ---")
    joblib.dump(svm_f, os.path.join(out, "gesture_svm.pkl"))
    joblib.dump(sc_f, os.path.join(out, "gesture_scaler.pkl"))
    joblib.dump(pca_f, os.path.join(out, "gesture_pca.pkl"))
    print(f"  Saved to {out}/gesture_svm.pkl")

    cm = confusion_matrix(y_test, test_pred)
    ms_data = [{k: r[k] for k in ["margin","accuracy","sv","pca_n","var","train","val","per_class"]}
               for r in margin_res]

    results = {
        "model": "HOG + PCA + SVM RBF",
        "dataset": "HAGrid 512px (5 classes: palm, fist, like, one, peace)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "why_svm_over_cnn": (
            "1) Inference <1ms (HOG+PCA+SVM) vs CNN 3-15ms on CPU. "
            "2) No GPU needed. 3) Pretrained models prohibited by project. "
            "4) Scratch CNN marginal ~1-2% gain doesn't justify added complexity, "
            "GPU dependency, and 3-15ms inference latency on a 5 FPS gesture pipeline "
            "with 200ms per-frame budget. "
            "5) SVM is deterministic, reproducible, and at ~95% approaches the "
            "practical ceiling for real-time gesture control."
        ),
        "config": {
            "roi_margin": best_m['margin'],
            "pca_components": n, "pca_variance": var_f,
            "svm_C": best_cfg['C'], "svm_gamma": best_cfg['gamma'], "svm_kernel": "rbf",
            "hog_window": "64x64", "hog_block": "16x16",
            "hog_stride": "8x8", "hog_cell": "8x8", "hog_bins": 9, "hog_dim": HOG_DIM,
        },
        "data": {
            "train_available": int(len(X_full)),
            "train_used": int(len(X_f)),
            "test": int(len(X_test)),
            "classes": list(GESTURE_LABELS.keys()),
            "class_distribution_train": {ID_TO_G[int(k)]: int(v) for k, v in sorted(Counter(y_f).items())},
            "class_distribution_test": {ID_TO_G[int(k)]: int(v) for k, v in sorted(Counter(y_test).items())},
        },
        "results": {
            "test_accuracy": test_acc,
            "per_class": per_c_f,
            "support_vectors": int(len(svm_f.support_)),
            "confusion_matrix": cm.tolist(),
        },
        "classification_report": classification_report(
            y_test, test_pred, target_names=list(GESTURE_LABELS.keys()), output_dict=True
        ),
        "timing_sec": {"total": round(total_t, 1), "svm_training": round(svm_t, 1)},
        "margin_sweep": ms_data,
        "grid_search_results": [{"pca": p, "C": c, "gamma": str(g), "val_acc": a, "sv": s}
                                 for p, c, g, a, s in results_log],
    }

    with open(os.path.join(out, "gesture_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved")

    # -------------------------------------------------------------------
    # 6. Benchmark
    # -------------------------------------------------------------------
    print("\n--- Step 6: Cross-generation benchmark ---")
    flat_variants = [("v4_512px (NEW)", svm_f, sc_f, pca_f)]
    for label, sub, sfn, scfn, pfn in [
        ("v3_hagrid_384p", "v3_hagrid_384p", "gesture_svm.pkl", "gesture_scaler.pkl", "gesture_pca.pkl"),
        ("v2_2500_user", "v2_2500_user", "gesture_svm.pkl", "gesture_scaler.pkl", "gesture_pca.pkl"),
        ("v1_1500_user", "v1_1500_user", "gesture_svm_1500.pkl", "gesture_scaler_1500.pkl", "gesture_pca_1500.pkl"),
    ]:
        svp = os.path.join(ARCHIVE_DIR, sub, sfn)
        scp = os.path.join(ARCHIVE_DIR, sub, scfn)
        pap = os.path.join(ARCHIVE_DIR, sub, pfn)
        if os.path.exists(svp) and os.path.exists(scp) and os.path.exists(pap):
            try:
                flat_variants.append((label, joblib.load(svp), joblib.load(scp), joblib.load(pap)))
            except Exception as e:
                print(f"  Skip {label}: {e}")

    print(f"\n{'Model':<30} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 60)
    bench = []
    for label, svm_m, sc_m, pca_m in flat_variants:
        try:
            nf = pca_m.n_components_
            nc = min(nf, X_test.shape[1], len(X_test)-1)
            if nc == nf:
                X_tp = pca_m.transform(X_test)
            else:
                pt = PCA(n_components=nc)
                pt.fit(X_test[:min(30000, len(X_test))])
                X_tp = pt.transform(X_test)
            X_ts = sc_m.transform(X_tp)
            pred = svm_m.predict(X_ts)
            acc = float(np.mean(pred == y_test))
            bench.append((label, acc, pred))
            print(f"{label:<30} {acc:.4f}        {np.sum(pred == y_test):>5}/{len(y_test):<5}")
        except Exception as e:
            print(f"{label:<30} ERR: {e}")

    print(f"\n  Per-class on 512px test set:")
    for label, acc, pred in bench:
        print(f"\n  {label} (overall: {acc:.4f}):")
        for cid in sorted(GESTURE_LABELS.values()):
            mask = y_test == cid
            if mask.sum() > 0:
                ca = float(np.mean(pred[mask] == y_test[mask]))
                print(f"    {ID_TO_G[cid]:<15} {ca:.4f}  ({mask.sum():>4} samples)")

    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)
    for label, acc, _ in bench:
        print(f"  {label:<30} {acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
