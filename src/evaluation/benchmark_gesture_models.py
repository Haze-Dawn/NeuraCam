import os
import sys
import numpy as np
import joblib
import json
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.train_gesture_hagrid import (
    GESTURE_LABELS, HOG_FEATURE_DIM, load_hagrid_data, load_existing_csv
)

ID_TO_GESTURE = {v: k for k, v in GESTURE_LABELS.items()}
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate_model(name, svm, scaler, pca, X_test, y_test):
    X_pca = pca.transform(X_test)
    X_scaled = scaler.transform(X_pca)
    pred = svm.predict(X_scaled)
    acc = float(np.mean(pred == y_test))
    return acc, pred


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hagrid-root",
                        default="/home/hazedawn/Documents/CV Project Rev2/Data/hagrid-sample-30k-384p")
    parser.add_argument("--existing-csv",
                        default=os.path.join(os.path.dirname(__file__),
                                "..", "data/gesture/raw/features_2500.csv"))
    parser.add_argument("--models-dir", default=MODELS_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("GESTURE MODEL BENCHMARK")
    print("=" * 70)

    print("\nLoading test data (HAGrid + existing)")
    X_h, y_h = load_hagrid_data(args.hagrid_root)
    X_ex, y_ex = load_existing_csv(args.existing_csv) if os.path.exists(args.existing_csv) else (np.empty((0, HOG_FEATURE_DIM)), np.empty(0))

    X_all = np.vstack([X_h, X_ex]) if len(X_ex) > 0 else X_h
    y_all = np.concatenate([y_h, y_ex]) if len(y_ex) > 0 else y_h
    print(f"Total: {len(X_all)} samples")

    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    print(f"Test set: {len(X_test)} samples")
    print(f"  Class distribution: {dict(sorted(Counter(y_test).items()))}")

    models_to_evaluate = [
        ("HAGrid+Existing (new)", args.models_dir, "", ""),
        ("User-Only 1500 (old)", args.models_dir, "_1500", "_1500"),
    ]

    results = []
    for label, model_dir, suffix_pkl, suffix_dir in models_to_evaluate:
        svm_path = os.path.join(model_dir, f"gesture_svm{suffix_pkl}.pkl")
        scaler_path = os.path.join(model_dir, f"gesture_scaler{suffix_pkl}.pkl")
        pca_path = os.path.join(model_dir, f"gesture_pca{suffix_pkl}.pkl")

        if not all(os.path.exists(p) for p in [svm_path, scaler_path, pca_path]):
            print(f"  Skipping {label}: files not found")
            continue

        svm = joblib.load(svm_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        acc, pred = evaluate_model(label, svm, scaler, pca, X_test, y_test)
        results.append((label, acc, pred, svm_path))
        print(f"  {label}: {acc:.4f}")

    print(f"\n{'Model':<35} {'Accuracy':<10} {'Support Vectors':<18} {'Size':<10}")
    print("-" * 75)
    for label, acc, pred, svm_path in results:
        svm = joblib.load(svm_path)
        n_sv = len(svm.support_)
        size = os.path.getsize(svm_path) / 1024
        print(f"{label:<35} {acc:.4f}     {n_sv:<16} {size:.0f} KB")

    print(f"\n--- Per-class accuracy ---")
    for label, acc, pred, _ in results:
        print(f"\n{label} (overall: {acc:.4f}):")
        for cls_id in sorted(GESTURE_LABELS.values()):
            mask = y_test == cls_id
            if mask.sum() > 0:
                cls_acc = float(np.mean(pred[mask] == y_test[mask]))
                print(f"  {ID_TO_GESTURE[cls_id]:<15} {cls_acc:.4f}  ({mask.sum():>4} samples)")

    print(f"\n--- Confusion Matrices ---")
    for label, acc, pred, _ in results:
        cm = confusion_matrix(y_test, pred)
        print(f"\n{label}:")
        header = " " * 14 + " ".join(f"{ID_TO_GESTURE[i]:>10}" for i in sorted(GESTURE_LABELS.values()))
        print(header)
        for i, row in enumerate(cm):
            print(f"  {ID_TO_GESTURE[i]:<12} " + " ".join(f"{v:>10}" for v in row))

    print(f"\n--- Detailed Classification Report ---")
    for label, acc, pred, _ in results:
        print(f"\n{label} (Accuracy: {acc:.4f}):")
        print(classification_report(y_test, pred, target_names=list(GESTURE_LABELS.keys())))

    print(f"\n{'='*70}")
    winner = max(results, key=lambda r: r[1])
    print(f"WINNER: {winner[0]} with {winner[1]:.4f} accuracy")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
