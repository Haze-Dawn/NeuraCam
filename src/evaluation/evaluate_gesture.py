import os
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import confusion_matrix, classification_report


GESTURE_LABELS = {
    "OPEN_PALM": 0, "FIST": 1, "THUMBS_UP": 2,
    "POINT": 3, "PEACE": 4,
}
ID_TO_GESTURE = {v: k for k, v in GESTURE_LABELS.items()}


def evaluate_gesture(data_csv: str, model_path: str,
                     scaler_path: str, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_csv)
    X = df.iloc[:, 1:].values
    y = df["gesture"].map(GESTURE_LABELS).values

    svm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    y_pred = svm.predict(X_scaled)
    accuracy = np.mean(y_pred == y)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(
        y, y_pred,
        target_names=list(GESTURE_LABELS.keys()),
        output_dict=True
    )

    results = {
        "accuracy": float(accuracy),
        "num_samples": int(len(y)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    report_path = os.path.join(output_dir, "logs", "gesture_evaluation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Gesture evaluation saved to {report_path}")
    print(f"Accuracy: {accuracy:.4f}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(GESTURE_LABELS.keys()),
                    yticklabels=list(GESTURE_LABELS.keys()),
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Gesture Confusion Matrix")
        fig_path = os.path.join(output_dir, "figures",
                                "gesture_confusion_matrix.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {fig_path}")
        plt.close()
    except ImportError:
        print("Matplotlib not available, skipping plot.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gesture/raw/landmarks.csv")
    parser.add_argument("--model", default="models/gesture_svm.pkl")
    parser.add_argument("--scaler", default="models/gesture_scaler.pkl")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    evaluate_gesture(args.data, args.model, args.scaler, args.output)
