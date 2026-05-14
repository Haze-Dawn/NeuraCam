import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from src.training.gaze_dataset import MPIIGazeDataset
from src.training.train_gaze import GazeCNN


def evaluate_gaze(data_dir: str, model_path: str,
                  output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    all_subjects = list(range(15))
    all_preds = []
    all_labels = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GazeCNN(num_classes=5).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    model.eval()

    for test_subject in all_subjects:
        test_dataset = MPIIGazeDataset(
            data_dir, subject_ids=[test_subject], transform=None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=0
        )

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["CENTER", "LEFT", "RIGHT", "UP", "DOWN"],
        output_dict=True
    )

    results = {
        "accuracy": float(accuracy),
        "num_samples": int(len(all_labels)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    report_path = os.path.join(output_dir, "logs", "gaze_evaluation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Gaze evaluation saved to {report_path}")
    print(f"Accuracy: {accuracy:.4f}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["CENTER", "LEFT", "RIGHT", "UP", "DOWN"],
                    yticklabels=["CENTER", "LEFT", "RIGHT", "UP", "DOWN"],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Gaze Confusion Matrix")
        fig_path = os.path.join(output_dir, "figures", "gaze_confusion_matrix.png")
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
    parser.add_argument("--data", default="data/gaze/mpiigaze")
    parser.add_argument("--model", default="models/gaze_cnn.pth")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    evaluate_gaze(args.data, args.model, args.output)
