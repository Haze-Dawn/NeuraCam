import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time

from src.training.gaze_dataset import MPIIGazeDataset
from src.training.augment import RandomHorizontalFlip, RandomRotation, RandomBrightness


class GazeCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (total_loss / len(loader), correct / total,
            np.array(all_preds), np.array(all_labels))


def train(data_dir: str, output_path: str = "models/gaze_cnn.pth",
          num_epochs: int = 50, batch_size: int = 64,
          lr: float = 0.001, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    all_subjects = list(range(15))
    subject_accs = []
    subject_results_dir = os.path.join(os.path.dirname(output_path), "subject_results")
    os.makedirs(subject_results_dir, exist_ok=True)

    for test_subject in all_subjects:
        print(f"\n{'='*40}")
        print(f"Fold: test on subject {test_subject}")
        print(f"{'='*40}")

        train_subjects = [s for s in all_subjects if s != test_subject]

        transform = torch.jit.script(
            nn.Sequential(
                RandomHorizontalFlip(p=0.5),
                RandomBrightness(max_delta=0.1, p=0.5),
            )
        )

        train_dataset = MPIIGazeDataset(
            data_dir, subject_ids=train_subjects, transform=transform
        )
        test_dataset = MPIIGazeDataset(
            data_dir, subject_ids=[test_subject], transform=None
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        model = GazeCNN(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, _, _ = evaluate(
                model, train_loader, criterion, device
            )
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        test_loss, test_acc, preds, labels = evaluate(
            model, test_loader, criterion, device
        )
        print(f"  Subject {test_subject}: Test Accuracy = {test_acc:.4f}")
        subject_accs.append(test_acc)

        fold_results = {
            "test_subject": test_subject,
            "accuracy": float(test_acc),
            "predictions": preds.tolist(),
            "labels": labels.tolist(),
        }
        with open(os.path.join(subject_results_dir,
                               f"subject_{test_subject}.json"), "w") as f:
            json.dump(fold_results, f, indent=2)

    mean_acc = np.mean(subject_accs)
    std_acc = np.std(subject_accs)
    print(f"\n{'='*40}")
    print(f"Leave-One-Person-Out Results:")
    print(f"Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"Per-subject accuracies: {[f'{a:.4f}' for a in subject_accs]}")

    full_dataset = MPIIGazeDataset(data_dir, subject_ids=all_subjects)
    final_loader = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    final_model = GazeCNN(num_classes=5).to(device)
    final_optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-4)
    final_criterion = nn.CrossEntropyLoss()

    print("\nTraining final model on all data...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            final_model, final_loader, final_criterion, final_optimizer, device
        )
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_model.eval()
    example_input = torch.randn(1, 3, 128, 128).to(device)
    traced_model = torch.jit.trace(final_model, example_input)
    traced_model.save(output_path)
    print(f"Final model saved to {output_path}")

    results = {
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "per_subject_accuracies": [float(a) for a in subject_accs],
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
    results_path = os.path.join(os.path.dirname(output_path), "gaze_training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gaze/mpiigaze")
    parser.add_argument("--output", default="models/gaze_cnn.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr)
