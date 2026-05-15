import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FaceFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class WIDERFaceDataset(Dataset):
    def __init__(self, root_dir, split="train", input_size=128):
        self.root_dir = root_dir
        self.input_size = input_size
        self.samples = []

        img_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        annot_file = os.path.join(root_dir, "wider_face_split",
                                  f"wider_face_{split}_bbx_gt.txt")

        if not os.path.exists(annot_file):
            print(f"Annotation file not found: {annot_file}")
            return

        with open(annot_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            i += 1
            if i >= len(lines):
                break
            num_faces = int(lines[i].strip())
            i += 1

            img_path = None
            for subdir in os.listdir(img_dir):
                candidate = os.path.join(img_dir, subdir, img_name)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                i += num_faces
                continue

            faces = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    if w > 0 and h > 0:
                        faces.append((x, y, w, h))

            if faces:
                self.samples.append((img_path, faces))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, faces = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        patch = cv2.resize(img, (self.input_size, self.input_size))
        patch = patch.astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).permute(2, 0, 1)

        label = 0.0
        bbox_target = torch.zeros(3)
        if faces:
            fx, fy, fw, fh = faces[0]
            cx = (fx + fw / 2) / w
            cy = (fy + fh / 2) / h
            size = max(fw, fh) / max(w, h)
            bbox_target = torch.tensor([cx - 0.5, cy - 0.5, np.log(max(size, 0.01))])
            label = 1.0

        return patch, torch.tensor([label]), bbox_target


def train_epoch(model, loader, optimizer, criterion_cls, criterion_bbox, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for patches, labels, bboxes in tqdm(loader, desc="Training"):
        patches = patches.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        outputs = model(patches)

        cls_logits = outputs[:, 0].mean(dim=(1, 2))
        cls_loss = criterion_cls(cls_logits, labels.squeeze())

        bbox_pred = outputs[:, 1:].mean(dim=(2, 3))
        pos_mask = labels.squeeze() > 0.5
        if pos_mask.sum() > 0:
            bbox_loss = criterion_bbox(bbox_pred[pos_mask], bboxes[pos_mask])
        else:
            bbox_loss = torch.tensor(0.0, device=device)

        loss = cls_loss + bbox_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(cls_logits) > 0.5).float()
        correct += (preds == labels.squeeze()).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def hard_negative_mine(model, loader, device, top_k=500):
    model.eval()
    false_positives = []

    with torch.no_grad():
        for patches, labels, _ in tqdm(loader, desc="Hard-negative mining"):
            patches = patches.to(device)
            labels = labels.to(device)
            outputs = model(patches)
            cls_logits = outputs[:, 0].mean(dim=(1, 2))
            probs = torch.sigmoid(cls_logits)

            neg_mask = labels.squeeze() < 0.5
            false_pos = (probs > 0.3) & neg_mask
            if false_pos.any():
                indices = torch.where(false_pos)[0]
                for idx in indices:
                    false_positives.append((patches[idx].cpu(), probs[idx].item()))

    false_positives.sort(key=lambda x: x[1], reverse=True)
    return [fp[0] for fp in false_positives[:top_k]]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/face/widerface")
    parser.add_argument("--output", default="models/face_cnn.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = WIDERFaceDataset(args.data, "train", args.input_size)
    val_dataset = WIDERFaceDataset(args.data, "val", args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = FaceFCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_bbox = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_bbox, device
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, labels, bboxes in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)
                outputs = model(patches)
                cls_logits = outputs[:, 0].mean(dim=(1, 2))
                cls_loss = criterion_cls(cls_logits, labels.squeeze())
                val_loss += cls_loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}")

        hard_negatives = hard_negative_mine(model, val_loader, device, top_k=200)
        if hard_negatives and epoch % 3 == 0:
            hard_dataset = [(hn, torch.tensor([0.0]),
                            torch.zeros(3)) for hn in hard_negatives]
            if len(hard_dataset) > 0:
                extra_loader = DataLoader(hard_dataset, batch_size=args.batch_size,
                                          shuffle=True)
                train_epoch(model, extra_loader, optimizer, criterion_cls,
                            criterion_bbox, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.output)
            print(f"Model saved to {args.output}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best model: {args.output}")


if __name__ == "__main__":
    main()
