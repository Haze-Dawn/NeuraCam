import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


GAZE_CLASSES = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]


def gaze_vector_to_class(gaze_vx, gaze_vy, threshold=0.2):
    vector = np.array([gaze_vx, gaze_vy])
    norm = np.linalg.norm(vector)
    if norm < threshold:
        return 0
    angle = np.arctan2(gaze_vy, gaze_vx)
    if -np.pi * 0.75 < angle <= -np.pi * 0.25:
        return 1
    elif -np.pi * 0.25 < angle <= np.pi * 0.25:
        return 2
    elif np.pi * 0.25 < angle <= np.pi * 0.75:
        return 3
    else:
        return 4


class MPIIGazeDataset(Dataset):
    def __init__(self, data_dir, subject_ids=None, transform=None,
                 img_size=128, gaze_threshold=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.gaze_threshold = gaze_threshold
        self.samples = []

        if subject_ids is None:
            subject_ids = list(range(15))

        self._load_subjects(subject_ids)
        print(f"Loaded {len(self.samples)} samples for subjects {subject_ids}")

    def _load_subjects(self, subject_ids):
        for sid in subject_ids:
            subject_dir = os.path.join(
                self.data_dir, f"p{sid:02d}"
            )
            if not os.path.exists(subject_dir):
                continue
            label_path = os.path.join(subject_dir, "label.txt")
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    img_path = os.path.join(subject_dir, parts[0])
                    gaze_vx = float(parts[1])
                    gaze_vy = float(parts[2])
                    gaze_vz = float(parts[3])
                    label = gaze_vector_to_class(
                        gaze_vx, gaze_vy, self.gaze_threshold
                    )
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        if self.transform:
            img = self.transform(img)
        return torch.FloatTensor(img), label
