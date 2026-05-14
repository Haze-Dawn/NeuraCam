import numpy as np
import torch


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            return img.flip(-1)
        return img


class RandomRotation:
    def __init__(self, max_angle=5, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            _, h, w = img.shape
            center = (w / 2, h / 2)
            import torchvision.transforms.functional as TF
            return TF.rotate(img, angle)
        return img


class RandomBrightness:
    def __init__(self, max_delta=0.1, p=0.5):
        self.max_delta = max_delta
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            delta = np.random.uniform(-self.max_delta, self.max_delta)
            return torch.clamp(img + delta, 0, 1)
        return img
