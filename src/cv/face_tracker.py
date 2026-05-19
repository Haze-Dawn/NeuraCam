import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2


@dataclass
class Face:
    bbox: BoundingBox
    confidence: float


def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    x_overlap = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    y_overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    intersection = x_overlap * y_overlap
    if intersection == 0:
        return 0.0
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class Track:
    def __init__(self, face: Face, track_id: int):
        self.track_id = track_id
        self.lost_count = 0
        self.history = [face]
        self.face = face

        cx = face.bbox.center_x
        cy = face.bbox.center_y
        self.state = np.array([cx, cy, face.bbox.w, face.bbox.h,
                               0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(8) * 10.0

    @property
    def age(self) -> int:
        return len(self.history)


class KalmanTracker:
    def __init__(self, process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 max_lost_frames: int = 5,
                 iou_threshold: float = 0.3):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self.next_id = 0
        self.primary_id: Optional[int] = None

        self.F = np.eye(8)
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)
        self.R = np.eye(4) * measurement_noise

    def _build_process_noise(self, dt: float) -> np.ndarray:
        Q = np.eye(8) * self.process_noise
        Q[4:, 4:] *= dt * dt
        Q[:4, :4] *= dt
        return Q

    def _predict(self, track: Track, dt: float):
        # Clamp dt to prevent state explosion during lag spikes
        # (e.g., system suspend, CPU throttling, or breakpoints).
        # At 30fps, normal dt ≈ 0.033s. 0.1s = ~3 missed frames max.
        dt = min(dt, 0.1)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt
        Q = self._build_process_noise(dt)
        track.state = self.F @ track.state
        track.covariance = self.F @ track.covariance @ self.F.T + Q

    def _update(self, track: Track, measurement: np.ndarray):
        innovation = measurement - self.H @ track.state
        S = self.H @ track.covariance @ self.H.T + self.R
        K = track.covariance @ self.H.T @ np.linalg.inv(S)
        track.state = track.state + K @ innovation
        track.covariance = (np.eye(8) - K @ self.H) @ track.covariance

    def _get_smoothed_bbox(self, track: Track) -> BoundingBox:
        cx, cy, w, h = track.state[:4]
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return BoundingBox(x=max(0, x), y=max(0, y),
                           w=max(10, int(w)), h=max(10, int(h)))

    def update(self, detections: list[Face], dt: float = 0.033) -> Optional[Face]:
        for track in self.tracks:
            self._predict(track, dt)

        matched_detections = set()
        for track in self.tracks:
            best_iou = self.iou_threshold
            best_det = None
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                iou = compute_iou(
                    BoundingBox(
                        x=int(track.state[0] - track.state[2] / 2),
                        y=int(track.state[1] - track.state[3] / 2),
                        w=int(track.state[2]),
                        h=int(track.state[3]),
                    ),
                    det.bbox,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_det = i

            if best_det is not None:
                det = detections[best_det]
                matched_detections.add(best_det)
                z = np.array([det.bbox.center_x, det.bbox.center_y,
                              float(det.bbox.w), float(det.bbox.h)], dtype=np.float64)
                self._update(track, z)
                track.face = det
                track.history.append(det)
                if len(track.history) > 30:
                    track.history.pop(0)
                track.lost_count = 0
            else:
                track.lost_count += 1

        for i, det in enumerate(detections):
            if i not in matched_detections:
                new_track = Track(det, self.next_id)
                self.next_id += 1
                self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks
                       if t.lost_count <= self.max_lost_frames]

        if not self.tracks:
            self.primary_id = None
            return None

        best_track = max(self.tracks, key=lambda t: t.age)
        self.primary_id = best_track.track_id
        smoothed = self._get_smoothed_bbox(best_track)
        return Face(bbox=smoothed, confidence=best_track.face.confidence)

    def reset(self):
        self.tracks.clear()
        self.primary_id = None

    @property
    def uncertainty(self) -> float:
        if not self.tracks:
            return 1.0
        primary = next((t for t in self.tracks
                        if t.track_id == self.primary_id), self.tracks[0])
        return float(np.trace(primary.covariance[:4, :4]) / 4)
