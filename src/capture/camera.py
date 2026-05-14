import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Frame:
    data: np.ndarray
    timestamp: float
    frame_id: int


class Camera:
    def __init__(self, source: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 30,
                 processing_width: int = 640,
                 processing_height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.target_fps = fps
        self._cap = None
        self._frame_count = 0
        self._connect()

    def _connect(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w > 0 and actual_h > 0:
            self.width = actual_w
            self.height = actual_h

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        self._frame_count += 1
        return Frame(
            data=frame,
            timestamp=time.time(),
            frame_id=self._frame_count
        )

    def get_processing_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self.processing_width, self.processing_height),
                          interpolation=cv2.INTER_LINEAR)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self.target_fps
