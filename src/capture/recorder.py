import cv2
import numpy as np
from typing import Optional


class Recorder:
    def __init__(self, output_path: str = "output.mp4",
                 fps: float = 30.0, width: int = 640, height: int = 480):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps,
            (self.width, self.height)
        )
        self._recording = True

    def write_frame(self, frame: np.ndarray):
        if self._recording and self._writer is not None:
            resized = cv2.resize(frame, (self.width, self.height))
            self._writer.write(resized)

    def stop(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False

    @property
    def recording(self) -> bool:
        return self._recording
