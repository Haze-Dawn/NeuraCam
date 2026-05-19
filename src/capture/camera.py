import cv2
import numpy as np
import time
import threading
import queue
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
                 processing_height: int = 480,
                 use_capture_thread: bool = True):
        self.source = source
        self.width = width
        self.height = height
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.target_fps = fps
        self.use_capture_thread = use_capture_thread
        self._cap = None
        self._frame_count = 0
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connect()
        if use_capture_thread:
            self._start_thread()

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

    def _capture_loop(self):
        while self._running:
            if self._cap is None:
                time.sleep(0.01)
                continue
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            ts = time.time()
            try:
                self._queue.put_nowait((frame, ts))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait((frame, ts))
                except queue.Empty:
                    pass

    def _start_thread(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def read(self) -> Optional[Frame]:
        if self.use_capture_thread:
            try:
                data, ts = self._queue.get_nowait()
                self._frame_count += 1
                return Frame(
                    data=data,
                    timestamp=ts,
                    frame_id=self._frame_count
                )
            except queue.Empty:
                return None
        else:
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
        self._running = False
        # Release the VideoCapture BEFORE joining the thread.
        # On Linux (V4L2), cap.read() can block indefinitely; releasing
        # the handle causes the blocking read to return (False, None)
        # or raise, unblocking the capture thread.
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def fps(self) -> float:
        return self.target_fps
