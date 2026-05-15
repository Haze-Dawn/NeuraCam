from enum import Enum


class Mode(Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    LOCKED = "LOCKED"
    HOME = "HOME"


class StateMachine:
    def __init__(self, idle_timeout_frames: int = 150):
        self.mode = Mode.IDLE
        self.idle_timeout = idle_timeout_frames
        self._last_face_frame = 0
        self._frame_count = 0
        self._current_gesture = "NONE"

    def process_gesture(self, gesture: str):
        self._current_gesture = gesture
        if gesture == "OPEN_PALM" and self.mode == Mode.TRACKING:
            self.mode = Mode.LOCKED
        elif gesture == "FIST" and self.mode == Mode.LOCKED:
            self.mode = Mode.TRACKING
        elif gesture == "THUMBS_UP":
            self.mode = Mode.HOME

    def update_face_status(self, face_detected: bool):
        self._frame_count += 1
        if face_detected:
            self._last_face_frame = self._frame_count
            if self.mode == Mode.IDLE:
                self.mode = Mode.TRACKING
        elif self.mode == Mode.TRACKING:
            if (self._frame_count - self._last_face_frame
                    > self.idle_timeout):
                self.mode = Mode.IDLE

    def finish_homing(self):
        if self.mode == Mode.HOME:
            self.mode = Mode.TRACKING

    def toggle_lock(self):
        if self.mode == Mode.TRACKING:
            self.mode = Mode.LOCKED
        elif self.mode == Mode.LOCKED:
            self.mode = Mode.TRACKING

    @property
    def gesture(self) -> str:
        return self._current_gesture
