from enum import Enum


class Mode(Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    LOCKED = "LOCKED"
    HOME = "HOME"
    SEARCH = "SEARCH"


class StateMachine:
    def __init__(self, idle_timeout_frames: int = 150, gesture_hold_frames: int = 5):
        self.mode = Mode.IDLE
        self.idle_timeout = idle_timeout_frames
        self.gesture_hold_frames = gesture_hold_frames
        self._last_face_frame = 0
        self._frame_count = 0
        self._current_gesture = "NONE"
        self._previous_gesture = "NONE"
        self._gesture_hold_counter = 0
        self._search_frames = 0
        self._search_duration = 90

    def process_gesture(self, gesture: str):
        self._current_gesture = gesture

        if gesture == "NONE":
            self._gesture_hold_counter = 0
            self._previous_gesture = gesture
            return

        if gesture != self._previous_gesture:
            self._gesture_hold_counter = 1
            self._previous_gesture = gesture
            return

        self._gesture_hold_counter += 1

        if self._gesture_hold_counter < self.gesture_hold_frames:
            return

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
            self._search_frames = 0
            if self.mode in (Mode.IDLE, Mode.SEARCH):
                self.mode = Mode.TRACKING
        elif self.mode == Mode.TRACKING:
            if (self._frame_count - self._last_face_frame
                    > self.idle_timeout):
                self.mode = Mode.SEARCH
                self._search_frames = 0
        elif self.mode == Mode.SEARCH:
            self._search_frames += 1
            if self._search_frames > self._search_duration:
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

    @property
    def gesture_hold_progress(self) -> float:
        if self.gesture_hold_frames <= 0:
            return 0.0
        return min(1.0, self._gesture_hold_counter / self.gesture_hold_frames)

    @property
    def search_active(self) -> bool:
        return self.mode == Mode.SEARCH
