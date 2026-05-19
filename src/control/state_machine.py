from enum import Enum


class Mode(Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    TRACKING_HAND = "TRACKING_HAND"
    LOCKED = "LOCKED"
    HOME = "HOME"
    SEARCH = "SEARCH"


class StateMachine:
    def __init__(self, idle_timeout_frames: int = 150,
                 gesture_hold_frames: int = 5, search_duration: int = 90):
        self.mode = Mode.IDLE
        self.idle_timeout = idle_timeout_frames
        self.gesture_hold_frames = gesture_hold_frames
        self._last_face_frame = 0
        self._frame_count = 0
        self._current_gesture = "NONE"
        self._previous_gesture = "NONE"
        self._gesture_hold_counter = 0
        self._search_frames = 0
        self._search_duration = search_duration
        self._pre_lock_mode = Mode.TRACKING
        self.zoom_level = 1.0
        self.zoom_mode_active = False
        self.last_hand_seen_frame = 0
        self.hand_timeout_frames = 30

    def process_gesture(self, gesture: str):
        self._current_gesture = gesture

        if gesture == "NONE":
            self._gesture_hold_counter = 0
            self._previous_gesture = gesture
            return

        if gesture != self._previous_gesture:
            self._gesture_hold_counter = 0
            self._previous_gesture = gesture

        self._gesture_hold_counter += 1

        if self._gesture_hold_counter >= self.gesture_hold_frames:
            if gesture == "OPEN_PALM" and self.mode in (
                    Mode.TRACKING, Mode.TRACKING_HAND):
                self._pre_lock_mode = self.mode
                self.mode = Mode.LOCKED
            elif gesture == "FIST" and self.mode == Mode.LOCKED:
                self.mode = self._pre_lock_mode
            elif gesture == "THUMBS_UP":
                self.mode = Mode.HOME
            elif gesture == "PEACE" and self.mode in (
                    Mode.TRACKING, Mode.TRACKING_HAND, Mode.LOCKED):
                self.zoom_mode_active = not self.zoom_mode_active
                if not self.zoom_mode_active:
                    self.zoom_level = 1.0
            elif gesture == "POINT" and self.mode in (
                    Mode.TRACKING, Mode.TRACKING_HAND):
                self.mode = Mode.TRACKING_HAND if self.mode == Mode.TRACKING else Mode.TRACKING

    def update_hand_zoom(self, defect_count: int):
        if not self.zoom_mode_active:
            return
        if defect_count >= 4:
            self.zoom_level = 1.0
        elif defect_count == 3:
            self.zoom_level = 1.3
        elif defect_count == 2:
            self.zoom_level = 1.7
        elif defect_count == 1:
            self.zoom_level = 2.5
        else:
            self.zoom_level = 3.0

    def update_face_status(self, face_detected: bool):
        self._frame_count += 1
        if face_detected:
            self._last_face_frame = self._frame_count
            self._search_frames = 0
            if self.mode == Mode.IDLE:
                self.mode = Mode.TRACKING
            elif self.mode == Mode.SEARCH:
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

    def update_hand_status(self, hand_detected: bool):
        if hand_detected:
            self.last_hand_seen_frame = self._frame_count

    def finish_homing(self):
        if self.mode == Mode.HOME:
            self.mode = Mode.TRACKING

    def toggle_lock(self):
        if self.mode in (Mode.TRACKING, Mode.TRACKING_HAND):
            self._pre_lock_mode = self.mode
            self.mode = Mode.LOCKED
        elif self.mode == Mode.LOCKED:
            self.mode = self._pre_lock_mode

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

    @property
    def search_progress(self) -> float:
        if self._search_duration <= 0:
            return 0.0
        return min(1.0, self._search_frames / self._search_duration)

    @property
    def tracking_target(self) -> str:
        if self.mode == Mode.TRACKING_HAND:
            return "HAND"
        return "FACE"
