import numpy as np
from typing import Optional, Tuple
from src.cv.face_tracker import BoundingBox


class PIDController:
    def __init__(self, Kp: float = 2.0, Ki: float = 0.05, Kd: float = 0.5,
                 output_limits: Tuple[float, float] = (-30, 30),
                 integral_limit: float = 10.0, dt: float = 0.033):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min, self.output_max = output_limits
        self.integral_limit = integral_limit
        self.dt = dt
        self.alpha = 0.1

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def update(self, error: float, dt: Optional[float] = None) -> float:
        if dt is not None:
            self.dt = max(dt, 0.001)

        p_term = self.Kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral,
                                -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral

        raw_derivative = (error - self.prev_error) / self.dt
        self.filtered_derivative = (
            self.alpha * raw_derivative +
            (1 - self.alpha) * self.filtered_derivative
        )
        d_term = self.Kd * self.filtered_derivative

        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)

        self.prev_error = error
        return float(output)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0


def compute_adaptive_dead_zone(face_bbox: BoundingBox, frame_size: Tuple[int, int],
                                base_ratio: float = 0.25,
                                min_dz: float = 0.02,
                                max_dz: float = 0.10) -> float:
    face_w = face_bbox.w
    frame_w = frame_size[0]
    if frame_w <= 0 or face_w <= 0:
        return min_dz
    ratio = face_w / frame_w
    dead_zone = np.clip(ratio * base_ratio, min_dz, max_dz)
    return float(dead_zone)
