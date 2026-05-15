import time
from typing import Optional


AUTO_PORTS = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyAMA0",
              "COM3", "COM4", "COM5", "COM6"]


class GimbalController:
    def __init__(self, port: str = "auto",
                 baud: int = 115200, timeout: float = 0.1,
                 batch_commands: bool = True,
                 control_rate_hz: float = 100.0,
                 max_delta: float = 5.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.batch_commands = batch_commands
        self.control_interval = 1.0 / max(control_rate_hz, 1.0)
        self.max_delta = max_delta
        self._ser = None
        self.pan_angle = 90
        self.tilt_angle = 90
        self._pending_pan = 90
        self._pending_tilt = 90
        self._connected = False
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.imu_yaw = 0.0
        self._last_write = 0.0
        self._connect()

    def _connect(self):
        import serial
        ports = AUTO_PORTS if self.port == "auto" else [self.port]
        for p in ports:
            try:
                self._ser = serial.Serial(
                    port=p,
                    baudrate=self.baud,
                    timeout=self.timeout
                )
                self.port = p
                self._connected = True
                self._read_response()
                print(f"Serial connected on {p}")
                return
            except Exception:
                continue
        print("Serial connection failed on all ports, no hardware control")
        self._connected = False

    def _read_response(self):
        if self._ser and self._ser.in_waiting:
            try:
                data = self._ser.read_all().decode().strip()
                self._parse_status(data)
                return data
            except Exception:
                return ""
        return ""

    def _parse_status(self, data: str):
        if "IMU_PITCH:" in data:
            try:
                parts = data.split()
                for p in parts:
                    if p.startswith("IMU_PITCH:"):
                        self.imu_pitch = float(p.split(":")[1])
                    elif p.startswith("IMU_ROLL:"):
                        self.imu_roll = float(p.split(":")[1])
                    elif p.startswith("IMU_YAW:"):
                        self.imu_yaw = float(p.split(":")[1])
            except Exception:
                pass

    def _send(self, cmd: str):
        now = time.monotonic()
        if now - self._last_write < self.control_interval:
            return
        if self._ser and self._connected:
            try:
                self._ser.write((cmd + "\n").encode())
                self._last_write = now
            except Exception as e:
                print(f"Serial write failed: {e}")
                self._connected = False

    def _flush_batch(self):
        if not self.batch_commands:
            return
        p = int(max(0, min(180, self._pending_pan)))
        t = int(max(45, min(135, self._pending_tilt)))
        self.pan_angle = p
        self.tilt_angle = t
        self._send(f"P:{p} T:{t}")

    def smooth_move(self, current: float, target: float, limit_min: float, limit_max: float) -> float:
        raw_delta = target - current
        if abs(raw_delta) <= self.max_delta:
            return target

        # Move toward target by max_delta, but slow down near limits
        step = self.max_delta if raw_delta > 0 else -self.max_delta
        candidate = current + step

        # Soft endstops: slow down as we approach limits
        dist_to_min = abs(candidate - limit_min)
        dist_to_max = abs(candidate - limit_max)
        nearest = min(dist_to_min, dist_to_max)

        if nearest <= 2.0:
            step *= 0.1  # crawl at 10% speed near limit
        elif nearest <= 10.0:
            fraction = (nearest - 2.0) / 8.0
            step *= 0.1 + 0.9 * fraction  # linear ramp 10%→100%

        candidate = current + step
        candidate = max(limit_min, min(limit_max, candidate))
        return candidate

    def set_pan(self, angle: float):
        clamped = max(0.0, min(180.0, angle))
        target = self.smooth_move(float(self.pan_angle), clamped, 0.0, 180.0)
        target = int(round(target))
        target = max(0, min(180, target))
        if self.batch_commands:
            self._pending_pan = target
            self._flush_batch()
        else:
            self.pan_angle = target
            self._send(f"PAN:{target}")

    def set_tilt(self, angle: float):
        clamped = max(45.0, min(135.0, angle))
        target = self.smooth_move(float(self.tilt_angle), clamped, 45.0, 135.0)
        target = int(round(target))
        target = max(45, min(135, target))
        if self.batch_commands:
            self._pending_tilt = target
            self._flush_batch()
        else:
            self.tilt_angle = target
            self._send(f"TILT:{target}")

    def set_pan_delta(self, delta: float):
        target = self.pan_angle + delta
        self.set_pan(target)

    def set_tilt_delta(self, delta: float):
        target = self.tilt_angle + delta
        self.set_tilt(target)

    def home(self):
        self.set_pan(90)
        self.set_tilt(90)

    def status(self) -> str:
        return (f"PAN:{self.pan_angle} TILT:{self.tilt_angle} "
                f"IMU_PITCH:{self.imu_pitch:.1f} IMU_ROLL:{self.imu_roll:.1f}")

    def close(self):
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            self._connected = False

    def __del__(self):
        self.close()
