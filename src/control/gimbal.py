import time
from typing import Optional


AUTO_PORTS = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1",
              "/dev/ttyAMA0", "/dev/ttyS0", "/dev/ttyS1",
              "COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]


class GimbalController:
    def __init__(self, port: str = "auto",
                 baud: int = 115200, timeout: float = 0.1,
                 batch_commands: bool = True,
                 control_rate_hz: float = 100.0,
                 max_delta: float = 5.0,
                 pan_min: int = 0, pan_max: int = 180,
                 tilt_min: int = 45, tilt_max: int = 135,
                 center: int = 90):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.batch_commands = batch_commands
        self.control_interval = 1.0 / max(control_rate_hz, 1.0)
        self.max_delta = max_delta
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.center = center
        self._ser = None
        self.pan_angle = center
        self.tilt_angle = center
        self._pending_pan = center
        self._pending_tilt = center
        self._connected = False
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.imu_yaw = 0.0
        self._last_write = 0.0
        self._last_imu_poll = 0.0
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
        p = int(max(self.pan_min, min(self.pan_max, self._pending_pan)))
        t = int(max(self.tilt_min, min(self.tilt_max, self._pending_tilt)))
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
        clamped = max(float(self.pan_min), min(float(self.pan_max), angle))
        target = self.smooth_move(float(self.pan_angle), clamped,
                                  float(self.pan_min), float(self.pan_max))
        target = int(round(target))
        target = max(self.pan_min, min(self.pan_max, target))
        self.pan_angle = target
        if self.batch_commands:
            self._pending_pan = target
        else:
            self._send(f"PAN:{target}")

    def set_tilt(self, angle: float):
        clamped = max(float(self.tilt_min), min(float(self.tilt_max), angle))
        target = self.smooth_move(float(self.tilt_angle), clamped,
                                  float(self.tilt_min), float(self.tilt_max))
        target = int(round(target))
        target = max(self.tilt_min, min(self.tilt_max, target))
        self.tilt_angle = target
        if self.batch_commands:
            self._pending_tilt = target
        else:
            self._send(f"TILT:{target}")

    def flush(self):
        if not self.batch_commands:
            return
        self._flush_batch()

    def set_pan_delta(self, delta: float):
        target = self.pan_angle + delta
        self.set_pan(target)

    def set_tilt_delta(self, delta: float):
        target = self.tilt_angle + delta
        self.set_tilt(target)

    def poll_imu(self):
        """Send STATUS command and parse IMU orientation from response (non-blocking).
        Called periodically (every N frames) to keep IMU data fresh.
        Uses a deferred read pattern: reads any pending response from the previous
        STATUS call, then sends a new request. Never blocks on I/O.
        Sets imu_pitch, imu_roll, imu_yaw on the controller instance."""
        if not self._connected or not self._ser:
            return
        # Read any pending response from the previous STATUS call
        self._read_response()
        # Send new STATUS request
        try:
            self._ser.write(b"STATUS\n")
        except Exception as e:
            print(f"IMU poll failed: {e}")

    def home(self):
        self.set_pan(self.center)
        self.set_tilt(self.center)

    def home_immediate(self):
        """Send HOME command directly to Arduino, bypassing smooth_move.
        Used during cleanup to ensure gimbal returns to center immediately
        regardless of current angle."""
        if self._ser and self._connected:
            try:
                self._ser.write(b"HOME\n")
            except Exception as e:
                print(f"Serial write failed during home_immediate: {e}")
        self.pan_angle = self.center
        self.tilt_angle = self.center

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


