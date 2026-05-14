import time
import sys
from typing import Optional


AUTO_PORTS = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyAMA0",
              "COM3", "COM4", "COM5", "COM6"]


class GimbalController:
    def __init__(self, port: str = "auto",
                 baud: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self.pan_angle = 90
        self.tilt_angle = 90
        self._connected = False
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.imu_yaw = 0.0
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
        print(f"Serial connection failed on all ports, no hardware control")
        self._connected = False

    def _read_response(self):
        if self._ser and self._ser.in_waiting:
            data = self._ser.read_all().decode().strip()
            self._parse_status(data)
            return data
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
        if self._ser and self._connected:
            try:
                self._ser.write((cmd + "\n").encode())
            except Exception as e:
                print(f"Serial write failed: {e}")
                self._connected = False

    def set_pan(self, angle: float):
        angle = int(max(0, min(180, angle)))
        self.pan_angle = angle
        self._send(f"PAN:{angle}")

    def set_tilt(self, angle: float):
        angle = int(max(45, min(135, angle)))
        self.tilt_angle = angle
        self._send(f"TILT:{angle}")

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
            except:
                pass
            self._ser = None
            self._connected = False

    def __del__(self):
        self.close()
