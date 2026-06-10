#!/usr/bin/env python3
"""
Mock NeuraCam data feed for testing the Rust TUI and GUI without hardware.

Simulates the full IPC protocol that src/main.py produces:
  - Unix socket at /tmp/neuracam.sock
  - State JSON every ~33ms (30fps)
  - Synthetic JPEG frame every ~33ms
  - Input socket at /tmp/neuracam_input.sock (keyboard echo)

Usage:
  python scripts/mock_neuracam.py [--fps 30] [--duration 0]

Then in another terminal:
  cd rust && cargo run --release -p neuracam-tui
  cd rust && cargo run --release -p neuracam-gui
"""

import argparse
import json
import math
import os
import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

SOCKET_STATE = "/tmp/neuracam.sock"
SOCKET_INPUT = "/tmp/neuracam_input.sock"
MSG_TYPE_STATE = 0
MSG_TYPE_FRAME = 1


def _pack_msg(msg_type: int, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + bytes([msg_type]) + payload


class MockNeuraCam:
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self._frame = 0
        self._mode_cycle = ["IDLE", "TRACKING", "TRACKING", "TRACKING",
                            "TRACKING_HAND", "TRACKING", "LOCKED",
                            "TRACKING", "SEARCH", "IDLE"]
        self._mode_idx = 0
        self._mode_hold = 0
        self._angle_pan = 90.0
        self._angle_tilt = 90.0
        self._target_pan = 90.0
        self._target_tilt = 90.0
        self._face_x = 640.0  # center of 1280x720
        self._face_y = 360.0
        self._face_vx = 2.0   # pixels per frame drift
        self._face_vy = -1.5
        self._pid_pan_error = 0.0
        self._pid_tilt_error = 0.0
        self._events = deque(maxlen=20)

        # IPC setup
        for p in [SOCKET_STATE, SOCKET_INPUT]:
            try:
                os.unlink(p)
            except OSError:
                pass

        self._state_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._state_server.bind(SOCKET_STATE)
        self._state_server.listen(5)
        self._state_server.setblocking(False)

        self._input_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._input_server.bind(SOCKET_INPUT)
        self._input_server.listen(5)
        self._input_server.setblocking(False)

        self._clients = []
        self._lock = threading.Lock()
        self._key_buffer = deque()
        self._running = True

        # Accept thread
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        # Generate a synthetic test pattern JPEG once
        self._test_jpeg = self._make_test_pattern()

        self._add_event("Mock NeuraCam started")
        self._add_event(f"Simulating {target_fps} fps")

    def _make_test_pattern(self) -> bytes:
        """Generate a 640x360 test frame with mode text, grid, and moving dot."""
        w, h = 640, 360
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Grid lines
        for x in range(0, w, 32):
            cv2.line(frame, (x, 0), (x, h), (20, 20, 20), 1)
        for y in range(0, h, 32):
            cv2.line(frame, (0, y), (w, y), (20, 20, 20), 1)
        # Crosshair at center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (40, 40, 40), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (40, 40, 40), 1)
        # Corner labels
        cv2.putText(frame, "MOCK FEED", (w // 2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes()

    def _add_event(self, msg: str):
        self._events.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

    def _accept_loop(self):
        import select
        while self._running:
            try:
                readable, _, _ = select.select(
                    [self._state_server, self._input_server], [], [], 0.1
                )
                for s in readable:
                    if s == self._state_server:
                        client, _ = s.accept()
                        with self._lock:
                            self._clients.append(client)
                    elif s == self._input_server:
                        client, _ = s.accept()
                        t = threading.Thread(
                            target=self._read_input_loop, args=(client,), daemon=True
                        )
                        t.start()
            except (BlockingIOError, OSError):
                pass

    def _read_input_loop(self, client: socket.socket):
        try:
            while self._running:
                data = client.recv(1024)
                if not data:
                    break
                for c in data.decode("utf-8", errors="replace"):
                    with self._lock:
                        self._key_buffer.append(c)
                    self._add_event(f"Key pressed: {repr(c)}")
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _send(self, state: dict, frame_jpeg: bytes = None):
        state_bytes = json.dumps(state).encode("utf-8")
        state_msg = _pack_msg(MSG_TYPE_STATE, state_bytes)
        dead = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(state_msg)
                    if frame_jpeg:
                        client.sendall(_pack_msg(MSG_TYPE_FRAME, frame_jpeg))
                except (BrokenPipeError, ConnectionError, OSError):
                    dead.append(client)
            for c in dead:
                self._clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def _read_keys(self):
        with self._lock:
            while self._key_buffer:
                yield self._key_buffer.popleft()

    def _simulate(self):
        """Advance simulation by one frame and return state dict."""
        self._frame += 1
        t = self._frame * self.frame_interval

        # ── Mode cycle ──
        self._mode_hold += 1
        hold_duration = int(self.target_fps * 3)  # 3 seconds per mode
        if self._mode_hold >= hold_duration:
            self._mode_hold = 0
            self._mode_idx = (self._mode_idx + 1) % len(self._mode_cycle)
            mode = self._mode_cycle[self._mode_idx]
            self._add_event(f"Mode: → {mode}")
        mode = self._mode_cycle[self._mode_idx]

        # ── Default face values (always assigned) ──
        face_detected = False
        face_cx = face_cy = face_w = face_h = face_conf = 0.0

        # ── Face motion (smooth Lissajous curve) ──
        if mode == "TRACKING" or mode == "TRACKING_HAND":
            face_cx = 640 + 300 * math.sin(t * 0.3)
            face_cy = 360 + 150 * math.sin(t * 0.5 + 1.0)
            face_w = 120 + 30 * math.sin(t * 0.7)
            face_h = 150 + 30 * math.cos(t * 0.6)
            face_conf = 0.75 + 0.2 * abs(math.sin(t * 0.1))
            face_detected = True
        elif mode == "LOCKED":
            # Face frozen at last known position with slight drift
            face_cx = 640 + 200 * math.sin(t * 0.1)
            face_cy = 360 + 100 * math.cos(t * 0.12)
            face_w = 130
            face_h = 160
            face_conf = 0.8
            face_detected = True
        elif mode == "SEARCH":
            # No face, gimbal sweeps
            sweep = math.sin(t * 0.5)
            self._target_pan = 90 + sweep * 80
        elif mode == "HOME":
            face_detected = (self._frame % 10 < 3)
            self._target_pan = 90
            self._target_tilt = 90
            if face_detected:
                face_cx, face_cy = 640, 360
                face_w, face_h = 120, 150
                face_conf = 0.7

        # ── Gimbal angles ──
        if mode == "TRACKING" or mode == "TRACKING_HAND":
            # Gimbal chases face with lag + overshoot
            target_pan = 90 + (face_cx - 640) * 0.15
            target_tilt = 90 + (face_cy - 360) * 0.15
            self._target_pan += (target_pan - self._target_pan) * 0.1
            self._target_tilt += (target_tilt - self._target_tilt) * 0.1
            self._angle_pan += (self._target_pan - self._angle_pan) * 0.15
            self._angle_tilt += (self._target_tilt - self._angle_tilt) * 0.12

        elif mode == "SEARCH":
            sweep = math.sin(t * 0.5)
            self._target_pan = 90 + sweep * 80
            self._angle_pan += (self._target_pan - self._angle_pan) * 0.08
            self._target_tilt = 90
            self._angle_tilt += (self._target_tilt - self._angle_tilt) * 0.1

        elif mode == "HOME":
            self._angle_pan += (90 - self._angle_pan) * 0.2
            self._angle_tilt += (90 - self._angle_tilt) * 0.2
            self._target_pan = 90
            self._target_tilt = 90

        elif mode == "LOCKED":
            # Hold position
            pass

        elif mode == "IDLE":
            self._angle_pan += (90 - self._angle_pan) * 0.05
            self._angle_tilt += (90 - self._angle_tilt) * 0.05
            self._target_pan = 90
            self._target_tilt = 90

        # ── PID errors (noise + tracking offset) ──
        target_x = 640
        target_y = 360
        if face_detected:
            self._pid_pan_error = (face_cx - target_x) / target_x * 0.5
            self._pid_tilt_error = (face_cy - target_y) / target_y * 0.5
        else:
            self._pid_pan_error = 0.1 * math.sin(t * 0.7)
            self._pid_tilt_error = 0.05 * math.cos(t * 0.9)

        # PID terms
        pid_pan_p = self._pid_pan_error * 2.0
        pid_pan_i = 0.05 * math.sin(t * 0.1) * 0.5
        pid_pan_d = -0.2 * math.cos(t * 0.3) * 0.3
        pid_tilt_p = self._pid_tilt_error * 2.0
        pid_tilt_i = 0.05 * math.cos(t * 0.15) * 0.5
        pid_tilt_d = -0.2 * math.sin(t * 0.2) * 0.3
        pid_pan_out = pid_pan_p + pid_pan_i + pid_pan_d
        pid_tilt_out = pid_tilt_p + pid_tilt_i + pid_tilt_d

        # ── Hand / gesture (cyclic) ──
        hand_detected = mode in ("TRACKING_HAND", "TRACKING", "LOCKED") and (
            self._frame % int(self.target_fps * 5) < int(self.target_fps * 2)
        )
        gestures = ["NONE", "NONE", "OPEN_PALM", "NONE", "FIST",
                     "NONE", "THUMBS_UP", "NONE", "PEACE", "NONE"]
        gesture = gestures[(self._frame // int(self.target_fps)) % len(gestures)]
        gesture_method = "svm" if gesture != "NONE" and self._frame % 2 == 0 else "rule"

        # ── Status ──
        kalman_unc = 0.05 + 0.3 * abs(math.sin(t * 0.2))
        serial_ok = True
        recording = self._frame % int(self.target_fps * 30) < int(self.target_fps * 5)
        zoom = 1.0 + 0.3 * abs(math.sin(t * 0.1))

        # ── Latency ──
        latencies = {
            "capture": 4 + 2 * abs(math.sin(t * 0.3)),
            "detect": 14 + 4 * abs(math.sin(t * 0.2 + 0.5)),
            "track": 0.3 + 0.2 * abs(math.sin(t * 0.7)),
            "pid": 0.1 + 0.1 * abs(math.sin(t * 1.1)),
            "display": 2 + 1 * abs(math.sin(t * 0.4)),
            "gesture": 7 + 4 * abs(math.sin(t * 0.25 + 1.0)),
            "ipc": 0.4 + 0.3 * abs(math.sin(t * 0.6)),
        }
        total_lat = sum(latencies.values())

        # ── FPS (slight jitter) ──
        fps = self.target_fps + 2 * math.sin(t * 0.15)

        # ── IMU data ──
        imu_pitch = 1.5 * math.sin(t * 0.1)
        imu_roll = -1.0 * math.cos(t * 0.12)
        imu_yaw = 0.5 * math.sin(t * 0.08)

        # ── Build state dict ──
        state = {
            "fps": round(fps, 1),
            "mode": mode,
            "tracking_target": "HAND" if mode == "TRACKING_HAND" else "FACE",
            "face_detected": face_detected,
            "face_x": round(face_cx, 1),
            "face_y": round(face_cy, 1),
            "face_w": round(face_w, 1),
            "face_h": round(face_h, 1),
            "face_confidence": round(face_conf, 3),
            "frame_w": 1280,
            "frame_h": 720,
            "pan_angle": round(self._angle_pan),
            "tilt_angle": round(self._angle_tilt),
            "pan_target": round(self._target_pan),
            "tilt_target": round(self._target_tilt),
            "imu_pitch": round(imu_pitch, 1),
            "imu_roll": round(imu_roll, 1),
            "imu_yaw": round(imu_yaw, 1),
            "kalman_uncertainty": round(kalman_unc, 3),
            "zoom_level": round(zoom, 1),
            "gesture": gesture,
            "gesture_method": gesture_method,
            "recording": recording,
            "serial_connected": serial_ok,
            "hand_detected": hand_detected,
            "pid_pan_error": round(self._pid_pan_error, 3),
            "pid_tilt_error": round(self._pid_tilt_error, 3),
            "pid_pan_output": round(pid_pan_out, 3),
            "pid_tilt_output": round(pid_tilt_out, 3),
            "pid_pan_p": round(pid_pan_p, 3),
            "pid_pan_i": round(pid_pan_i, 3),
            "pid_pan_d": round(pid_pan_d, 3),
            "pid_tilt_p": round(pid_tilt_p, 3),
            "pid_tilt_i": round(pid_tilt_i, 3),
            "pid_tilt_d": round(pid_tilt_d, 3),
            "latency_ms": latencies,
            "events": list(self._events),
            "timestamp": time.time(),
        }

        # ── Generate live test frame with current state drawn ──
        frame = self._make_live_frame(mode, face_cx, face_cy, face_w, face_h,
                                      face_detected, fps, recording,
                                      self._angle_pan, self._angle_tilt)
        frame_jpeg = frame

        return state, frame_jpeg

    def _make_live_frame(self, mode, face_cx, face_cy, face_w, face_h,
                          face_detected, fps, recording, pan, tilt) -> bytes:
        """Generate a test frame with current sim state drawn on it."""
        w, h = 640, 360
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Grid
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (15, 15, 15), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (15, 15, 15), 1)

        # Crosshair
        cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (40, 40, 40), 1)
        cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (40, 40, 40), 1)

        # Face bounding box
        if face_detected:
            sx, sy = w / 1280, h / 720
            fx = int((face_cx - face_w / 2) * sx)
            fy = int((face_cy - face_h / 2) * sy)
            fw = int(face_w * sx)
            fh = int(face_h * sy)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
            cv2.circle(frame, (int(face_cx * sx), int(face_cy * sy)), 4, (0, 255, 0), -1)

        # Mode text
        mode_colors = {
            "TRACKING": (0, 200, 0), "TRACKING_HAND": (0, 165, 255),
            "LOCKED": (0, 165, 255), "IDLE": (100, 100, 100),
            "SEARCH": (0, 100, 255), "HOME": (0, 255, 255),
        }
        mc = mode_colors.get(mode, (255, 255, 255))
        cv2.putText(frame, f"[{mode}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 2)
        cv2.putText(frame, f"FPS:{fps:.0f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, f"P:{pan:.0f} T:{tilt:.0f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if recording:
            cv2.putText(frame, "REC", (w - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes()

    def run(self, duration: float = 0):
        """Main loop. If duration==0, runs forever."""
        start = time.time()
        last_send = 0.0

        print(f"Mock NeuraCam running at {self.target_fps} fps")
        print(f"  State socket: {SOCKET_STATE}")
        print(f"  Input socket: {SOCKET_INPUT}")
        print(f"  Duration: {'forever' if duration == 0 else f'{duration}s'}")
        print()
        print("  Mode cycle (3s each): IDLE → TRACKING(×3) → TRACKING_HAND")
        print("                      → TRACKING → LOCKED → TRACKING")
        print("                      → SEARCH → IDLE → (repeat)")
        print()
        print("  Connect the TUI:  cargo run --release -p neuracam-tui")
        print("  Connect the GUI:  cargo run --release -p neuracam-gui")
        print("  Press Ctrl+C to stop")

        try:
            while self._running:
                now = time.time()
                if now - last_send < self.frame_interval:
                    time.sleep(0.001)
                    continue
                last_send = now

                # Process any input keys
                for key in self._read_keys():
                    print(f"  [INPUT] Key: {repr(key)}")

                state, jpeg = self._simulate()
                self._send(state, jpeg)

                if duration > 0 and (now - start) > duration:
                    print(f"\nDuration {duration}s reached, stopping.")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self._running = False
            with self._lock:
                for c in self._clients:
                    try:
                        c.close()
                    except OSError:
                        pass
                self._clients.clear()
            for s in [self._state_server, self._input_server]:
                if s:
                    try:
                        s.close()
                    except OSError:
                        pass
            for p in [SOCKET_STATE, SOCKET_INPUT]:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            print("Mock NeuraCam stopped.")


def main():
    parser = argparse.ArgumentParser(description="Mock NeuraCam data feed")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Run duration in seconds (0 = forever)")
    args = parser.parse_args()

    mock = MockNeuraCam(target_fps=args.fps)
    mock.run(duration=args.duration)


if __name__ == "__main__":
    main()
