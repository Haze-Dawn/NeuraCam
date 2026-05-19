import serial
import time
import sys
import math

PORT = "/dev/ttyUSB0"
BAUD = 115200
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 45, 135
HZ = 50
DT = 1.0 / HZ
BATCHED = True  # Use combined P:T command format

def send(ser, cmd):
    ser.write((cmd + "\n").encode())

def set_angle(ser, pan, tilt):
    pan = max(PAN_MIN, min(PAN_MAX, int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    if BATCHED:
        send(ser, f"P:{pan} T:{tilt}")
    else:
        send(ser, f"PAN:{pan}")
        send(ser, f"TILT:{tilt}")

def s_curve(t):
    return 1.0 / (1.0 + math.exp(-10.0 * (t - 0.5)))

def interpolate(current, target, steps):
    diff = target - current
    for i in range(1, steps + 1):
        t = i / steps
        pos = s_curve(t)
        yield current + diff * pos

def smooth_move(ser, end_pan, end_tilt, duration, current_pan, current_tilt):
    pan_steps = max(int(duration / DT), 1)
    tilt_steps = max(int(duration / DT), 1)
    pan_gen = interpolate(current_pan, end_pan, pan_steps)
    tilt_gen = interpolate(current_tilt, end_tilt, tilt_steps)
    pan_val = current_pan
    tilt_val = current_tilt
    for p, t in zip(pan_gen, tilt_gen):
        p_int, t_int = int(p), int(t)
        if p_int != int(pan_val) or t_int != int(tilt_val):
            set_angle(ser, p_int, t_int)
            pan_val = p_int
            tilt_val = t_int
        time.sleep(DT)
    return float(pan_val), float(tilt_val)

print("Connecting to Arduino...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Failed to open {PORT}: {e}")
    print("Check the port and try: ls /dev/ttyU*")
    sys.exit(1)

time.sleep(2)
print("Connected.")

while ser.in_waiting:
    ser.readline()

set_angle(ser, 90, 90)
print("Homing...")
time.sleep(1)
pan, tilt = 90, 90

print("Routine starting.")
time.sleep(1)

sweep_angle = 60
tilt_angle = 45
sweep_duration = 3.0

print("  Pan sweep")
for _ in range(2):
    pan, tilt = smooth_move(ser, 90 + sweep_angle, 90, sweep_duration, pan, tilt)
    pan, tilt = smooth_move(ser, 90 - sweep_angle, 90, sweep_duration, pan, tilt)
pan, tilt = smooth_move(ser, 90, 90, 1.0, pan, tilt)

print("  Tilt sweep")
for _ in range(2):
    pan, tilt = smooth_move(ser, 90, 90 + tilt_angle, sweep_duration, pan, tilt)
    pan, tilt = smooth_move(ser, 90, 90 - tilt_angle, sweep_duration, pan, tilt)
pan, tilt = smooth_move(ser, 90, 90, 1.0, pan, tilt)

print("  Combined figure-8")
waypoints = [
    (90, 90, 1.0),
    (120, 110, 1.5),
    (90, 90, 1.0),
    (60, 70, 1.5),
    (90, 90, 1.0),
    (60, 110, 1.5),
    (90, 90, 1.0),
    (120, 70, 1.5),
    (90, 90, 1.0),
]
for wp_pan, wp_tilt, dur in waypoints:
    pan, tilt = smooth_move(ser, wp_pan, wp_tilt, dur, pan, tilt)

print("  Home")
pan, tilt = smooth_move(ser, 90, 90, 1.0, pan, tilt)
time.sleep(1)

ser.close()
print("Done.")
