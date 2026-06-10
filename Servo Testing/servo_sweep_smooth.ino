/*
 * NeuraCam — Smooth servo sweep demo
 * Moves both pan and tilt from 0° to 90° over 20 seconds,
 * holds, then returns to 0° over 20 seconds. Repeats forever.
 *
 * Pan:  D9   (0-180° range)
 * Tilt: D10  (45-135° range for gimbal, but uses 0-90 here for demo)
 *
 * Smoothness: 1° steps every 222ms = 4.5°/sec
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN   = 9;
const int TILT_PIN  = 10;

const int STEP_DELAY_MS = 222;    // ~4.5 degrees/second
const int HOLD_MS       = 3000;   // pause at each end

void setup() {
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(0);
  tiltServo.write(0);
  delay(500);
}

void loop() {
  // ── Sweep up: 0 → 90 over 20 seconds ──
  for (int angle = 0; angle <= 90; angle++) {
    panServo.write(angle);
    tiltServo.write(angle);
    delay(STEP_DELAY_MS);
  }

  // ── Hold at 90 ──
  delay(HOLD_MS);

  // ── Sweep down: 90 → 0 over 20 seconds ──
  for (int angle = 90; angle >= 0; angle--) {
    panServo.write(angle);
    tiltServo.write(angle);
    delay(STEP_DELAY_MS);
  }

  // ── Hold at 0 ──
  delay(HOLD_MS);
}
