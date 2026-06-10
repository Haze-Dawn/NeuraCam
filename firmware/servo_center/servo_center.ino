/*
 * NeuraCam - Servo Center / Horn Alignment
 *
 * Centers both servos to 90 degrees so you can attach the servo
 * horns at the mechanical center position. Run this before mounting.
 *
 * Instructions:
 *   1. Upload this sketch to the Arduino
 *   2. Open Serial Monitor (115200 baud)
 *   3. Wait for "SERVOS CENTERED" message
 *   4. Attach both servo horns at the straight/center position
 *   5. Send any character over serial to start centering verification
 *   6. Verify: pan to 0, back to 90, to 180 — horn should point
 *      symmetrically left/center/right
 *   7. Re-upload the main gimbal firmware
 *
 * Pins:    Pan = D9, Tilt = D10
 * Ranges:  Pan 0-180, Tilt 45-135
 * Library: Servo.h (built-in)
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN = 9;
const int TILT_PIN = 10;

const int PAN_MIN = 0;
const int PAN_MAX = 180;
const int TILT_MIN = 45;
const int TILT_MAX = 135;

void prompt() {
  Serial.println();
  Serial.println(F("Send any character when horns are attached to continue."));
  while (Serial.available()) Serial.read();
  while (!Serial.available()) { delay(10); }
  while (Serial.available()) Serial.read();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println(F("========================================"));
  Serial.println(F("  NeuraCam - Servo Center / Horn Alignment"));
  Serial.println(F("========================================"));
  Serial.println();

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  panServo.write(90);
  tiltServo.write(90);

  Serial.println(F("Servos commanded to 90 degrees (center)."));
  Serial.println();
  Serial.println(F("=== ATTACH SERVO HORNS NOW ==="));
  Serial.println(F("Both servos are at their electrical center (1500us pulse)."));
  Serial.println(F("Attach each servo horn so it points straight ahead / level."));
  Serial.println(F("Pan horn: straight forward  |  Tilt horn: horizontal"));
  Serial.println();
  delay(2000);
  Serial.println(F("SERVOS CENTERED - waiting for confirmation..."));
  Serial.println(F("Send any character over Serial Monitor when horns are attached."));

  prompt();
}

void loop() {
  Serial.println(F("--- Verification Sweep ---"));
  Serial.println(F("Pan: 90 -> 0 (horn should rotate one way)"));
  for (int a = 90; a >= PAN_MIN; a -= 5) {
    panServo.write(a);
    Serial.print(F("PAN:"));
    Serial.println(a);
    delay(300);
  }

  Serial.println(F("Pan: 0 -> 180 (horn should sweep across center to other side)"));
  for (int a = PAN_MIN; a <= PAN_MAX; a += 5) {
    panServo.write(a);
    Serial.print(F("PAN:"));
    Serial.println(a);
    delay(300);
  }

  Serial.println(F("Pan: 180 -> 90 (return to center)"));
  for (int a = PAN_MAX; a >= 90; a -= 5) {
    panServo.write(a);
    Serial.print(F("PAN:"));
    Serial.println(a);
    delay(300);
  }

  delay(500);

  Serial.println(F("Tilt: 90 -> 45 (horn should tilt one way)"));
  for (int b = 90; b >= TILT_MIN; b -= 5) {
    tiltServo.write(b);
    Serial.print(F("TILT:"));
    Serial.println(b);
    delay(300);
  }

  Serial.println(F("Tilt: 45 -> 135 (horn should sweep past level to other side)"));
  for (int b = TILT_MIN; b <= TILT_MAX; b += 5) {
    tiltServo.write(b);
    Serial.print(F("TILT:"));
    Serial.println(b);
    delay(300);
  }

  Serial.println(F("Tilt: 135 -> 90 (return to center)"));
  for (int b = TILT_MAX; b >= 90; b -= 5) {
    tiltServo.write(b);
    Serial.print(F("TILT:"));
    Serial.println(b);
    delay(300);
  }

  panServo.write(90);
  tiltServo.write(90);

  Serial.println();
  Serial.println(F("=== DONE ==="));
  Serial.println(F("If horns sweep symmetrically around center, alignment is good."));
  Serial.println(F("Re-upload the main gimbal firmware now."));
  Serial.println();

  delay(5000);
}
