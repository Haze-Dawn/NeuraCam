/*
 * AI Gimbal Camera - Arduino Nano Firmware (v2.1)
 * Controls: 2x EMAX ES08MAII servos (pan/tilt), MPU6050 IMU (I2C)
 * Protocol: P:T (batched), PAN, TILT, HOME, STATUS
 * Baud: 115200
 * Libraries: Servo.h (built-in), Wire.h (built-in), MPU6050 (install via Library Manager)
 *
 * v2.1 changes:
 * - Watchdog timer (WDT) enabled, reset every loop() iteration
 * - IMU data cached: read every ~100ms in background, STATUS returns cached values
 *   (prevents I2C read latency from blocking servo command processing)
 */

#include <Servo.h>
#include <Wire.h>
#include <MPU6050.h>
#include <avr/wdt.h>

Servo panServo;
Servo tiltServo;
MPU6050 imu;

const int PAN_PIN = 9;
const int TILT_PIN = 10;

const int PAN_MIN = 0;
const int PAN_MAX = 180;
const int TILT_MIN = 45;
const int TILT_MAX = 135;

int panAngle = 90;
int tiltAngle = 90;

// Cached IMU data
float cached_pitch = 0.0;
float cached_roll = 0.0;
unsigned long last_imu_read = 0;
const unsigned long IMU_READ_INTERVAL = 100;  // read IMU every 100ms (10Hz)

void read_imu_cache() {
  unsigned long now = millis();
  if (now - last_imu_read < IMU_READ_INTERVAL) return;
  last_imu_read = now;

  int16_t ax, ay, az, gx, gy, gz;
  imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  cached_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  cached_roll = atan2(ay, az) * 180.0 / PI;
}

int parseValue(String cmd, char key) {
  int idx = cmd.indexOf(key);
  if (idx < 0) return -1;
  int colon = cmd.indexOf(':', idx);
  if (colon < 0) return -1;
  int space = cmd.indexOf(' ', colon);
  String val = (space > colon) ? cmd.substring(colon + 1, space) : cmd.substring(colon + 1);
  val.trim();
  return val.toInt();
}

void setup() {
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);

  Wire.begin();
  Wire.setClock(100000);
  imu.initialize();
  Serial.begin(115200);

  wdt_enable(WDTO_2S);

  if (imu.testConnection()) {
    Serial.println("GIMBAL_READY IMU_OK");
  } else {
    Serial.println("GIMBAL_READY IMU_FAIL");
  }
}

void loop() {
  wdt_reset();

  read_imu_cache();

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("P:")) {
      int p = parseValue(cmd, 'P');
      int t = parseValue(cmd, 'T');
      if (p >= 0) {
        panAngle = constrain(p, PAN_MIN, PAN_MAX);
        panServo.write(panAngle);
      }
      if (t >= 0) {
        tiltAngle = constrain(t, TILT_MIN, TILT_MAX);
        tiltServo.write(tiltAngle);
      }
    }
    else if (cmd.startsWith("PAN:")) {
      int angle = cmd.substring(4).toInt();
      panAngle = constrain(angle, PAN_MIN, PAN_MAX);
      panServo.write(panAngle);
    }
    else if (cmd.startsWith("TILT:")) {
      int angle = cmd.substring(5).toInt();
      tiltAngle = constrain(angle, TILT_MIN, TILT_MAX);
      tiltServo.write(tiltAngle);
    }
    else if (cmd == "HOME") {
      panAngle = 90;
      tiltAngle = 90;
      panServo.write(90);
      tiltServo.write(90);
    }
    else if (cmd == "STATUS") {
      Serial.print("PAN:");
      Serial.print(panAngle);
      Serial.print(" TILT:");
      Serial.print(tiltAngle);
      Serial.print(" IMU_PITCH:");
      Serial.print(cached_pitch, 1);
      Serial.print(" IMU_ROLL:");
      Serial.print(cached_roll, 1);
      Serial.print(" IMU_YAW:0.0");
      Serial.println();
    }
  }
}
