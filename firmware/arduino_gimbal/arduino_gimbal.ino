/*
 * NeuraCam — Arduino Nano Firmware v3.0 FINAL
 * ============================================
 * Controls:  2x EMAX ES08MAII servos (pan/tilt)
 * Protocol:  P:T (batched, preferred), PAN, TILT, HOME, STATUS
 * Baud:      115200
 * Features:
 *   - Watchdog timer (2s) — auto-recover from lockups
 *   - IMU support (MPU6050) — enable by uncommenting USE_IMU below
 *   - Soft servo limits with jitter guard
 *   - Robust char-by-char serial parser
 *
 * Pinout:
 *   D9  → Pan servo signal (local, base-mounted)
 *   D10 → 100Ω → Ribbon pin 10 → Tilt servo signal (housing)
 *
 * IMU (optional, requires MPU6050 + I2C pull-ups on A4/A5):
 *   A4  → Ribbon pin 8 → MPU6050 SDA
 *   A5  → Ribbon pin 9 → MPU6050 SCL
 *   Uncomment #define USE_IMU below and re-flash
 */

#include <Servo.h>
#include <avr/wdt.h>

// ── Uncomment the next line if MPU6050 is connected ──
// #define USE_IMU

#ifdef USE_IMU
#include <Wire.h>
#endif

Servo panServo;
Servo tiltServo;

const int PAN_PIN  = 9;
const int TILT_PIN = 10;

const int PAN_MIN  = 0;
const int PAN_MAX  = 180;
const int TILT_MIN = 45;
const int TILT_MAX = 135;

int panAngle  = 90;
int tiltAngle = 90;

#ifdef USE_IMU
/* IMU state */
bool  imu_ok      = false;
float cached_pitch = 0.0;
float cached_roll  = 0.0;
unsigned long last_imu_ms = 0;
const  unsigned long IMU_INTERVAL_MS = 100;

static void initIMU() {
  Wire.begin();
  Wire.setClock(100000);

  Wire.beginTransmission(0x68);
  Wire.write(0x75);
  uint8_t err = Wire.endTransmission(true);
  if (err != 0) { imu_ok = false; return; }

  if (Wire.requestFrom((uint8_t)0x68, (uint8_t)1) < 1) { imu_ok = false; return; }
  uint8_t whoami = Wire.read();
  if (whoami != 0x68 && whoami != 0x70) { imu_ok = false; return; }

  Wire.beginTransmission(0x68);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
  imu_ok = true;
}

static void readIMU() {
  if (!imu_ok) return;
  unsigned long now = millis();
  if (now - last_imu_ms < IMU_INTERVAL_MS) return;
  last_imu_ms = now;

  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  if (Wire.endTransmission(true) != 0) return;

  if (Wire.requestFrom((uint8_t)0x68, (uint8_t)6) < 6) return;
  int16_t ax = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t ay = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t az = ((int16_t)Wire.read() << 8) | Wire.read();

  float fax = (float)ax / 16384.0;
  float fay = (float)ay / 16384.0;
  float faz = (float)az / 16384.0;

  cached_pitch = atan2(-fax, sqrt(fay * fay + faz * faz)) * 180.0 / PI;
  cached_roll  = atan2(fay, faz) * 180.0 / PI;
}
#endif

static int parseValue(const String &cmd, char key) {
  int idx = cmd.indexOf(key);
  if (idx < 0) return -1;
  int colon = cmd.indexOf(':', idx);
  if (colon < 0) return -1;
  int space = cmd.indexOf(' ', colon);
  String val = (space > colon)
    ? cmd.substring(colon + 1, space)
    : cmd.substring(colon + 1);
  val.trim();
  return val.toInt();
}

static void setPan(int angle) {
  int clamped = constrain(angle, PAN_MIN, PAN_MAX);
  if (clamped != panAngle) {
    panAngle = clamped;
    panServo.write(panAngle);
  }
}

static void setTilt(int angle) {
  int clamped = constrain(angle, TILT_MIN, TILT_MAX);
  if (clamped != tiltAngle) {
    tiltAngle = clamped;
    tiltServo.write(tiltAngle);
  }
}

static void processCommand(const String &cmd) {
  if (cmd.startsWith("P:")) {
    int p = parseValue(cmd, 'P');
    int t = parseValue(cmd, 'T');
    if (p >= 0) setPan(p);
    if (t >= 0) setTilt(t);
    return;
  }
  if (cmd.startsWith("PAN:")) {
    int angle = cmd.substring(4).toInt();
    setPan(angle);
    return;
  }
  if (cmd.startsWith("TILT:")) {
    int angle = cmd.substring(5).toInt();
    setTilt(angle);
    return;
  }
  if (cmd == "HOME") {
    setPan(90);
    setTilt(90);
    return;
  }
  if (cmd == "STATUS") {
    Serial.print("PAN:");
    Serial.print(panAngle);
    Serial.print(" TILT:");
    Serial.print(tiltAngle);
#ifdef USE_IMU
    Serial.print(" IMU_PITCH:");
    Serial.print(cached_pitch, 1);
    Serial.print(" IMU_ROLL:");
    Serial.print(cached_roll, 1);
#else
    Serial.print(" IMU_PITCH:0.0 IMU_ROLL:0.0");
#endif
    Serial.print(" IMU_YAW:0.0");
    Serial.println();
    return;
  }
}

const int CMD_BUF_SIZE = 64;
char cmd_buf[CMD_BUF_SIZE];
int  cmd_len = 0;

void setup() {
  Serial.begin(115200);

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(90);
  tiltServo.write(90);

#ifdef USE_IMU
  initIMU();
  Serial.print("GIMBAL_READY ");
  Serial.println(imu_ok ? "IMU_OK" : "IMU_FAIL");
#else
  Serial.println("GIMBAL_READY IMU_FAIL");
#endif

  wdt_enable(WDTO_2S);
}

void loop() {
  wdt_reset();

#ifdef USE_IMU
  readIMU();
#endif

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmd_len > 0) {
        cmd_buf[cmd_len] = '\0';
        String cmd = String(cmd_buf);
        cmd.trim();
        if (cmd.length() > 0) {
          processCommand(cmd);
        }
        cmd_len = 0;
      }
    } else if (cmd_len < CMD_BUF_SIZE - 1) {
      cmd_buf[cmd_len++] = c;
    } else {
      cmd_len = 0;
    }
  }
}
