#include <Servo.h>
#include <Wire.h>
#include <MPU6050.h>

Servo panServo;
Servo tiltServo;
MPU6050 imu;

const int PAN_PIN = 9;
const int TILT_PIN = 10;

const int PAN_MIN = 0;
const int PAN_MAX = 180;  
const int TILT_MIN = 45;
const int TILT_MAX = 135;

const int STEP_DELAY = 15;
const int PAUSE_DELAY = 1000;

int16_t ax, ay, az, gx, gy, gz;
float pitch, roll;
bool imuOK = false;

void readIMU() {
  imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  pitch = atan2(-ay, sqrt(ax * ax + az * az)) * 180.0 / PI;
  roll  = atan2(ax, az) * 180.0 / PI;
}

void printStatus(int panAngle, int tiltAngle) {
  readIMU();
  Serial.print("PAN:"); Serial.print(panAngle);
  Serial.print(" TILT:"); Serial.print(tiltAngle);
  Serial.print(" PITCH:"); Serial.print(pitch, 1);
  Serial.print(" ROLL:"); Serial.print(roll, 1);
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println(F("========================================"));
  Serial.println(F("  NeuraCam - Servo & IMU Sweep Test"));
  Serial.println(F("========================================"));
  Serial.println();

  Wire.begin();
  Wire.setClock(100000);
  imu.initialize();

  if (imu.testConnection()) {
    imuOK = true;
    Serial.println(F("MPU6050 detected OK"));
    Serial.println(F("Calibrating IMU (keep gimbal still)..."));
    imu.CalibrateAccel(6);
    imu.CalibrateGyro(6);
    Serial.println(F("IMU calibration done"));
  } else {
    imuOK = false;
    Serial.println(F("WARNING: MPU6050 not detected! Check I2C wiring (A4=SDA, A5=SCL)"));
  }
  Serial.println();

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  panServo.write(90);
  tiltServo.write(90);
  Serial.println(F("Servos initialized at center (90 deg)"));
  delay(1000);

  Serial.println(F("Pan servo on D9  |  Tilt servo on D10"));
  Serial.println(F("Pan range: 0-180 | Tilt range: 45-135"));
  Serial.println();
  Serial.println(F("Starting sweep cycle..."));
  Serial.println();
}

void loop() {
  Serial.println(F("--- Pan sweep ---"));
  for (int a = PAN_MIN; a <= PAN_MAX; a++) {
    panServo.write(a);
    if (a % 10 == 0) printStatus(a, 90);
    delay(STEP_DELAY);
  }
  for (int a = PAN_MAX; a >= PAN_MIN; a--) {
    panServo.write(a);
    if (a % 10 == 0) printStatus(a, 90);
    delay(STEP_DELAY);
  }
  panServo.write(90);
  delay(PAUSE_DELAY);

  Serial.println(F("--- Tilt sweep ---"));
  for (int b = TILT_MIN; b <= TILT_MAX; b++) {
    tiltServo.write(b);
    if (b % 10 == 0) printStatus(90, b);
    delay(STEP_DELAY);
  }
  for (int b = TILT_MAX; b >= TILT_MIN; b--) {
    tiltServo.write(b);
    if (b % 10 == 0) printStatus(90, b);
    delay(STEP_DELAY);
  }
  tiltServo.write(90);
  delay(PAUSE_DELAY);

  Serial.println(F("--- Both together ---"));
  for (int a = PAN_MIN, b = TILT_MIN; a <= PAN_MAX && b <= TILT_MAX; a++, b++) {
    panServo.write(a);
    tiltServo.write(b);
    if (a % 10 == 0) printStatus(a, b);
    delay(STEP_DELAY);
  }
  for (int a = PAN_MAX, b = TILT_MAX; a >= PAN_MIN && b >= TILT_MIN; a--, b--) {
    panServo.write(a);
    tiltServo.write(b);
    if (a % 10 == 0) printStatus(a, b);
    delay(STEP_DELAY);
  }
  delay(PAUSE_DELAY);

  Serial.println(F("--- Center + stability check ---"));
  panServo.write(90);
  tiltServo.write(90);
  for (int i = 0; i < 5; i++) {
    printStatus(90, 90);
    delay(200);
  }

  if (imuOK) {
    Serial.println();
    Serial.println(F("IMU status: OK - pitch/roll tracking with servo movement"));
    Serial.println(F("Expect pitch/roll to change as tilt servo moves."));
  } else {
    Serial.println();
    Serial.println(F("FIX: Check MPU6050 wiring:"));
    Serial.println(F("  VCC -> 5V    GND -> GND"));
    Serial.println(F("  SDA -> A4    SCL -> A5"));
    Serial.println(F("  Install MPU6050 library in Arduino IDE"));
  }

  Serial.println();
  Serial.println(F("--- Cycle complete, restarting ---"));
  Serial.println();
  delay(2000);
}
