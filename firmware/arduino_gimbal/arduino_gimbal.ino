# AI Gimbal Camera - Firmware
# Arduino Nano (ATmega328P) - Servo Controller + MPU6050 IMU
# Serial protocol: PAN:{angle}\n, TILT:{angle}\n, HOME\n, STATUS\n
# Baud rate: 115200
# IMU: MPU6050 on I2C (A4=SDA, A5=SCL)

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

int panAngle = 90;
int tiltAngle = 90;

void setup() {
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);

  Wire.begin();
  imu.initialize();
  if (imu.testConnection()) {
    Serial.begin(115200);
    Serial.println("GIMBAL_READY IMU_OK");
  } else {
    Serial.begin(115200);
    Serial.println("GIMBAL_READY IMU_FAIL");
  }
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("PAN:")) {
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
      int16_t ax, ay, az, gx, gy, gz;
      imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
      float accel_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
      float accel_roll = atan2(ay, az) * 180.0 / PI;
      Serial.print("PAN:");
      Serial.print(panAngle);
      Serial.print(" TILT:");
      Serial.print(tiltAngle);
      Serial.print(" IMU_PITCH:");
      Serial.print(accel_pitch, 1);
      Serial.print(" IMU_ROLL:");
      Serial.print(accel_roll, 1);
      Serial.print(" IMU_YAW:0.0");
      Serial.println();
    }
  }
}
