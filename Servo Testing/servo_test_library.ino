// Servo.h sweep test -- compares against Timer1 version
// Same timing (6s sweep) but uses the standard library
// Accepts the combined P:{pan} T:{tilt}\n command format from Python.
// Standalone test -- not part of gimbal firmware.

#include <Servo.h>

Servo myServo;

const int SERVO_PIN = 9;

const int PULSE_MIN = 500;
const int PULSE_MAX = 2500;
const int PULSE_MID = 1500;

const int SWEEP_MS = 6000;

const int UPDATE_US = 20000;

void setup() {
    Serial.begin(115200);
    myServo.attach(SERVO_PIN, PULSE_MIN, PULSE_MAX);
    myServo.writeMicroseconds(PULSE_MID);
    delay(1000);
    Serial.println("Servo.h sweep started");
}

float easeInOutCos(float t) {
    return (1.0 - cos(t * PI)) / 2.0;
}

void smoothMove(int fromPulse, int toPulse, int durationMs) {
    long steps = (durationMs * 1000L) / UPDATE_US;
    if (steps < 2) steps = 2;

    unsigned long nextTick = micros();

    for (long i = 0; i <= steps; i++) {
        float t = (float)i / (float)steps;
        float eased = easeInOutCos(t);
        int pulse = fromPulse + (int)((toPulse - fromPulse) * eased);

        myServo.writeMicroseconds(pulse);

        nextTick += UPDATE_US;
        long wait = nextTick - micros();
        while (wait > 0) {
            if (wait >= 1000) {
                delay(1);
                wait = nextTick - micros();
            } else {
                delayMicroseconds(wait);
                wait = 0;
            }
        }
    }
}

void loop() {
    smoothMove(PULSE_MID, PULSE_MIN, SWEEP_MS / 2);
    smoothMove(PULSE_MIN, PULSE_MAX, SWEEP_MS);
    smoothMove(PULSE_MAX, PULSE_MID, SWEEP_MS / 2);
    delay(2000);
}
