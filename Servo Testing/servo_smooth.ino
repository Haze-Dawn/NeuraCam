// Butter-smooth servo sweep using direct Timer1 hardware PWM on pin 9 (OC1A)
// Bypasses the Servo library entirely for cleaner timing and higher frequency
// Standalone test -- not part of gimbal firmware. The production firmware
// (firmware/arduino_gimbal/arduino_gimbal.ino) uses the Servo.h library for
// compatibility with Servo.write(angle) command format from Python.

const int SERVO_PIN = 9;

const int PULSE_MIN = 500;
const int PULSE_MAX = 2500;
const int PULSE_MID = 1500;

const int FREQ_HZ = 50;
const int PERIOD_US = 1000000 / FREQ_HZ;

const int SWEEP_MS = 6000;

const int UPDATE_US = PERIOD_US;

void setupTimer1() {
    pinMode(SERVO_PIN, OUTPUT);

    TCCR1A = 0;
    TCCR1B = 0;
    TCNT1 = 0;

    TCCR1A |= (1 << WGM11);
    TCCR1A |= (1 << COM1A1);
    TCCR1B |= (1 << WGM13) | (1 << WGM12);
    TCCR1B |= (1 << CS11);

    long ticksPerPeriod = (long)PERIOD_US * 2;
    ICR1 = (unsigned int)constrain(ticksPerPeriod, 1000, 65535);

    OCR1A = PULSE_MID * 2;
}

void setPulse(int us) {
    us = constrain(us, PULSE_MIN, PULSE_MAX);
    OCR1A = (unsigned int)((long)us * 2);
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

        setPulse(pulse);

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

void setup() {
    Serial.begin(115200);
    setupTimer1();
    setPulse(PULSE_MID);
    delay(1000);
    Serial.println("Smooth sweep started");
}

void loop() {
    smoothMove(PULSE_MID, PULSE_MIN, SWEEP_MS / 2);
    smoothMove(PULSE_MIN, PULSE_MAX, SWEEP_MS);
    smoothMove(PULSE_MAX, PULSE_MID, SWEEP_MS / 2);
    delay(2000);
}
