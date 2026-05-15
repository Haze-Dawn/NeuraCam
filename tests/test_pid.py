import numpy as np
from src.control.pid import PIDController


def test_pid_init():
    pid = PIDController(Kp=2.0, Ki=0.05, Kd=0.5,
                        output_limits=(-30, 30))
    assert pid.Kp == 2.0
    assert pid.Ki == 0.05
    assert pid.Kd == 0.5
    print("PASS: PID initialization")


def test_pid_zero_error():
    pid = PIDController()
    output = pid.update(0.0)
    assert output == 0.0
    print("PASS: PID zero error gives zero output")


def test_pid_positive_error():
    pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)
    output = pid.update(1.0)
    assert output == 2.0
    print("PASS: PID proportional term correct")


def test_pid_negative_error():
    pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)
    output = pid.update(-1.0)
    assert output == -2.0
    print("PASS: PID negative error correct")


def test_pid_output_clamping():
    pid = PIDController(Kp=10.0, Ki=0.0, Kd=0.0,
                        output_limits=(-5, 5))
    output = pid.update(1.0)
    assert output == 5.0
    print("PASS: PID output clamping")


def test_pid_reset():
    pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.0)
    pid.update(1.0)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid.filtered_derivative == 0.0
    print("PASS: PID reset clears state")


def test_pid_derivative():
    pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0)
    out1 = pid.update(0.0)
    out2 = pid.update(1.0)
    assert out2 > out1
    print("PASS: PID derivative term increases with error rate")


def test_pid_integral_anti_windup():
    pid = PIDController(Kp=0.0, Ki=10.0, Kd=0.0,
                        integral_limit=5.0)
    for _ in range(100):
        pid.update(1.0)
    assert abs(pid.integral) <= 5.0
    print("PASS: PID anti-windup clamping")


if __name__ == "__main__":
    test_pid_init()
    test_pid_zero_error()
    test_pid_positive_error()
    test_pid_negative_error()
    test_pid_output_clamping()
    test_pid_reset()
    test_pid_derivative()
    test_pid_integral_anti_windup()
    print("\nAll PID tests passed!")
