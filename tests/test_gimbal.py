from src.control.gimbal import GimbalController


def test_gimbal_init():
    gimbal = GimbalController()
    assert gimbal.pan_angle == 90
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal initialization")


def test_gimbal_pan():
    gimbal = GimbalController()
    gimbal.set_pan(45)
    assert gimbal.pan_angle == 45
    print("PASS: Gimbal pan angle set")


def test_gimbal_tilt():
    gimbal = GimbalController()
    gimbal.set_tilt(90)
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal tilt angle set")


def test_gimbal_pan_clamp():
    gimbal = GimbalController()
    gimbal.set_pan(-10)
    assert gimbal.pan_angle == 0
    gimbal.set_pan(200)
    assert gimbal.pan_angle == 180
    print("PASS: Gimbal pan clamping")


def test_gimbal_tilt_clamp():
    gimbal = GimbalController()
    gimbal.set_tilt(0)
    assert gimbal.tilt_angle == 45
    gimbal.set_tilt(180)
    assert gimbal.tilt_angle == 135
    print("PASS: Gimbal tilt clamping")


def test_gimbal_home():
    gimbal = GimbalController()
    gimbal.set_pan(45)
    gimbal.set_tilt(60)
    gimbal.home()
    assert gimbal.pan_angle == 90
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal home")


def test_gimbal_delta():
    gimbal = GimbalController()
    gimbal.set_pan_delta(10)
    assert gimbal.pan_angle == 100
    gimbal.set_tilt_delta(-10)
    assert gimbal.tilt_angle == 80
    print("PASS: Gimbal delta movement")


def test_gimbal_status():
    gimbal = GimbalController()
    gimbal.set_pan(45)
    gimbal.set_tilt(60)
    status = gimbal.status()
    assert "PAN:45" in status
    assert "TILT:60" in status
    print("PASS: Gimbal status string")


if __name__ == "__main__":
    test_gimbal_init()
    test_gimbal_pan()
    test_gimbal_tilt()
    test_gimbal_pan_clamp()
    test_gimbal_tilt_clamp()
    test_gimbal_home()
    test_gimbal_delta()
    test_gimbal_status()
    print("\nAll gimbal tests passed!")
