from src.control.gimbal import GimbalController


def test_gimbal_init():
    gimbal = GimbalController()
    assert gimbal.pan_angle == 90
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal initialization")


def test_gimbal_pan():
    gimbal = GimbalController()
    gimbal.set_pan(45)
    # smooth_move limits to 5° per call from 90 → 85
    assert gimbal.pan_angle == 85
    print("PASS: Gimbal pan angle set")


def test_gimbal_tilt():
    gimbal = GimbalController()
    gimbal.set_tilt(90)
    # Already at 90, no move needed
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal tilt angle set")


def test_gimbal_pan_clamp():
    gimbal = GimbalController()
    gimbal.set_pan(-10)
    # smooth_move limits: 90 → 85 (5° toward 0)
    assert gimbal.pan_angle == 85
    gimbal.set_pan(200)
    # 85 → 90 (5° toward 180)
    assert gimbal.pan_angle == 90
    print("PASS: Gimbal pan clamping")


def test_gimbal_tilt_clamp():
    gimbal = GimbalController()
    gimbal.set_tilt(0)
    # smooth_move: 90 → 85 (5° toward 45)
    assert gimbal.tilt_angle == 85
    gimbal.set_tilt(180)
    # 85 → 90 (5° toward 135)
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal tilt clamping")


def test_gimbal_home():
    gimbal = GimbalController()
    gimbal.set_pan(45)
    gimbal.set_tilt(60)
    # After smooth_move: pan=85, tilt=85
    gimbal.home()
    # home: pan=90, tilt=90 (5° from 85 → 90)
    assert gimbal.pan_angle == 90
    assert gimbal.tilt_angle == 90
    print("PASS: Gimbal home")


def test_gimbal_delta():
    gimbal = GimbalController()
    gimbal.set_pan_delta(10)
    # smooth_move: 90 → 95 (5° up, limited by max_delta=5)
    assert gimbal.pan_angle == 95
    gimbal.set_tilt_delta(-10)
    # smooth_move: 90 → 85 (5° down)
    assert gimbal.tilt_angle == 85
    print("PASS: Gimbal delta movement")


def test_gimbal_status():
    gimbal = GimbalController()
    gimbal.set_pan(45)
    gimbal.set_tilt(60)
    status = gimbal.status()
    assert "PAN:85" in status  # 90 → 85 with smooth_move
    assert "TILT:85" in status  # 90 → 85 with smooth_move
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
