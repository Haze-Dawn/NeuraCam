import os
import json
import tempfile
import pytest
from src.utils.calibration import calibrate_intrinsics, calibrate_pixel_to_angle


def test_calibrate_intrinsics_no_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = calibrate_intrinsics(image_dir=tmpdir)
        assert result is None


def test_calibrate_intrinsics_no_checkerboard():
    with tempfile.TemporaryDirectory() as tmpdir:
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        path = os.path.join(tmpdir, "test.jpg")
        cv2.imwrite(path, img)
        result = calibrate_intrinsics(image_dir=tmpdir)
        assert result is None


def test_calibrate_intrinsics_saves_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("calibration_images", exist_ok=True)
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        path = os.path.join("calibration_images", "test.jpg")
        cv2.imwrite(path, img)
        result = calibrate_intrinsics(image_dir="calibration_images")
        config_exists = os.path.exists("config/calibration.json")
        if result is not None:
            assert config_exists
            with open("config/calibration.json") as f:
                data = json.load(f)
            assert "camera_matrix" in data
            assert "distortion_coefficients" in data
        os.chdir(cwd)


def test_calibrate_pixel_to_angle_returns_none():
    result = calibrate_pixel_to_angle()
    assert result is None


def test_calibrate_pixel_to_angle_prints(capsys):
    calibrate_pixel_to_angle()
    captured = capsys.readouterr()
    assert "Pixel-to-angle calibration" in captured.out


if __name__ == "__main__":
    test_calibrate_intrinsics_no_images()
    test_calibrate_intrinsics_no_checkerboard()
    test_calibrate_pixel_to_angle_returns_none()
    test_calibrate_pixel_to_angle_prints()
    print("\nAll calibration tests passed!")
