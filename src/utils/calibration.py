import cv2
import numpy as np
import glob
import os
import json


def calibrate_intrinsics(checkerboard=(9, 6), square_size_mm=25.0,
                         image_dir="calibration_images"):
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_mm / 1000.0

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not images:
        print(f"No images found in {image_dir}/")
        print("Capture 20-30 checkerboard images first.")
        return None

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            imgpoints.append(corners2)

    if not imgpoints:
        print("No checkerboards detected in any image.")
        return None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    result = {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.tolist(),
        "reprojection_error": float(ret),
        "image_width": gray.shape[1],
        "image_height": gray.shape[0],
    }

    os.makedirs("config", exist_ok=True)
    with open("config/calibration.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"Camera matrix:\n{K}")
    print("Saved to config/calibration.json")
    return result


def calibrate_pixel_to_angle():
    print("Pixel-to-angle calibration:")
    print("1. Place a marker at 1.0m from the gimbal")
    print("2. Center the marker in frame (gimbal at home: PAN:90 TILT:90)")
    print("3. Record marker pixel position")
    print("4. Pan +10 degrees, record new position")
    print("5. Compute px/deg")
    print("\nRun manually with gimbal + marker.")
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--pixel-to-angle":
        calibrate_pixel_to_angle()
    else:
        calibrate_intrinsics()
