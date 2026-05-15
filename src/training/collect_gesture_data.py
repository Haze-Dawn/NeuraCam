import cv2
import numpy as np
import os
import csv


GESTURES = ["OPEN_PALM", "FIST", "THUMBS_UP", "POINT", "PEACE"]
SAMPLES_PER_GESTURE = 50
HOG_WIN_SIZE = (64, 64)


def find_hand_roi(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]),
                              np.array([255, 173, 127]))
    mask_hsv = cv2.inRange(hsv, np.array([0, 30, 60]),
                            np.array([20, 150, 200]))
    skin = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    kernel = np.ones((5, 5), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.dilate(skin, kernel, iterations=2)

    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) < 2000:
        return None
    x, y, w, h = cv2.boundingRect(hand)
    margin = int(max(w, h) * 0.15)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    roi = frame[y1:y2, x1:x2]
    return cv2.resize(roi, HOG_WIN_SIZE)


def extract_hog(roi: np.ndarray) -> list:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    hog = cv2.HOGDescriptor(
        _winSize=HOG_WIN_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(gray).flatten().tolist()


def main(output_path: str = "data/gesture/raw/features.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Collecting {SAMPLES_PER_GESTURE} samples per gesture x {len(GESTURES)} gestures")
    print("Hold the pose and press SPACE to start recording (50 frames)")
    print("Press ESC to skip current gesture, q to quit")

    all_features = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for gesture in GESTURES:
        collected = 0
        print(f"\n--- Gesture: {gesture} ---")
        print("  Hold the pose and press SPACE")

        recording = False
        record_count = 0

        while collected < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            hand_roi = find_hand_roi(frame)

            if hand_roi is not None:
                h_disp, w_disp = hand_roi.shape[:2]
                display[10:10 + h_disp, 10:10 + w_disp] = hand_roi
                cv2.rectangle(display, (10, 10),
                              (10 + w_disp, 10 + h_disp), (0, 255, 0), 1)

            status = f"Gesture: {gesture}  [{collected}/{SAMPLES_PER_GESTURE}]"
            if recording:
                status += f" RECORDING {record_count}"
            cv2.putText(display, status,
                        (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=record  ESC=skip  q=quit",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Gesture Data Collection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                recording = True
                record_count = 0
                print("  Recording...")

            elif key == 27:
                print(f"  Skipping {gesture}")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                _save_csv(all_features, output_path)
                return

            if recording:
                if hand_roi is not None:
                    features = extract_hog(hand_roi)
                    all_features.append([gesture] + features)
                    collected += 1
                    record_count += 1
                    if collected % 10 == 0:
                        print(f"  Captured {collected}/{SAMPLES_PER_GESTURE}")
                    if record_count >= SAMPLES_PER_GESTURE:
                        recording = False
                else:
                    print("  Hand lost during recording -- continuing")

    cap.release()
    cv2.destroyAllWindows()
    _save_csv(all_features, output_path)
    print(f"\nCollection complete. {len(all_features)} samples saved to {output_path}")


def _save_csv(rows: list, path: str):
    if not rows:
        print("No data collected")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["gesture"] + [f"f{i}" for i in range(1764)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gesture/raw/features.csv")
    args = parser.parse_args()
    main(args.output)
