import cv2
import numpy as np
import os
import time


GESTURES = ["OPEN_PALM", "FIST", "THUMBS_UP", "POINT", "PEACE"]
SAMPLES_PER_GESTURE = 50
CROP_SIZE = (64, 64)


def find_hand_roi(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 200])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
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
    return cv2.resize(roi, CROP_SIZE)


def main(output_dir: str = "data/gesture/hand_crops"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for gesture in GESTURES:
        os.makedirs(os.path.join(output_dir, gesture), exist_ok=True)

    print(f"Collecting {SAMPLES_PER_GESTURE} samples per gesture")
    print("Hold the gesture in frame and press SPACE to capture")
    print("Press ESC to skip current gesture")
    print("Press q to quit")

    for g_idx, gesture in enumerate(GESTURES):
        collected = 0
        gesture_dir = os.path.join(output_dir, gesture)
        existing = len(os.listdir(gesture_dir))
        print(f"\n--- Gesture: {gesture} (have {existing}) ---")

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

            cv2.putText(display,
                        f"Gesture: {gesture}  [{collected}/{SAMPLES_PER_GESTURE}]",
                        (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=capture  ESC=skip  q=quit",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Gesture Data Collection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if hand_roi is not None:
                    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    path = os.path.join(gesture_dir, f"{gesture}_{collected:03d}.jpg")
                    cv2.imwrite(path, gray)
                    collected += 1
                    print(f"  Captured {collected}/{SAMPLES_PER_GESTURE}")
                else:
                    print("  No hand detected -- adjust position or lighting")
            elif key == 27:
                print(f"  Skipping {gesture}")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print(f"Done. Saved to {output_dir}")
                return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCollection complete. Saved {SAMPLES_PER_GESTURE * 5} samples to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gesture/hand_crops")
    args = parser.parse_args()
    main(args.output)
