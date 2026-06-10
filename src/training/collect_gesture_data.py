import cv2
import numpy as np
import os
import csv
import time


GESTURES = ["OPEN_PALM", "FIST", "THUMBS_UP", "POINT", "PEACE"]
HANDS = ["RIGHT", "LEFT"]
SAMPLES_PER_GESTURE = 250
HOG_WIN_SIZE = (64, 64)
COUNTDOWN_SECONDS = 3
INTERMISSION_SECONDS = 4


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


def _show_countdown(display, text, seconds):
    start = time.time()
    while time.time() - start < seconds:
        remaining = int(seconds - (time.time() - start)) + 1
        overlay = display.copy()
        cv2.putText(overlay, text, (50, display.shape[0] // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(overlay, str(remaining),
                    (display.shape[1] // 2 - 30, display.shape[0] // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 3)
        cv2.imshow("Gesture Data Collection", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def main(output_path: str = "data/gesture/raw/features.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_expected = SAMPLES_PER_GESTURE * len(GESTURES) * len(HANDS)
    print(f"Auto-collecting {SAMPLES_PER_GESTURE} samples per gesture "
          f"x {len(GESTURES)} gestures x {len(HANDS)} hands = {total_expected} total")
    print("Controls while running:")
    print("  q = quit and save")

    all_features = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for hand in HANDS:
        for gesture in GESTURES:
            print(f"\n--- {hand} Hand: {gesture} ---")

            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()

            if not _show_countdown(display, f"{hand} hand - Get ready for {gesture}",
                                   COUNTDOWN_SECONDS):
                break

            collected = 0
            hand_lost_warnings = 0
            max_hand_lost_warnings = 10

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

                    features = extract_hog(hand_roi)
                    all_features.append([gesture] + features)
                    collected += 1
                    hand_lost_warnings = 0

                    if collected % 10 == 0:
                        print(f"  Captured {collected}/{SAMPLES_PER_GESTURE}")
                else:
                    hand_lost_warnings += 1
                    if hand_lost_warnings > max_hand_lost_warnings:
                        print("  Hand not detected for too long -- "
                              "moving to next gesture")
                        break

                bar_width = int(300 * collected / SAMPLES_PER_GESTURE)
                cv2.rectangle(display, (50, display.shape[0] - 60),
                              (50 + bar_width, display.shape[0] - 50),
                              (0, 255, 0), -1)
                cv2.putText(display,
                            f"{gesture} [{collected}/{SAMPLES_PER_GESTURE}]",
                            (10, display.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, "Hold pose -- auto-capturing",
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Gesture Data Collection", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    _save_csv(all_features, output_path)
                    return

            if collected > 0:
                print(f"  Finished {hand} {gesture}: {collected} samples")

            g_idx = GESTURES.index(gesture)
            next_text = f"Next: {HANDS[HANDS.index(hand) + 1]} hand - {GESTURES[0]}" if g_idx == len(GESTURES) - 1 and HANDS.index(hand) < len(HANDS) - 1 else (f"Next: {GESTURES[g_idx + 1]}" if g_idx < len(GESTURES) - 1 else "All done!")
            overlay = np.zeros_like(display)
            cv2.putText(overlay, f"{hand} {gesture} done!",
                        (display.shape[1] // 2 - 140, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(overlay, next_text,
                        (display.shape[1] // 2 - 100, display.shape[0] // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            for _ in range(int(INTERMISSION_SECONDS * 10)):
                cv2.imshow("Gesture Data Collection", overlay)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            else:
                continue
            break

    cap.release()
    cv2.destroyAllWindows()
    _save_csv(all_features, output_path)
    print(f"\nCollection complete. {len(all_features)} samples "
          f"saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gesture/raw/features.csv")
    args = parser.parse_args()
    main(args.output)
