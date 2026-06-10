"""Collect face images for fine-tuning the FaceCNN on your own camera/subject.
Auto-captures frames on a timer when in capture mode.
Press SPACE to toggle auto-capture on/off, move your head naturally
between captures for variety.

Run: python src/training/collect_face_data.py
"""
import os, cv2, time
from datetime import datetime

OUT_DIR = "data/face/custom"
os.makedirs(os.path.join(OUT_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "crops"), exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

capture_interval_frames = 8
frame_counter = 0
count = 0
auto_capture = False

print("Face Data Collector (Auto-Capture Mode)")
print("========================================")
print("Instructions:")
print("  1. Press SPACE to START auto-capture")
print("  2. Move your head naturally -- vary distance, angle, lighting")
print("  3. Press SPACE to STOP auto-capture")
print("  4. Press 'd' for a background reference frame (no face)")
print("  5. Press 'q' to quit")
print()
print(f"Auto-capture interval: every {capture_interval_frames} frames")
print(f"Auto-capture jitter: random crop offset for variety")
print(f"Output: {OUT_DIR}/")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    display = frame.copy()
    frame_counter += 1

    if auto_capture and frame_counter % capture_interval_frames == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = os.path.join(OUT_DIR, "raw", f"face_{ts}.jpg")
        crop_path = os.path.join(OUT_DIR, "crops", f"face_{ts}.jpg")
        cv2.imwrite(raw_path, frame)

        crop_sz = min(w, h) // 3
        cx = w // 2 + np.random.randint(-crop_sz // 4, crop_sz // 4)
        cy = h // 2 + np.random.randint(-crop_sz // 4, crop_sz // 4)
        x1 = max(0, cx - crop_sz // 2)
        y1 = max(0, cy - crop_sz // 2)
        x2 = min(w, x1 + crop_sz)
        y2 = min(h, y1 + crop_sz)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 128))
        cv2.imwrite(crop_path, crop)
        count += 1
        print(f"  [{count}] Captured (jitter: ({cx-w//2}, {cy-h//2})px)")

    status_color = (0, 255, 0) if auto_capture else (100, 100, 100)
    mode_text = "AUTO-CAPTURING" if auto_capture else "PAUSED"
    cv2.circle(display, (30, 40), 10, status_color, -1)
    cv2.putText(display, mode_text, (50, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(display, f"Captures: {count}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(display,
                "[SPACE] toggle capture  [d] background  [q] quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    if auto_capture:
        progress = ((frame_counter % (capture_interval_frames * 10))
                    / (capture_interval_frames * 10))
        rx = int(w * progress)
        cv2.rectangle(display, (0, h - 5), (rx, h), (0, 255, 0), -1)

    cv2.imshow("Face Data Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        auto_capture = not auto_capture
        print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
        frame_counter = 0

    elif key == ord("d"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = os.path.join(OUT_DIR, "raw", f"diff_{ts}.jpg")
        cv2.imwrite(raw_path, frame)
        print(f"  [diff] Saved {raw_path}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. {count} face captures saved to {OUT_DIR}/")
print("To fine-tune: python src/training/train_face_cnn.py "
      "--data data/face --output models/face_cnn_finetuned.pth")
