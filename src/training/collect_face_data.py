"""Collect face images for fine-tuning the FaceCNN on your own camera/subject.
Captures frames when SPACE is pressed, saves face crops and full frames.
Run: python src/training/collect_face_data.py
"""
import os, cv2, time, sys
from datetime import datetime

OUT_DIR = "data/face/custom"
os.makedirs(os.path.join(OUT_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "crops"), exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0
print("Face Data Collector")
print("===================")
print("Instructions:")
print("  1. Position your face in the frame at various distances/angles")
print("  2. Press SPACE to capture")
print("  3. Press 'd' for different lighting/background")
print("  4. Press 'q' to quit")
print()
print(f"Output: {OUT_DIR}/")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    # Draw center crosshair + capture hint
    display = frame.copy()
    cv2.line(display, (w//2 - 30, h//2), (w//2 + 30, h//2), (0, 255, 0), 2)
    cv2.line(display, (w//2, h//2 - 30), (w//2, h//2 + 30), (0, 255, 0), 2)
    cv2.putText(display, f"Captures: {count}  [SPACE] save  [d] flag  [q] quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Data Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        # Save full frame and crop
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = os.path.join(OUT_DIR, "raw", f"face_{ts}.jpg")
        crop_path = os.path.join(OUT_DIR, "crops", f"face_{ts}.jpg")
        cv2.imwrite(raw_path, frame)
        # Save center crop (40% of frame)
        cx, cy = w // 2, h // 2
        crop_sz = min(w, h) // 3
        crop = frame[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2]
        crop = cv2.resize(crop, (128, 128))
        cv2.imwrite(crop_path, crop)
        count += 1
        print(f"  [{count}] Saved {raw_path}")

    elif key == ord("d"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = os.path.join(OUT_DIR, "raw", f"diff_{ts}.jpg")
        cv2.imwrite(raw_path, frame)
        print(f"  [diff] Saved {raw_path} (no face, background reference)")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. {count} face captures saved to {OUT_DIR}/")
print("To fine-tune: python src/training/train_face_cnn.py --data data/face --output models/face_cnn_finetuned.pth")
