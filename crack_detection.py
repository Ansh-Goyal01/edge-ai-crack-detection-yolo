import cv2
import numpy as np
from ultralytics import YOLO
import time

# ── CONFIG ──
MODEL_PATH        = "best.pt"
CONFIDENCE_THRESH = 0.15
IOU_THRESH        = 0.45
IMGSZ             = 1280      # High res single pass — catches small cracks
CAMERA_INDEX      = 0
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
ENABLE_PREPROCESS = True
SKIP_FRAMES       = 1         # Process every 2nd frame (0 = process all)


# ── PREPROCESSING (fast version) ──
def preprocess_frame(frame):
    # Fast sharpening only — no denoising
    blurred   = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(frame, 2.0, blurred, -1.0, 0)

    # CLAHE on L channel
    lab              = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b  = cv2.split(lab)
    clahe            = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_channel        = clahe.apply(l_channel)
    enhanced         = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── DRAW ──
def draw_detections(frame, results, crack_detected):
    for result in results:
        for box in result.boxes:
            conf        = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            color       = (0, 0, 255) if conf >= 0.5 else (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label       = f"Crack {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if crack_detected:
        h, w    = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "CRACK DETECTED", (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
    return frame


# ── MAIN ──
def main():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)
    model.fuse()
    print("[INFO] Model loaded. Opening camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Running! Press 'q' to quit, 's' to save frame.")

    t_prev        = time.time()
    fps_smooth    = 0.0
    frame_count   = 0
    last_results  = []
    last_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count   += 1
        display_frame  = frame.copy()

        # ── Only run inference every (SKIP_FRAMES+1) frames ──
        if frame_count % (SKIP_FRAMES + 1) == 0:
            infer_frame  = preprocess_frame(frame) if ENABLE_PREPROCESS else frame
            last_results = model.predict(
                source=infer_frame,
                conf=CONFIDENCE_THRESH,
                iou=IOU_THRESH,
                imgsz=IMGSZ,
                verbose=False,
                device="cpu"
            )
            last_detected = any(len(r.boxes) > 0 for r in last_results)

        # ── Always draw last known results ──
        display_frame = draw_detections(display_frame, last_results, last_detected)

        # ── FPS ──
        t_now      = time.time()
        dt         = t_now - t_prev
        t_prev     = t_now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt if dt > 0 else 0)

        num_cracks = sum(len(r.boxes) for r in last_results)
        cv2.putText(display_frame, f"FPS: {fps_smooth:.1f}",
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
        cv2.putText(display_frame,
                    f"Cracks: {num_cracks} | IMGSZ:{IMGSZ} | CONF:{CONFIDENCE_THRESH}",
                    (10, FRAME_HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Crack Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import datetime, os
            os.makedirs("detections", exist_ok=True)
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"detections/crack_{ts}.jpg"
            cv2.imwrite(path, display_frame)
            print(f"[SAVED] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()