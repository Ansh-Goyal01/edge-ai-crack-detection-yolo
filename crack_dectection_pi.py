import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import datetime
import os

# ── CONFIG ──
MODEL_PATH        = "best_ncnn_model"
CONFIDENCE_THRESH = 0.25    # balanced — not too sensitive, not too strict
IOU_THRESH        = 0.50    # higher = fewer overlapping boxes
IMGSZ             = 640
CAMERA_INDEX      = 0
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
ENABLE_PREPROCESS = True


# ── THREADED CAMERA ──
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret     = False
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret   = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# ── THREADED INFERENCE ──
# Runs detection in background so display never waits for AI
class ThreadedInference:
    def __init__(self, model, conf, iou, imgsz):
        self.model    = model
        self.conf     = conf
        self.iou      = iou
        self.imgsz    = imgsz
        self.results  = []
        self.detected = False
        self.running  = True
        self.lock     = threading.Lock()
        self._frame_to_process = None
        self._new_frame        = False
        self.thread   = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame):
        # Give inference thread a new frame to process
        with self.lock:
            self._frame_to_process = frame.copy()
            self._new_frame        = True

    def _run(self):
        while self.running:
            frame = None
            with self.lock:
                if self._new_frame and self._frame_to_process is not None:
                    frame           = self._frame_to_process
                    self._new_frame = False

            if frame is not None:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False,
                    device="cpu"
                )
                detected = any(len(r.boxes) > 0 for r in results)
                with self.lock:
                    self.results  = results
                    self.detected = detected
            else:
                time.sleep(0.001)  # Tiny sleep to avoid busy loop

    def get(self):
        with self.lock:
            return self.results, self.detected

    def stop(self):
        self.running = False
        self.thread.join()


# ── PREPROCESSING ──
def preprocess_frame(frame):
    blurred   = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(frame, 2.0, blurred, -1.0, 0)
    lab             = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe           = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_channel       = clahe.apply(l_channel)
    enhanced        = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── DRAW ──
def draw_detections(frame, results, crack_detected):
    for result in results:
        for box in result.boxes:
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            color        = (0, 0, 255) if conf >= 0.5 else (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label        = f"Crack {conf:.2f}"
            (tw, th), _  = cv2.getTextSize(
                               label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-6),
                          (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if crack_detected:
        h, w    = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "!! CRACK DETECTED !!", (10, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    return frame


# ── MAIN ──
def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded. Starting camera...")

    cam       = ThreadedCamera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    inferencer = ThreadedInference(model, CONFIDENCE_THRESH, IOU_THRESH, IMGSZ)
    time.sleep(2)

    print("[INFO] Running! Press 'q' to quit, 's' to save.")

    t_prev     = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # ── Send frame to inference thread (non-blocking) ──
        infer_frame = preprocess_frame(frame) if ENABLE_PREPROCESS else frame
        inferencer.submit(infer_frame)

        # ── Get latest results (non-blocking) ──
        results, crack_detected = inferencer.get()

        # ── Draw on display frame ──
        display_frame = frame.copy()
        display_frame = draw_detections(display_frame, results, crack_detected)

        # ── FPS ──
        t_now      = time.time()
        dt         = t_now - t_prev
        t_prev     = t_now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt if dt > 0 else 0)

        num_cracks = sum(len(r.boxes) for r in results)
        cv2.putText(display_frame, f"FPS:{fps_smooth:.1f}",
                    (5, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        cv2.putText(display_frame,
                    f"Cracks:{num_cracks} | CONF:{CONFIDENCE_THRESH}",
                    (5, FRAME_HEIGHT - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Crack Detection - Pi", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("/home/pi/detections", exist_ok=True)
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"/home/pi/detections/crack_{ts}.jpg"
            cv2.imwrite(path, display_frame)
            print(f"[SAVED] {path}")

    cam.release()
    inferencer.stop()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()