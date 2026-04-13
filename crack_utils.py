"""
Crack Detection Utilities — Alerts, Saving, Logging
"""

import os
import csv
import datetime
import threading
import sys


# ── 1. ALERT SYSTEM ──
class AlertSystem:
    def __init__(self, cooldown_seconds=3.0, sound_file=None):
        self.cooldown   = cooldown_seconds
        self.sound_file = sound_file
        self._last_alert = 0.0
        self._lock = threading.Lock()

    def trigger(self, force=False):
        import time
        now = time.time()
        with self._lock:
            if not force and (now - self._last_alert) < self.cooldown:
                return False
            self._last_alert = now
        threading.Thread(target=self._play, daemon=True).start()
        return True

    def _play(self):
        try:
            if self.sound_file and os.path.exists(self.sound_file):
                self._play_file(self.sound_file)
            else:
                self._system_beep()
        except Exception as e:
            print(f"[AlertSystem] Audio error: {e}")

    def _play_file(self, path):
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sys.platform == "darwin":
            os.system(f"afplay '{path}' &")
        else:
            os.system(f"aplay -q '{path}' &")

    def _system_beep(self):
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 300)
        elif sys.platform == "darwin":
            os.system("osascript -e 'beep 1'")
        else:
            ret = os.system("beep -f 1000 -l 300 2>/dev/null")
            if ret != 0:
                print("\a", end="", flush=True)


# ── 2. FRAME SAVER ──
class FrameSaver:
    def __init__(self, save_dir="detections/", cooldown_seconds=2.0,
                 max_files=500, quality=90):
        self.save_dir  = save_dir
        self.cooldown  = cooldown_seconds
        self.max_files = max_files
        self.quality   = quality
        self._last_save = 0.0
        os.makedirs(save_dir, exist_ok=True)

    def save(self, frame, num_cracks=1):
        import time
        import cv2
        now = time.time()
        if (now - self._last_save) < self.cooldown:
            return None
        self._last_save = now
        self._cleanup()
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"crack_{num_cracks}det_{ts}.jpg"
        path     = os.path.join(self.save_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return path

    def _cleanup(self):
        files = sorted(
            [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir)
             if f.endswith(".jpg")],
            key=os.path.getmtime
        )
        while len(files) >= self.max_files:
            os.remove(files.pop(0))


# ── 3. DETECTION LOGGER ──
class DetectionLogger:
    def __init__(self, log_dir="detections/logs/"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        date_str      = datetime.datetime.now().strftime("%Y%m%d")
        self.txt_path = os.path.join(log_dir, f"detection_log_{date_str}.txt")
        self.csv_path = os.path.join(log_dir, f"detection_data_{date_str}.csv")
        self._session_count = 0
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "num_cracks",
                                 "avg_confidence", "frame_path"])

    def log(self, num_cracks, confidences, frame_path=""):
        self._session_count += 1
        ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        txt_line = (f"[{ts}] Cracks: {num_cracks} | "
                    f"Avg Conf: {avg_conf:.4f} | "
                    f"Frame: {frame_path or 'not saved'}\n")
        with open(self.txt_path, "a") as f:
            f.write(txt_line)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, num_cracks, avg_conf, frame_path])

        print(f"[LOG #{self._session_count}] {txt_line.strip()}")

    def summary(self):
        if not os.path.exists(self.csv_path):
            return {}
        rows = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        if not rows:
            return {"total_events": 0}
        confs = [float(r["avg_confidence"]) for r in rows if r["avg_confidence"]]
        return {
            "total_events"    : len(rows),
            "total_cracks"    : sum(int(r["num_cracks"]) for r in rows),
            "avg_confidence"  : round(sum(confs) / len(confs), 4) if confs else 0,
            "log_path"        : self.csv_path
        }