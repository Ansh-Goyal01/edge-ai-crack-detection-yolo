![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green)
![Status](https://img.shields.io/badge/Status-Active-success)



# 🚧 Edge AI Crack Detection using YOLO (Real-Time)

> Real-time surface crack detection system using YOLO, optimized for **edge devices like Raspberry Pi**, with enhanced small-crack detection via preprocessing and high-resolution inference.

---

## 🧠 Key Highlights

* ⚡ Real-time crack detection (live camera feed)
* 🎯 Detects **small and thin cracks** using high-resolution inference (1280px)
* 🧩 Edge deployment on Raspberry Pi (low-resource optimization)
* 🧠 Advanced preprocessing (CLAHE + sharpening)
* 📉 Tuned confidence thresholds for improved recall
* 🛠️ Modular and scalable code structure

---

## 🏗️ System Architecture

```
Camera Input → Preprocessing → YOLO Model → Detection → Visualization
```

---

## ⚙️ Tech Stack

| Category        | Tools Used         |
| --------------- | ------------------ |
| Language        | Python             |
| Computer Vision | OpenCV             |
| Deep Learning   | YOLO (Ultralytics) |
| Edge Device     | Raspberry Pi       |
| Optimization    | NCNN (optional)    |

---

## 📁 Project Structure

```
crack-detection-yolo/
│
├── crack_detection.py        # Main pipeline (desktop)
├── crack_detection_pi.py     # Raspberry Pi optimized version
├── crack_utils.py            # Preprocessing + helper functions
├── best.pt                   # YOLO trained weights
├── requirements.txt
│
├── assets/
│   ├── demo.gif
│   └── output.jpg
```

---

## ▶️ Demo

> Add your demo here (VERY IMPORTANT for resume impact)

![Demo](assets/demo.gif)

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/edge-ai-crack-detection-yolo.git
cd edge-ai-crack-detection-yolo

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Usage

### 💻 Run on PC

```bash
python crack_detection.py
```

### 🍓 Run on Raspberry Pi

```bash
python crack_detection_pi.py
```

---

## 🎯 Model Optimization Techniques

* Increased `IMGSZ = 1280` for small object detection
* Reduced `CONFIDENCE_THRESH = 0.15`
* Applied CLAHE for contrast enhancement
* Sharpening to highlight crack edges
* Frame skipping for real-time performance

---

## 📊 Performance

| Metric             | Value                             |
| ------------------ | --------------------------------- |
| Inference Speed    | ~50 FPS                           |
| Resolution         | 1280                              |
| Device             | Raspberry Pi / Laptop             |
| Detection Accuracy | High (optimized for small cracks) |

---

## 🚀 Future Improvements

* 📱 Mobile app integration
* ☁️ Cloud dashboard for monitoring
* 📊 Crack severity classification (ML model)
* 📍 Geo-tagging cracks for smart city use

---

## 🧑‍💻 Author

**Ansh Goyal**
B.Tech ECE | AI + Computer Vision Enthusiast

---

## 📜 License

MIT License
