# 🤖 AutoNav Detector

A clean, modular **real-time object detection web app** built with YOLOv8, Flask, and OpenCV.  
Upload an image → get back bounding boxes, class labels, and confidence scores — instantly.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat-square)

---

## ✨ Features

- 📤 Drag-and-drop or click-to-upload image interface
- 🔍 YOLOv8n inference with bounding boxes + confidence labels
- 🎨 Per-class colour palette for clear visualisation
- 💾 Saves annotated results in `outputs/`
- ⚠️ Handles errors gracefully (wrong type, no file, decode failure)
- 📱 Responsive dark UI — no heavy frontend frameworks

---

## 📁 Project Structure

```
autonav-detector/
├── app.py               # Flask routes & file validation
├── detector.py          # YOLOv8 model + drawing logic
├── templates/
│   └── index.html       # Frontend (HTML + vanilla JS)
├── static/
│   └── style.css        # Dark industrial theme
├── outputs/             # Auto-created; stores result images
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/autonav-detector.git
cd autonav-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **YOLOv8n weights** (`yolov8n.pt`, ~6 MB) are downloaded automatically on first run.

---

## 🚀 Run Locally

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🔧 Configuration

Edit the top of `detector.py` to change:

| Variable          | Default       | Description                          |
|-------------------|---------------|--------------------------------------|
| `model_name`      | `yolov8n.pt`  | Swap to `yolov8s.pt` for better accuracy |
| `conf_threshold`  | `0.40`        | Minimum confidence score to display  |

---

## 🛠 How It Works

| Layer | File | Responsibility |
|---|---|---|
| Frontend | `templates/index.html` | Drag-drop upload, fetch `/detect`, render result |
| Styles | `static/style.css` | Dark grid theme, amber accents, responsive layout |
| Routes | `app.py` | File validation, POST `/detect`, static file serving |
| Detection | `detector.py` | YOLO inference, box drawing, save to `outputs/` |

---

## 📸 Sample Screenshots

> Add screenshots of your results here after running the app!

```
screenshots/
├── upload_screen.png
└── result_with_boxes.png
```

---

## 🧠 Part of the AutoNav Rover Project

This detector is built as a standalone web demo of the vision pipeline used in the **AutoNav Rover** — an autonomous indoor mapping robot using SLAM, ROS 2, LiDAR, and a Jetson Nano.

---

## 📜 License

MIT — free to use, learn from, and extend.
