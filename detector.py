"""
detector.py
-----------
Handles all YOLOv8 model loading and object detection logic.
Kept separate from app.py to keep concerns clean and modular.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colour palette for bounding boxes (one per class index) ───────────────────
PALETTE = [
    (255, 56,  56),  (255, 157,  151), (255, 112, 31),  (255, 178, 29),
    (207, 210,  49), (72,  249, 10),   (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,  212, 187),   (44,  153, 168), (0,  194, 255),
    (52,   69, 147), (100,  115, 255), (0,   24, 236),  (132,  56, 255),
    (82,    0, 133), (203,  56, 255),  (255,  149, 200),(255,  55, 199),
]


class ObjectDetector:
    """
    Wraps a YOLOv8 model and exposes a single `detect()` method.

    Parameters
    ----------
    model_name : str
        Any YOLOv8 checkpoint name, e.g. 'yolov8n.pt', 'yolov8s.pt'.
        The file is downloaded automatically on first run.
    conf_threshold : float
        Minimum confidence to keep a detection.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.40):
        print(f"[detector] Loading model: {model_name}")
        self.model = YOLO(model_name)
        self.conf = conf_threshold
        self.class_names = self.model.names   # dict {idx: 'label'}
        print("[detector] Model ready ✓")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _color_for(self, class_id: int) -> tuple:
        """Returns a consistent BGR colour for a given class index."""
        return PALETTE[class_id % len(PALETTE)]

    def _draw_box(self, img: np.ndarray, x1, y1, x2, y2,
                  label: str, conf: float, color: tuple) -> None:
        """Draws one bounding box + label on *img* in-place."""
        thickness = max(1, round(0.002 * max(img.shape[:2])))
        font_scale = max(0.4, 0.5 * thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background
        text = f"{label}  {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
        label_y = max(y1, th + 4)
        cv2.rectangle(img,
                      (x1, label_y - th - baseline - 2),
                      (x1 + tw + 4, label_y),
                      color, -1)

        # Label text
        cv2.putText(img, text,
                    (x1 + 2, label_y - baseline),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, image_bytes: bytes) -> dict:
        """
        Run YOLOv8 detection on raw image bytes.

        Returns
        -------
        dict with keys:
            output_path : str   – path to the annotated image
            count       : int   – number of detections
            labels      : list  – detected class names (with duplicates)
            error       : str | None
        """
        # Decode bytes → OpenCV image
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image. Please upload a valid JPEG or PNG."}

        # Run inference
        results = self.model.predict(img, conf=self.conf, verbose=False)[0]

        labels_found = []

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = self.class_names.get(cls_id, str(cls_id))
            color   = self._color_for(cls_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            self._draw_box(img, x1, y1, x2, y2, label, conf, color)
            labels_found.append(label)

        # Save annotated image
        filename  = f"result_{int(time.time() * 1000)}.jpg"
        out_path  = OUTPUT_DIR / filename
        cv2.imwrite(str(out_path), img)

        return {
            "output_path": str(out_path),
            "count":       len(labels_found),
            "labels":      labels_found,
            "error":       None,
        }
