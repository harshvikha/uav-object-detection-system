"""
app.py
------
Flask application entry point.
Handles routing, file validation, and bridges the frontend to detector.py.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from detector import ObjectDetector

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload cap

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# Load model once at startup (not on every request)
detector = ObjectDetector(model_name="yolov8n.pt", conf_threshold=0.40)


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Returns True if the filename has a permitted image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Accepts a multipart/form-data request with an 'image' field.
    Returns JSON:
        { success, image_url, count, labels, error }
    """
    # ── Validate upload ───────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file field named 'image' in the request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected. Please choose an image first."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # ── Run detection ─────────────────────────────────────────────────────────
    image_bytes = file.read()
    result = detector.detect(image_bytes)

    if result.get("error"):
        return jsonify({"success": False, "error": result["error"]}), 422

    # Build a URL the browser can fetch for the annotated image
    output_filename = Path(result["output_path"]).name
    image_url = f"/outputs/{output_filename}"

    return jsonify({
        "success":   True,
        "image_url": image_url,
        "count":     result["count"],
        "labels":    result["labels"],
    })


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve saved detection-result images from the outputs/ folder."""
    return send_from_directory("outputs", filename)


# ── Main ──────────────────────────────────────────────────────────────────────

import os

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 AutoNav Detector running on port {port}")
    app.run(host="0.0.0.0", port=port)