from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import uuid
import zipfile
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename

from ml.model import load_model
from ml.inference import run_inference
from utils.image_utils import make_overlay, make_heatmap
from geo.postprocess import mask_to_geojson

# ------------------ APP SETUP ------------------

app = Flask(__name__)

# Base directory (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use /tmp for Render (ephemeral storage)
UPLOAD_DIR = "/tmp/uploads"
BASE_RESULT_DIR = "/tmp/results"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BASE_RESULT_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path fix
MODEL_PATH = os.path.join(BASE_DIR, "model", "unet_mumbai_roads.pth")

# Load model
model = load_model(MODEL_PATH, device)
model.eval()

# ------------------ HELPERS ------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def road_coverage(mask, threshold=0.3):
    binary = (mask > threshold).astype("uint8")
    total = binary.size
    road = binary.sum()
    return (road / total) * 100.0


def connectivity_status(mask, threshold=0.3):
    binary = (mask > threshold).astype("uint8") * 255
    n_labels, _ = cv2.connectedComponents(binary)
    if n_labels <= 2:
        return "Well Connected"
    return "Fragmented"


# ------------------ ROUTES ------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=False)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if not file or file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            result=False,
            error="Invalid file type. Please upload PNG/JPG/TIFF."
        )

    # Unique session
    session_id = str(uuid.uuid4())[:8]
    result_dir = os.path.join(BASE_RESULT_DIR, session_id)
    os.makedirs(result_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
    file.save(img_path)

    img = cv2.imread(img_path)
    if img is None:
        return render_template(
            "index.html",
            result=False,
            error="Invalid image file."
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    # Inference
    mask, inf_time = run_inference(model, img, device)

    # Metrics
    cov = road_coverage(mask)
    conn = connectivity_status(mask)

    if cov < 10:
        density = "Low"
    elif cov < 25:
        density = "Medium"
    else:
        density = "High"

    if conn == "Well Connected" and cov >= 12:
        health = "Good"
    elif cov >= 6:
        health = "Moderate"
    else:
        health = "Poor"

    insight = (
        f"{density} road density detected with {conn.lower()} connectivity. "
        f"Roads cover {round(float(cov), 2)}% of the analysed area."
    )

    # Binary mask
    bin_mask = (mask > 0.3).astype("uint8") * 255

    # Save outputs
    cv2.imwrite(
        os.path.join(result_dir, "input_image.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(os.path.join(result_dir, "predicted_mask.png"), bin_mask)

    overlay = make_overlay(img, bin_mask / 255.0, threshold=0.3)
    cv2.imwrite(
        os.path.join(result_dir, "overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    heat = make_heatmap(mask)
    cv2.imwrite(
        os.path.join(result_dir, "confidence_heatmap.png"),
        cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)
    )

    mask_to_geojson(
        bin_mask / 255.0,
        os.path.join(result_dir, "road_centerlines.geojson"),
        threshold=0.3
    )

    info = {
        "session_id": session_id,
        "device": str(device),
        "inference_time_s": round(float(inf_time), 3),
        "road_coverage_pct": round(float(cov), 2),
        "connectivity": conn,
        "density": density,
        "health": health,
        "insight": insight,
    }

    return render_template("index.html", result=True, info=info)


@app.route("/download_results/<session_id>", methods=["GET"])
def download_results(session_id):

    if not session_id.isalnum() or len(session_id) > 12:
        return "Invalid session", 400

    result_dir = os.path.join(BASE_RESULT_DIR, session_id)
    zip_path = os.path.join(result_dir, "results.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "input_image.png",
            "predicted_mask.png",
            "overlay.png",
            "confidence_heatmap.png",
            "road_centerlines.geojson",
        ]:
            fp = os.path.join(result_dir, fname)
            if os.path.exists(fp):
                zf.write(fp, arcname=fname)

    return send_file(zip_path, as_attachment=True)
@app.route("/results/<session_id>/<filename>")
def serve_result(session_id, filename):
    result_dir = os.path.join(BASE_RESULT_DIR, session_id)
    file_path = os.path.join(result_dir, filename)

    if not os.path.exists(file_path):
        return "File not found", 404

    return send_file(file_path)

# ------------------ RUN ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)