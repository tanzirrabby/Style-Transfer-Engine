from __future__ import annotations

import io
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from style_transfer import run_style_transfer

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file_storage, prefix: str) -> Path:
    filename = file_storage.filename or "image.png"
    ext = filename.rsplit(".", 1)[-1].lower()
    output = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex}.{ext}"
    file_storage.save(output)
    return output


def _parse_float(name: str, default: float) -> float:
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return default


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/preview")
def preview():
    content = request.files.get("content")
    style = request.files.get("style")

    if not content or not style:
        return jsonify({"error": "Both content and style images are required."}), 400

    if not (_allowed(content.filename or "") and _allowed(style.filename or "")):
        return jsonify({"error": "Unsupported file format."}), 400

    content_path = _save_upload(content, "content")
    style_path = _save_upload(style, "style")

    style_weight = _parse_float("style_weight", 1e6)
    content_weight = _parse_float("content_weight", 1.0)
    num_steps = int(_parse_float("steps", 200))

    out_img = run_style_transfer(
        content_image=content_path,
        style_image=style_path,
        style_weight=style_weight,
        content_weight=content_weight,
        num_steps=num_steps,
        image_size=512,
    )

    buffer = io.BytesIO()
    out_img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.post("/api/batch")
def batch_export():
    style = request.files.get("style")
    content_files = request.files.getlist("contents")

    if not style or not content_files:
        return jsonify({"error": "A style image and at least one content image are required."}), 400

    style_path = _save_upload(style, "style")

    style_weight = _parse_float("style_weight", 1e6)
    content_weight = _parse_float("content_weight", 1.0)
    num_steps = int(_parse_float("steps", 200))

    batch_id = uuid.uuid4().hex
    batch_dir = EXPORT_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    for idx, content in enumerate(content_files):
        if not _allowed(content.filename or ""):
            continue

        content_path = _save_upload(content, f"batch_{idx}")
        out_img = run_style_transfer(
            content_image=content_path,
            style_image=style_path,
            style_weight=style_weight,
            content_weight=content_weight,
            num_steps=num_steps,
            image_size=512,
        )

        stem = Path(content.filename or f"image_{idx}").stem
        output_path = batch_dir / f"{stem}_stylized.png"
        out_img.save(output_path)
        exported.append(str(output_path.relative_to(BASE_DIR)))

    return jsonify({"batch_id": batch_id, "exported": exported, "count": len(exported)})


if __name__ == "__main__":
    app.run(debug=True)
