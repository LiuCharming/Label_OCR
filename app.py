"""Local browser UI for testing the label OCR pipeline."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from label_reader import IMAGE_SUFFIXES, build_ocr, scan_image

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "web_runs"
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = build_ocr("ch")
    return _ocr


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/runs/<batch>/<path:filename>")
def run_asset(batch: str, filename: str):
    return send_from_directory(RUNS / batch, filename)


@app.post("/api/scan")
def scan():
    files = request.files.getlist("images")
    valid = [f for f in files if f and Path(f.filename).suffix.lower() in IMAGE_SUFFIXES]
    if not valid:
        return jsonify(error="请选择至少一张支持的图片。"), 400
    batch = uuid.uuid4().hex[:10]
    batch_dir, upload_dir = RUNS / batch, RUNS / batch / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ocr, payload = get_ocr(), []
    try:
        for number, uploaded in enumerate(valid, start=1):
            name = secure_filename(uploaded.filename) or f"image_{number}.jpg"
            source = upload_dir / name
            uploaded.save(source)
            enhance = request.form.get('enhance') == 'true'
            original_ocr = request.form.get('original_ocr') == 'true'
            result, stem = scan_image(source, ocr, batch_dir, enhance=enhance, original_ocr=original_ocr), source.stem
            item = {"file": uploaded.filename, "label_found": result.label_found, "label_angle": result.label_angle, "qr_content": result.qr_content, "ocr_text": result.ocr_text, "confidence": result.confidence, "error": result.error, "original_url": f"/runs/{batch}/uploads/{name}"}
            if result.label_found:
                item.update(original_ocr=original_ocr, original_text=result.original_text, original_confidence=result.original_confidence)
                item['ocr_input_url'] = f'/runs/{batch}/{stem}_ocr_input.png'
                item.update(boundary_refined=result.boundary_refined, glare_ratio=result.glare_ratio,
                            quality_warning=result.quality_warning, glare_url=f'/runs/{batch}/{stem}_glare.png')
                item['correction_method'] = result.correction_method
                item.update(label_url=f"/runs/{batch}/{stem}_label.jpg", rectified_url=f"/runs/{batch}/{stem}_rectified.jpg", annotated_url=f"/runs/{batch}/{stem}_annotated.jpg")
                if enhance:
                    item['enhancement_note'] = result.enhancement_note
                    if (batch_dir / f'{stem}_enhanced.png').exists():
                        item['segmentation_url'] = f'/runs/{batch}/{stem}_segmentation.png'
                        item.update(enhanced_url=f'/runs/{batch}/{stem}_enhanced.png', enhanced_text=result.enhanced_text, enhanced_confidence=result.enhanced_confidence)
            payload.append(item)
    except Exception:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise
    return jsonify(items=payload, engine=ocr[0] if ocr else "unavailable")


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
