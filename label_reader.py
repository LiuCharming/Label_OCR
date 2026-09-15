"""Batch locate product labels, decode QR codes, and recognise label text.

Designed for photos such as the supplied samples: a light, rectangular label on
an otherwise darker object.  It also works with ordinary document/photos when
the label has sufficient contrast with its immediate background.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class ScanResult:
    file: str
    label_found: bool
    label_box: list[int] | None
    label_angle: float | None
    qr_content: list[str]
    ocr_text: str
    confidence: float | None
    error: str | None = None


def find_qr_points(image: np.ndarray) -> np.ndarray | None:
    """Return QR corners even when the basic detector cannot localise it."""
    detector = cv2.QRCodeDetector()
    try:
        found, points = detector.detect(image)
        if found and points is not None:
            return np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except cv2.error:
        pass
    try:
        found, _decoded, points, _straight = detector.detectAndDecodeMulti(image)
        if found and points is not None:
            return np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except cv2.error:
        pass
    return None


def label_box_from_qr(qr_points: np.ndarray, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Estimate the whole label rectangle from a QR printed on its right side."""
    min_x, min_y = np.min(qr_points, axis=0)
    max_x, max_y = np.max(qr_points, axis=0)
    side = max(20.0, (max_x - min_x + max_y - min_y) / 2)
    # The supplied inventory-label layout has the QR on the right, with text
    # to its left. The generous margins tolerate camera perspective.
    x0 = max(0, round(min_x - 2.4 * side))
    y0 = max(0, round(min_y - 0.3 * side))
    x1 = min(image_width, round(max_x + 0.75 * side))
    y1 = min(image_height, round(max_y + 0.75 * side))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def find_label(image: np.ndarray, qr_points: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    """Return the strongest light rectangular label candidate as x, y, w, h."""
    original_height, original_width = image.shape[:2]
    # Work at a stable resolution so morphology is not dependent on phone
    # camera megapixels.  These labels are deliberately white, so a bright
    # threshold separates them more reliably than an adaptive one on wood.
    scale = min(1.0, 1000 / original_width)
    resized = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    height, width = resized.shape[:2]
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = width * height
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    qr_center = np.mean(qr_points, axis=0) * scale if qr_points is not None else None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * 0.001 or area > image_area * 0.22:
            continue
        aspect = w / max(h, 1)
        if not 0.45 <= aspect <= 7.5:
            continue
        # Require a bright interior: this rejects dark product outlines.
        inset_x, inset_y = max(2, w // 12), max(2, h // 12)
        inside = gray[y + inset_y : y + h - inset_y, x + inset_x : x + w - inset_x]
        if inside.size == 0:
            continue
        brightness = float(np.mean(inside)) / 255
        contour_fill = cv2.contourArea(contour) / max(area, 1)
        # Most printed inventory labels are 1.5--2.5 times wider than high.
        # The looser term still permits rotated labels.
        shape_score = max(0.15, 1 - abs(np.log(aspect / 2.0)) / 1.8)
        score = (area / image_area) ** 0.45 * brightness * shape_score * (0.5 + contour_fill / 2)
        # A real label must contain its QR. This turns the QR into an anchor
        # and prevents bright wood/table regions from winning the score.
        if qr_center is not None and x <= qr_center[0] <= x + w and y <= qr_center[1] <= y + h:
            score *= 20
        candidates.append((score, (x, y, w, h)))
    candidate = max(candidates, default=(0.0, None), key=lambda item: item[0])[1]
    if qr_points is not None:
        # If thresholding did not join the white label into one contour (as on
        # the dark/shadowed sample), derive it directly from the QR geometry.
        if candidate is None:
            return label_box_from_qr(qr_points, original_width, original_height)
        x, y, w, h = candidate
        if not (x <= qr_center[0] <= x + w and y <= qr_center[1] <= y + h):
            return label_box_from_qr(qr_points, original_width, original_height)
    if candidate is None:
        return None
    x, y, w, h = candidate
    return tuple(round(value / scale) for value in (x, y, w, h))  # type: ignore[return-value]


def decode_qr(crop: np.ndarray) -> list[str]:
    detector = cv2.QRCodeDetector()
    values: list[str] = []
    try:
        ok, decoded, _points, _ = detector.detectAndDecodeMulti(crop)
        if ok:
            values.extend(value for value in decoded if value)
    except cv2.error:
        pass
    if not values:
        try:
            value, _points, _ = detector.detectAndDecode(crop)
            if value:
                values.append(value)
        except cv2.error:
            pass
    return list(dict.fromkeys(values))


def qr_rotation_angle(image: np.ndarray) -> float | None:
    """Get label clockwise tilt from the QR's ordered top edge, in degrees."""
    points = find_qr_points(image)
    if points is None:
        return None
    corners = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(corners) < 4:
        return None
    # QRCodeDetector orders points top-left, top-right, bottom-right,
    # bottom-left. Average opposite edges to reduce perspective noise.
    top = corners[1] - corners[0]
    bottom = corners[2] - corners[3]
    vector = (top + bottom) / 2
    if np.linalg.norm(vector) < 2:
        return None
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def rotate_to_horizontal(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate QR-derived label tilt back to horizontal without clipping corners."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    # QRCodeDetector's image-coordinate angle follows OpenCV's rotation sign.
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(
        image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def build_ocr(language: str) -> Any | None:
    """Prefer PP-OCR's compact mobile models for low CPU usage on label crops."""
    try:
        from paddleocr import PaddleOCR

        return (
            "paddleocr",
            PaddleOCR(
                lang=language,
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv4_mobile_rec",
                text_det_limit_side_len=640,
                # Avoid oneDNN/PIR incompatibilities on some Windows CPUs.
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
        )
    except Exception:
        try:
            from rapidocr_onnxruntime import RapidOCR

            return ("rapidocr", RapidOCR())
        except Exception as exc:  # Keep batch scans useful if the model/runtime is absent.
            print(f"Warning: OCR unavailable ({exc}). QR and label location will still run.")
            return None


def recognise_text(ocr: Any | None, crop: np.ndarray) -> tuple[str, float | None]:
    if ocr is None:
        return "", None
    try:
        engine, recogniser = ocr
        if engine == "rapidocr":
            result, _elapsed = recogniser(crop)
            if not result:
                return "", None
            lines = [str(line[1]) for line in result if str(line[1]).strip()]
            scores = [float(line[2]) for line in result]
            return "\n".join(lines), sum(scores) / len(scores) if scores else None

        result = recogniser.predict(crop)
        lines: list[str] = []
        scores: list[float] = []
        for page in result:
            data = page.json.get("res", {})
            texts = data.get("rec_texts", [])
            confidences = data.get("rec_scores", [])
            lines.extend(str(text) for text in texts if str(text).strip())
            scores.extend(float(score) for score in confidences)
        return "\n".join(lines), (sum(scores) / len(scores) if scores else None)
    except Exception:
        return "", None


def ocr_quality(text: str, confidence: float | None) -> float:
    """Prefer a result with more independently detected text lines."""
    return len([line for line in text.splitlines() if line.strip()]) * (confidence or 0)


def scan_image(path: Path, ocr: Any | None, output_dir: Path) -> ScanResult:
    image = cv2.imread(str(path))
    if image is None:
        return ScanResult(path.name, False, None, None, [], "", None, "Image could not be read")
    full_qr_points = find_qr_points(image)
    box = find_label(image, full_qr_points)
    if box is None:
        return ScanResult(path.name, False, None, None, decode_qr(image), "", None)
    x, y, w, h = box
    pad = max(6, round(max(w, h) * 0.025))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    crop = image[y0:y1, x0:x1]
    angle = qr_rotation_angle(crop)
    rectified = rotate_to_horizontal(crop, angle) if angle is not None else crop
    text, confidence = recognise_text(ocr, rectified)
    # Rotation normally improves tilted labels.  Keep a raw-crop fallback for
    # QR detections affected by strong perspective distortion.
    if angle is not None:
        raw_text, raw_confidence = recognise_text(ocr, crop)
        if ocr_quality(raw_text, raw_confidence) > ocr_quality(text, confidence):
            text, confidence = raw_text, raw_confidence
    # If a strongly lit background fooled the label locator, use full-image OCR
    # as a recovery path.  Keep only serial/field-like lines so unrelated
    # posters or packaging text is not written into the label result.
    if not text and ocr is not None:
        recovered, recovered_confidence = recognise_text(ocr, image)
        field_lines = [
            line for line in recovered.splitlines()
            if re.search(r"(?:[A-Za-z]{1,8}\s*[:：]\s*[\w.-]+)|(?:\b\d{5,}\b)|(?:\b[A-Za-z]\d[\w.-]{3,}\b)", line)
            and not re.fullmatch(r"\d{3,4}-\d{7,8}", line.strip())  # telephone-like background text
        ]
        if field_lines:
            text, confidence = "\n".join(field_lines), recovered_confidence
    qr = decode_qr(crop) or decode_qr(image)
    annotated = image.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 215, 255), max(2, image.shape[1] // 500))
    cv2.imwrite(str(output_dir / f"{path.stem}_label.jpg"), crop)
    cv2.imwrite(str(output_dir / f"{path.stem}_rectified.jpg"), rectified)
    cv2.imwrite(str(output_dir / f"{path.stem}_annotated.jpg"), annotated)
    return ScanResult(path.name, True, [x0, y0, x1 - x0, y1 - y0], angle, qr, text, confidence)


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def write_reports(results: list[ScanResult], output_dir: Path) -> None:
    payload = [asdict(result) for result in results]
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "label_found", "label_box", "label_angle", "qr_content", "ocr_text", "confidence", "error"])
        writer.writeheader()
        for item in payload:
            item["label_box"] = json.dumps(item["label_box"], ensure_ascii=False)
            item["qr_content"] = " | ".join(item["qr_content"])
            writer.writerow(item)


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate labels and recognise QR/text from product photos.")
    parser.add_argument("source", type=Path, help="An image file or directory to scan")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Directory for crops, marked images and reports")
    parser.add_argument("--lang", default="ch", help="PaddleOCR language, e.g. ch or en")
    parser.add_argument("--no-ocr", action="store_true", help="Only locate labels and decode QR codes")
    args = parser.parse_args()
    files = iter_images(args.source)
    if not files:
        parser.error("No supported image files were found.")
    args.output.mkdir(parents=True, exist_ok=True)
    ocr = None if args.no_ocr else build_ocr(args.lang)
    results = [scan_image(path, ocr, args.output) for path in files]
    write_reports(results, args.output)
    found = sum(item.label_found for item in results)
    print(f"Scanned {len(results)} image(s); located {found} label(s). Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
