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
    enhanced_text: str = ""
    enhanced_confidence: float | None = None
    original_text: str = ""
    original_confidence: float | None = None
    enhancement_note: str = ""
    correction_method: str = "none"
    boundary_refined: bool = False
    glare_ratio: float = 0.0
    quality_warning: str = ""


def refine_boundary(image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], bool]:
    """Find a compact white rectangle enclosing the QR within the coarse crop."""
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    points = find_qr_points(crop)
    if points is None or len(points) != 4:
        return box, False
    scale = min(1.0, 800 / max(w, h))
    gray = cv2.cvtColor(cv2.resize(crop, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    center = tuple((points.mean(axis=0) * scale).astype(float))
    qr_area = abs(cv2.contourArea(points * scale))
    candidates = []
    for threshold in (160, 180, 200, 220, 240):
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, center, False) < 0:
                continue
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) < 1 or not 3 < area / max(qr_area, 1) < 18:
                continue
            fill = area / (rw * rh)
            if fill < .84 or not 1.25 < max(rw,rh)/min(rw,rh) < 3.5:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            candidates.append((fill, (x+round(bx/scale), y+round(by/scale), round(bw/scale), round(bh/scale))))
    if candidates:
        return max(candidates, key=lambda c:c[0])[1], True
    # A reflection can touch the lower label edge, making one connected blob.
    # For nearly horizontal labels, split at the sharp drop in bright-row span.
    edge = points[1] - points[0]
    if abs(edge[1]) < abs(edge[0]) * .25:
        bright = gray > 220
        spans = np.zeros(len(gray))
        for row, values in enumerate(bright):
            indexes = np.flatnonzero(values)
            if len(indexes):
                spans[row] = indexes[-1] - indexes[0] + 1
        rows = spans >= max(spans.max() * .8, 1)
        anchor = int(np.clip(center[1], 0, len(rows)-1))
        if rows[anchor]:
            top = anchor; bottom = anchor
            while top > 0 and rows[top-1]: top -= 1
            while bottom+1 < len(rows) and rows[bottom+1]: bottom += 1
            indexes = np.argwhere(bright[top:bottom+1])
            left, right = indexes[:,1].min(), indexes[:,1].max()
            if top <= (points[:,1]*scale).min() and bottom >= (points[:,1]*scale).max() and 1.25 < (right-left+1)/(bottom-top+1) < 3.5:
                return (x+round(left/scale), y+round(top/scale), round((right-left+1)/scale), round((bottom-top+1)/scale)), True
    return box, False


def inspect_glare(image: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Mark large near-clipped regions. White paper can also trigger this heuristic."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Exclude the crop border and dark housing. Require all color channels to
    # approach clipping; bright paper alone is not evidence of lost text.
    mask = (np.min(image, axis=2) >= 248).astype(np.uint8) * 255
    margin = max(2, min(gray.shape)//30)
    mask[:margin] = 0; mask[-margin:] = 0
    mask[:, :margin] = 0; mask[:, -margin:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    suspect = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= gray.size * .015:
            suspect[labels == i] = 255
    ratio = float(np.count_nonzero(suspect) / gray.size)
    overlay = image.copy()
    if ratio >= .03:
        tint = image.copy(); tint[suspect > 0] = (0, 140, 255)
        overlay = cv2.addWeighted(image, .65, tint, .35, 0)
        warning = '疑似反光或过曝：橙色区域接近纯白，可能影响笔画。请关闭闪光灯并换角度补拍；白色底纸也可能触发提示。'
    else:
        warning = ''
    return overlay, ratio, warning


def rectify_label(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Rectify the label plane from the ordered QR corners.

    Assumes the printed QR is square and coplanar with the text. Reject
    unstable homographies rather than allocating huge or inverted images.
    """
    points = find_qr_points(image)
    if points is None or len(points) != 4:
        return image, "none"
    points = points.astype(np.float32)
    side = float(np.mean(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))
    if side < 20:
        return image, "none"
    target = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    homogeneous = np.column_stack((corners, np.ones(4))) @ matrix.T
    denom = homogeneous[:, 2]
    if np.all(denom > 1e-6) or np.all(denom < -1e-6):
        projected = homogeneous[:, :2] / denom[:, None]
        lower = np.floor(projected.min(axis=0))
        upper = np.ceil(projected.max(axis=0))
        size = upper - lower
        if np.isfinite(size).all() and np.all(size > 0) and max(size) <= 6000 and np.prod(size) <= 4 * w * h:
            shift = np.array([[1, 0, -lower[0]], [0, 1, -lower[1]], [0, 0, 1]])
            result = cv2.warpPerspective(image, shift @ matrix, tuple(size.astype(int)),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))
            return result, "qr_perspective"
    angle = qr_rotation_angle(image)
    return (rotate_to_horizontal(image, angle), "rotation") if angle is not None else (image, "none")


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
    # Estimate in the QR's own coordinate system, not screen coordinates.
    # This rotates the text-side margin with the label at every orientation.
    points = np.asarray(qr_points, np.float32).reshape(-1, 2)[:4]
    unit = np.array([[0,0], [1,0], [1,1], [0,1]], np.float32)
    matrix = cv2.getPerspectiveTransform(unit, points)
    extent = np.array([[-2.4,-.35], [1.75,-.35], [1.75,1.8], [-2.4,1.8]], np.float32)
    projected = cv2.perspectiveTransform(extent[None], matrix)[0]
    if not np.isfinite(projected).all():
        return 0, 0, image_width, image_height
    minimum, maximum = projected.min(axis=0), projected.max(axis=0)
    x0 = int(np.clip(np.floor(minimum[0]), 0, image_width-1))
    y0 = int(np.clip(np.floor(minimum[1]), 0, image_height-1))
    x1 = int(np.clip(np.ceil(maximum[0]), x0+1, image_width))
    y1 = int(np.clip(np.ceil(maximum[1]), y0+1, image_height))
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


def mask_qr_for_ocr(image: np.ndarray) -> np.ndarray:
    """Blank every detected QR polygon only in an independent OCR copy."""
    output = image.copy()
    detector = cv2.QRCodeDetector()
    polygons = []
    try:
        found, points = detector.detectMulti(image)
        if found and points is not None:
            polygons = list(np.asarray(points, np.float32).reshape(-1, 4, 2))
    except cv2.error:
        pass
    if not polygons:
        points = find_qr_points(image)
        if points is not None:
            polygons = list(points.reshape(-1, 4, 2))
    for points in polygons:
        # Small padding removes boundary modules without a large axis-aligned
        # rectangle that could cover neighbouring text on rotated labels.
        center = points.mean(axis=0)
        expanded = center + (points - center) * 1.08
        cv2.fillConvexPoly(output, np.round(expanded).astype(np.int32), (255, 255, 255))
    return output


def recognise_text(ocr: Any | None, crop: np.ndarray) -> tuple[str, float | None]:
    if ocr is None:
        return "", None
    try:
        engine, recogniser = ocr
        crop = mask_qr_for_ocr(crop)
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


def label_interior_mask(image: np.ndarray) -> np.ndarray | None:
    """Find the paper contour in the rectified image, anchored by the QR."""
    points = find_qr_points(image)
    if points is None or len(points) != 4:
        return None
    scale = min(1., 800 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    center = tuple((points.mean(axis=0)*scale).astype(float))
    qr_area = abs(cv2.contourArea(points*scale))
    candidates = []
    for threshold in (100, 120, 140, 160, 180, 200, 220):
        binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, center, False) < 0: continue
            area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour)
            if not (3 < area/max(qr_area,1) < 18 and 1.3 < w/h < 3.6 and area/(w*h) > .9): continue
            if x <= 1 or y <= 1 or x+w >= gray.shape[1]-1 or y+h >= gray.shape[0]-1: continue
            candidates.append((area/(w*h), contour))
    if not candidates: return None
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [max(candidates,key=lambda x:x[0])[1]], -1, 255, -1)
    mask = cv2.erode(mask, np.ones((3,3),np.uint8))
    return cv2.resize(mask, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)


def enhance_label(image: np.ndarray) -> np.ndarray:
    """Whiten uneven paper illumination, then darken gray ink with gamma."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 25, 25)
    # Closing estimates the local light paper; it is applied only to the
    # background estimate, never to the output strokes themselves.
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255)
    # White stays white; a gray pixel at 200 becomes approximately 123.
    enhanced = np.round(255 * (normalized.astype(np.float32) / 255) ** 1.6).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def scan_image(path: Path, ocr: Any | None, output_dir: Path, enhance: bool = False, original_ocr: bool = False) -> ScanResult:
    image = cv2.imread(str(path))
    if image is None:
        return ScanResult(path.name, False, None, None, [], "", None, "Image could not be read")
    full_qr_points = find_qr_points(image)
    box = find_label(image, full_qr_points)
    if box is None:
        return ScanResult(path.name, False, None, None, decode_qr(image), "", None)
    box, boundary_refined = refine_boundary(image, box)
    x, y, w, h = box
    pad = max(6, round(max(w, h) * 0.025))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    crop = image[y0:y1, x0:x1]
    angle = qr_rotation_angle(crop)
    rectified, correction_method = rectify_label(crop)
    text, confidence = recognise_text(ocr, rectified)
    # Rotation normally improves tilted labels.  Keep a raw-crop fallback for
    # QR detections affected by strong perspective distortion.
    raw_text, raw_confidence = "", None
    if original_ocr:
        raw_text, raw_confidence = recognise_text(ocr, crop)
    qr = decode_qr(crop) or decode_qr(image)
    annotated = image.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 215, 255), max(2, image.shape[1] // 500))
    cv2.imwrite(str(output_dir / f"{path.stem}_label.jpg"), crop)
    cv2.imwrite(str(output_dir / f"{path.stem}_rectified.jpg"), rectified)
    cv2.imwrite(str(output_dir / f"{path.stem}_ocr_input.png"), mask_qr_for_ocr(rectified))
    cv2.imwrite(str(output_dir / f"{path.stem}_annotated.jpg"), annotated)
    result = ScanResult(path.name, True, [x0, y0, x1 - x0, y1 - y0], angle, qr, text, confidence)
    result.original_text, result.original_confidence = raw_text, raw_confidence
    result.correction_method = correction_method
    result.boundary_refined = boundary_refined
    glare_image, result.glare_ratio, result.quality_warning = inspect_glare(crop)
    cv2.imwrite(str(output_dir / f"{path.stem}_glare.png"), glare_image)
    if enhance:
        interior = label_interior_mask(rectified)
        if interior is None:
            result.enhancement_note = '未确认完整标签边界，已跳过增强。请使用矫正后识别结果。'
            return result
        clean = mask_qr_for_ocr(rectified)
        enhanced = enhance_label(clean)
        enhanced[interior == 0] = 255
        result.enhancement_note = '仅增强标签内部：背景已遮盖，先降噪，再温和压暗笔画。'
        cv2.imwrite(str(output_dir / f"{path.stem}_enhanced.png"), enhanced)
        result.enhanced_text, result.enhanced_confidence = recognise_text(ocr, enhanced)
    return result


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def write_reports(results: list[ScanResult], output_dir: Path) -> None:
    payload = [asdict(result) for result in results]
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(ScanResult.__dataclass_fields__))
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
    parser.add_argument("--enhance", action="store_true", help="Also compare OCR on a contrast-enhanced crop")
    parser.add_argument("--original-ocr", action="store_true", help="Also recognise the unrectified label crop")
    args = parser.parse_args()
    files = iter_images(args.source)
    if not files:
        parser.error("No supported image files were found.")
    args.output.mkdir(parents=True, exist_ok=True)
    ocr = None if args.no_ocr else build_ocr(args.lang)
    results = [scan_image(path, ocr, args.output, enhance=args.enhance, original_ocr=args.original_ocr) for path in files]
    write_reports(results, args.output)
    found = sum(item.label_found for item in results)
    print(f"Scanned {len(results)} image(s); located {found} label(s). Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
