"""
License Plate Detection + OCR
------------------------------
Uses:  YOLOv8 (pretrained on license plates) + EasyOCR
Run:   python lp_detect.py --video your_video.mp4
Output: output.mp4 with bounding boxes and stable plate text overlaid
"""

import cv2
import easyocr
import argparse
import re
from collections import Counter
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "best.pt"
CONF         = 0.5
LANGS        = ['en']

MAX_BOX_AREA_RATIO = 0.05
MIN_ASPECT         = 1.0
MAX_ASPECT         = 5.0

OCR_EVERY_N   = 4
CONFIRM_AFTER = 2
# ─────────────────────────────────────────────────────────────────────────────


def is_valid_box(x1, y1, x2, y2, frame_w, frame_h):
    w, h = x2 - x1, y2 - y1
    if h == 0:
        return False
    area_ratio = (w * h) / (frame_w * frame_h)
    aspect     = w / h
    if area_ratio > MAX_BOX_AREA_RATIO:
        return False
    if not (MIN_ASPECT < aspect < MAX_ASPECT):
        return False
    return True


def clean_ocr(text):
    return re.sub(r'[^A-Z0-9 ]', '', text.upper()).strip()


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def main(video_path, output_path):
    print("[1/3] Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    print("[2/3] Loading EasyOCR...")
    reader = easyocr.Reader(LANGS, gpu=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          FPS, (W, H))

    print(f"[3/3] Processing {total} frames  →  {output_path}")

    tracks    = {}
    next_id   = 0
    frame_idx = 0

    # Grows over time, never shrinks — every unique plate locked in gets added here
    all_detected_plates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"    frame {frame_idx}/{total}")

        # ── Step 1: Detect ────────────────────────────────────────────────────
        results     = model(frame, conf=CONF, verbose=False)[0]
        raw_boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
        valid_boxes = [b for b in raw_boxes
                       if is_valid_box(b[0], b[1], b[2], b[3], W, H)]

        # ── Step 2: Match to tracks ───────────────────────────────────────────
        matched_ids = []
        for box in valid_boxes:
            best_id, best_score = None, 0.3
            for tid, t in tracks.items():
                score = iou(box, t['box'])
                if score > best_score:
                    best_score = score
                    best_id    = tid
            if best_id is None:
                best_id = next_id
                tracks[best_id] = {'box': box, 'votes': Counter(), 'locked_text': None}
                next_id += 1
            tracks[best_id]['box'] = box
            matched_ids.append(best_id)

        tracks = {tid: t for tid, t in tracks.items() if tid in matched_ids}

        # ── Step 3: OCR ───────────────────────────────────────────────────────
        if frame_idx % OCR_EVERY_N == 0:
            for tid in matched_ids:
                t = tracks[tid]
                if t['locked_text']:
                    continue
                x1, y1, x2, y2 = t['box']
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ocr_raw  = reader.readtext(crop, detail=0)
                ocr_text = clean_ocr(" ".join(ocr_raw))
                ocr_text = ocr_text.replace(" ", "")  # strip spaces before storing
                if len(ocr_text) ==7:
                    t['votes'][ocr_text] += 1
                    top_text, top_count = t['votes'].most_common(1)[0]
                    if top_count >= CONFIRM_AFTER:
                        t['locked_text'] = top_text
                        # Add to persistent list only if new
                        if top_text not in all_detected_plates:
                            all_detected_plates.append(top_text)
                            print(f"  Locked: {top_text}")

        # ── Step 4: Draw bounding boxes on active detections ──────────────────
        for tid in matched_ids:
            t     = tracks[tid]
            x1, y1, x2, y2 = t['box']
            text  = t['locked_text'] or "Reading..."
            color = (0, 255, 0) if t['locked_text'] else (0, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_y = y1 - 8 if y1 - 8 > 10 else y2 + 20
            cv2.putText(frame, text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # ── Step 5: Persistent panel (top-right) ──────────────────────────────
        # Every plate ever locked stays here for the entire video
        panel_x = W - 300
        line_h  = 34
        n_lines = max(len(all_detected_plates), 1)
        panel_h = 14 + line_h * (n_lines + 1)

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 10, 0), (W, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Header
        cv2.putText(frame, "DETECTED PLATES",
                    (panel_x, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        if all_detected_plates:
            for i, plate in enumerate(all_detected_plates):
                y_pos = 26 + (i + 1) * line_h
                cv2.putText(frame, plate,
                            (panel_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 100), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Scanning...",
                        (panel_x, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\nDone! Saved to: {output_path}")
    print(f"All plates detected: {all_detected_plates}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True,  help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Path to output video")
    args = parser.parse_args()

    main(args.video, args.output)