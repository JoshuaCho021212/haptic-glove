#!/usr/bin/env python3
"""
Integrated safety zone system:
- ArUco marker defines the surgical zone
- Custom YOLOv8-pose detects gloved hand keypoints
- System checks if fingertips are inside/outside the zone
- Visual + text alerts

http://<pi-ip>:5005
"""
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response
import threading
import time
import onnxruntime as ort

# --- Config ---
WIDTH, HEIGHT = 640, 480
MODEL_PATH = "/home/joshuacho/best.onnx"
PORT = 5005
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640
NUM_KEYPOINTS = 21
INFER_EVERY_N = 3

# ArUco config
MARKER_ID = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 145
ZONE_RADIUS_IN_MARKER_WIDTHS = 1.5

# Fingertip indices (0-indexed): thumb_tip=4, index_tip=8, middle_tip=12, ring_tip=16, pinky_tip=20
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

app = Flask(__name__)

print("Starting camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(3)

print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded.")

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

state = {"latest_frame": None}
lock = threading.Lock()


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def postprocess(output, scale):
    preds = output[0].transpose(1, 0)
    conf = preds[:, 4]
    mask = conf > CONF_THRESHOLD
    if not np.any(mask):
        return []
    cx, cy, w, h = preds[mask, 0], preds[mask, 1], preds[mask, 2], preds[mask, 3]
    conf = conf[mask]
    kpts_raw = preds[mask, 5:]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(boxes, conf, IOU_THRESHOLD)
    results = []
    for i in keep:
        box = boxes[i] / scale
        score = conf[i]
        kpts = kpts_raw[i].reshape(NUM_KEYPOINTS, 3)
        kpts[:, 0] /= scale
        kpts[:, 1] /= scale
        results.append((box, score, kpts))
    return results


def detect_marker_zone(gray):
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None or MARKER_ID not in ids.flatten():
        return None, None, None
    idx = list(ids.flatten()).index(MARKER_ID)
    marker_corners = corners[idx][0]
    center = marker_corners.mean(axis=0)
    side_lengths = [
        np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
        for i in range(4)
    ]
    marker_px = np.mean(side_lengths)
    zone_radius = marker_px * ZONE_RADIUS_IN_MARKER_WIDTHS
    return center, zone_radius, marker_corners


def check_fingertips_in_zone(kpts, zone_center, zone_radius):
    results = []
    for i, fidx in enumerate(FINGERTIP_INDICES):
        x, y, conf = kpts[fidx]
        if conf < 0.3:
            results.append((FINGER_NAMES[i], None, False))
            continue
        dist = np.sqrt((x - zone_center[0])**2 + (y - zone_center[1])**2)
        inside = dist < zone_radius
        results.append((FINGER_NAMES[i], (int(x), int(y)), inside))
    return results


def camera_loop():
    frame_count = 0
    last_detections = []
    last_zone = (None, None, None)

    while True:
        try:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # ArUco detection every frame (lightweight)
            zone_center, zone_radius, marker_corners = detect_marker_zone(gray)
            if zone_center is not None:
                last_zone = (zone_center, zone_radius, marker_corners)

            # YOLO inference every N frames
            if frame_count % INFER_EVERY_N == 0:
                blob, scale = preprocess(frame_bgr)
                output = session.run(None, {input_name: blob})[0]
                last_detections = postprocess(output, scale)

            # --- Draw everything ---
            display = frame_bgr.copy()
            z_center, z_radius, z_corners = last_zone

            # Draw surgical zone
            if z_center is not None:
                cv2.circle(display, tuple(z_center.astype(int)), int(z_radius),
                           (0, 0, 255), 2)
                cv2.putText(display, "SURGICAL ZONE",
                            (int(z_center[0] - 70), int(z_center[1] - z_radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if z_corners is not None:
                    cv2.polylines(display, [z_corners.astype(int)], True, (0, 255, 0), 2)

            # Draw hand skeleton + safety check
            any_inside = False
            for box, score, kpts in last_detections:
                # Draw skeleton
                for a, b in SKELETON:
                    if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(display, pt1, pt2, (255, 0, 255), 2)

                for x, y, kconf in kpts:
                    if kconf > 0.3:
                        cv2.circle(display, (int(x), int(y)), 4, (0, 255, 0), -1)

                # Safety check
                if z_center is not None:
                    finger_results = check_fingertips_in_zone(kpts, z_center, z_radius)
                    y_offset = 30
                    for name, pt, inside in finger_results:
                        if pt is None:
                            continue
                        if inside:
                            any_inside = True
                            color = (0, 0, 255)  # red = inside zone
                            cv2.circle(display, pt, 8, color, -1)
                        else:
                            color = (0, 255, 0)  # green = safe
                        cv2.putText(display, f"{name}: {'IN ZONE' if inside else 'SAFE'}",
                                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        y_offset += 25

            # Overall status
            if z_center is None:
                status = "NO MARKER"
                status_color = (0, 165, 255)
            elif not last_detections:
                status = "NO HAND"
                status_color = (0, 165, 255)
            elif any_inside:
                status = "WARNING: FINGER IN ZONE"
                status_color = (0, 0, 255)
            else:
                status = "ALL CLEAR"
                status_color = (0, 255, 0)

            cv2.putText(display, status, (WIDTH - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            with lock:
                state["latest_frame"] = display

            frame_count += 1
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)


def generate():
    while True:
        with lock:
            frame = state["latest_frame"]
        if frame is None:
            time.sleep(0.03)
            continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return '''
    <html><head><title>Haptic Glove Safety System</title></head>
    <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
      <h2>Integrated Safety Zone System</h2>
      <p>ArUco marker = surgical zone | YOLOv8-pose = hand tracking</p>
      <img src="/stream" style="max-width:95%;border:2px solid #444;">
      <p>Green = safe | Red = finger inside surgical zone</p>
    </body></html>
    '''


@app.route('/stream')
def stream():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=PORT, threaded=True)
