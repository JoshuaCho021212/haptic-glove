#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response
import threading
import time
import onnxruntime as ort

WIDTH, HEIGHT = 320, 240
MODEL_PATH = "/home/joshuacho/best.onnx"
PORT = 5004
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640
NUM_KEYPOINTS = 21
INFER_EVERY_N = 3

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


def draw_detections(display, detections):
    if not detections:
        cv2.putText(display, "NO HAND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        for box, score, kpts in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
            for a, b in SKELETON:
                if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                    pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                    pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(display, pt1, pt2, (255, 0, 255), 2)
            for x, y, kconf in kpts:
                if kconf > 0.3:
                    cv2.circle(display, (int(x), int(y)), 4, (0, 255, 0), -1)
    return display


def camera_loop():
    frame_count = 0
    last_detections = []
    while True:
        try:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if frame_count % INFER_EVERY_N == 0:
                blob, scale = preprocess(frame_bgr)
                output = session.run(None, {input_name: blob})[0]
                last_detections = postprocess(output, scale)

            display = draw_detections(frame_bgr.copy(), last_detections)

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
    <html><head><title>YOLO Hand Detection</title></head>
    <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
      <h2>Custom YOLOv8-pose Hand Detection</h2>
      <img src="/stream" style="max-width:95%;border:2px solid #444;">
    </body></html>
    '''


@app.route('/stream')
def stream():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=PORT, threaded=True)
