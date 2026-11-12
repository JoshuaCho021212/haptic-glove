#!/usr/bin/env python3
"""
Run this ON THE PI.
Live camera feed with ArUco marker + surgical zone overlay,
served over Flask MJPEG so you can watch detection in real time
in a browser.

Usage:
    python3 marker_zone_live.py

Then open: http://<pi-ip>:5002
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response
import threading
import time

# --- Config (must match generate_marker.py) ---
MARKER_ID = 0
MARKER_SIZE_MM = 147
DICT = cv2.aruco.DICT_4X4_50
ZONE_RADIUS_IN_MARKER_WIDTHS = 2.0

app = Flask(__name__)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 960), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)


def apply_color_correction(frame):
    frame = frame.astype(np.float32)
    frame[:, :, 2] = np.clip(frame[:, :, 2] + 60, 0, 255)  # red +60
    frame[:, :, 0] = np.clip(frame[:, :, 0] - 40, 0, 255)  # blue -40
    frame = np.clip(frame * 1.3 + 20, 0, 255)               # alpha/beta
    return frame.astype(np.uint8)


def process_frame():
    frame = picam2.capture_array()
    corrected = apply_color_correction(frame)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)
    output = corrected.copy()

    status_text = "SEARCHING..."
    status_color = (0, 165, 255)  # orange

    if ids is not None and MARKER_ID in ids.flatten():
        idx = list(ids.flatten()).index(MARKER_ID)
        marker_corners = corners[idx][0]
        center = marker_corners.mean(axis=0)

        side_lengths = [
            np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
            for i in range(4)
        ]
        marker_px = np.mean(side_lengths)
        zone_radius_px = marker_px * ZONE_RADIUS_IN_MARKER_WIDTHS

        cv2.aruco.drawDetectedMarkers(output, corners, ids)
        cv2.circle(output, tuple(center.astype(int)), int(zone_radius_px),
                   (0, 0, 255), 3)
        cv2.putText(output, "SURGICAL ZONE",
                    (int(center[0] - 80), int(center[1] - zone_radius_px - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        px_per_mm = marker_px / MARKER_SIZE_MM
        status_text = f"DETECTED  scale={px_per_mm:.2f}px/mm  zone_r={zone_radius_px/px_per_mm:.0f}mm"
        status_color = (0, 255, 0)  # green
    else:
        # show rejected candidates faintly for debugging
        if rejected:
            cv2.polylines(output, [r.astype(int) for r in rejected],
                          True, (255, 0, 255), 1)

    cv2.putText(output, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    return output


def generate():
    while True:
        frame = process_frame()
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return '''
    <html>
      <head><title>Marker Zone Detection - Live</title></head>
      <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
        <h2>Live Marker / Surgical Zone Detection</h2>
        <img src="/stream" style="max-width:95%;border:2px solid #444;">
        <p>Green box = marker outline. Red circle = inferred surgical zone.
           Magenta outlines = rejected candidates (didn't decode).</p>
      </body>
    </html>
    '''


@app.route('/stream')
def stream():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True)
