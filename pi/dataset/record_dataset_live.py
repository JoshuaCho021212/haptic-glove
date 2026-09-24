#!/usr/bin/env python3
"""
Run this ON THE PI.
Live camera preview in browser (always on) + terminal-triggered
tagged clip recording for the hand dataset.

Usage:
    python3 record_dataset_live.py

1. Open http://<pi-ip>:5003 in a browser to see the live feed.
2. Position the glove/hand/camera how you want.
3. Back in the terminal, press Enter, type a tag, and recording
   starts for DURATION_SEC while you keep watching the browser
   (a red "REC" indicator + timer overlays on the stream).
4. Repeat for as many clips/tags as you want. Ctrl+C to quit.
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response
import threading
import time
import os
from datetime import datetime

# --- Config ---
WIDTH, HEIGHT = 1536, 864
FPS = 30
DURATION_SEC_DEFAULT = 15
OUT_DIR = os.path.expanduser("~/hand_dataset/videos")
PORT = 5003

os.makedirs(OUT_DIR, exist_ok=True)

app = Flask(__name__)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Shared state between camera thread, Flask stream, and terminal input
state = {
    "recording": False,
    "writer": None,
    "rec_start": None,
    "duration": DURATION_SEC_DEFAULT,
    "tag": "",
    "latest_frame": None,
}
lock = threading.Lock()


def camera_loop():
    while True:
        frame = picam2.capture_array()  # BGR888

        with lock:
            display = frame.copy()

            if state["recording"]:
                elapsed = time.time() - state["rec_start"]
                remaining = max(0, state["duration"] - elapsed)

                if state["writer"] is not None:
                    state["writer"].write(frame)

                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC {elapsed:0.1f}s / {state['duration']}s  [{state['tag']}]",
                            (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if elapsed >= state["duration"]:
                    state["writer"].release()
                    state["writer"] = None
                    state["recording"] = False
                    print(f"\n[SAVED] {state['tag']} -> {OUT_DIR}")
            else:
                cv2.putText(display, "LIVE (not recording)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            state["latest_frame"] = display

        time.sleep(1.0 / FPS)


def generate():
    while True:
        with lock:
            frame = state["latest_frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return '''
    <html><head><title>Dataset Recorder</title></head>
    <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
      <h2>Hand Dataset Recorder - Live</h2>
      <img src="/stream" style="max-width:95%;border:2px solid #444;">
    </body></html>
    '''


@app.route('/stream')
def stream():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def flask_thread():
    app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)


def terminal_loop():
    pi_ip = "<pi-ip>"
    print(f"\nLive preview: http://{pi_ip}:{PORT}  (replace with your Pi's actual IP)\n")
    while True:
        input("\nPress Enter to arm a new recording (Ctrl+C to quit)...")
        tag = input("Tag for this clip (e.g. gloved_wires_front_close): ").strip()
        if not tag:
            print("Tag cannot be empty, try again.")
            continue
        dur_str = input(f"Duration in seconds [{DURATION_SEC_DEFAULT}]: ").strip()
        duration = float(dur_str) if dur_str else DURATION_SEC_DEFAULT

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{timestamp}.mp4"
        path = os.path.join(OUT_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

        with lock:
            state["writer"] = writer
            state["tag"] = tag
            state["duration"] = duration
            state["rec_start"] = time.time()
            state["recording"] = True

        print(f"Recording '{tag}' for {duration}s -> watch the browser...")
        time.sleep(duration + 0.5)
        print(f"Done. Clip count: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.mp4')])}")


if __name__ == '__main__':
    t1 = threading.Thread(target=camera_loop, daemon=True)
    t2 = threading.Thread(target=flask_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        terminal_loop()
    except KeyboardInterrupt:
        print("\nExiting.")
