#!/usr/bin/env python3
import os
import sys
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import gpiod
from gpiod.line import Direction, Value
from flask import Flask, Response
from picamera2 import Picamera2

CAP_W, CAP_H = 1280, 960
PREVIEW_W, PREVIEW_H = 640, 480
FPS = 15
PREP_SEC = 5
CLIP_SEC = 20
REST_SEC = 3
OUT_DIR = os.path.expanduser("~/hand_dataset/videos_zone")
PORT = 5003

LED_PINS = (22, 17, 27, 23, 24)
BUZZER_PIN = 18

SHOTS = [
    ("center_open", False, "Open hand in center, move slowly", "영역 가운데에서 손 펴고 천천히 움직이기"),
    ("center_fist_pinch", False, "Fist / pinch / grab in center", "가운데에서 주먹 쥐기, 집기, 잡기"),
    ("edges_trace", False, "Slowly trace all 4 edges", "영역 네 변을 따라 천천히 이동"),
    ("cross_in_out", False, "Cross the boundary in and out", "경계를 여러 번 넘나들기"),
    ("partial_out", False, "Keep part of the hand outside", "손 일부만 영역 밖에 두기"),
    ("rotate_wrist", False, "Rotate wrist to many angles", "손목을 여러 각도로 돌리기"),
    ("near_far", False, "Lift hand up / down (height)", "손을 들었다 내렸다 높이 바꾸기"),
    ("fast_motion", False, "Move fast", "빠르게 움직이기"),
    ("led_on_center", True, "LEDs ON - center, open/fist", "LED 켜짐 - 가운데에서 펴기/주먹"),
    ("led_on_edges", True, "LEDs ON - edges and crossing", "LED 켜짐 - 가장자리와 경계 넘기"),
    ("no_hand", False, "NO HAND - empty table", "손 빼기 - 빈 테이블만"),
]

os.makedirs(OUT_DIR, exist_ok=True)

try:
    gpio = gpiod.Chip("/dev/gpiochip0").request_lines(
        config={LED_PINS + (BUZZER_PIN,): gpiod.LineSettings(direction=Direction.OUTPUT,
                                                              output_value=Value.INACTIVE)},
        consumer="zone_recorder")
except OSError:
    print("GPIO 사용 중이에요. 먼저 인코더를 끄세요: tmux kill-window -t glove:encoder")
    sys.exit(1)


def leds(on):
    v = Value.ACTIVE if on else Value.INACTIVE
    for p in LED_PINS:
        gpio.set_value(p, v)


def beep(duration=0.12, freq=2000):
    half = 0.5 / freq
    end = time.monotonic() + duration
    while time.monotonic() < end:
        gpio.set_value(BUZZER_PIN, Value.ACTIVE)
        time.sleep(half)
        gpio.set_value(BUZZER_PIN, Value.INACTIVE)
        time.sleep(half)


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (CAP_W, CAP_H), "format": "RGB888"}, buffer_count=2, queue=False))
picam2.start()
try:
    picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
except Exception as e:
    print(f"Autofocus not available: {e}")
time.sleep(2)

params = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.01
aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

app = Flask(__name__)
lock = threading.Lock()
state = {"writer": None, "banner": "starting...", "sub": "", "color": (200, 200, 200),
         "frames": 0, "preview": None}


def camera_loop():
    period = 1.0 / FPS
    while True:
        t0 = time.perf_counter()
        frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        n = 0 if ids is None else len(ids)
        for c in corners:
            cv2.polylines(preview, [(c[0] * PREVIEW_W / CAP_W).astype(np.int32)], True, (0, 255, 0), 2)
        with lock:
            if state["writer"] is not None:
                state["writer"].write(frame)
                state["frames"] += 1
                cv2.circle(preview, (18, 22), 8, (0, 0, 255), -1)
            cv2.rectangle(preview, (0, 0), (PREVIEW_W, 62), (0, 0, 0), -1)
            cv2.putText(preview, state["banner"], (34, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state["color"], 2)
            cv2.putText(preview, state["sub"], (34, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            cv2.putText(preview, f"markers: {n}/4", (10, PREVIEW_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if n >= 4 else (0, 165, 255), 2)
            state["preview"] = preview
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def generate():
    while True:
        with lock:
            frame = state["preview"]
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(1.0 / FPS)


@app.route("/")
def index():
    return '''<html><body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
    <h2>Zone Dataset Recorder (auto)</h2><img src="/stream" style="max-width:95%;border:2px solid #444;"></body></html>'''


@app.route("/stream")
def stream():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def set_banner(banner, sub="", color=(200, 200, 200)):
    with lock:
        state.update(banner=banner, sub=sub, color=color)


def record_shot(idx, tag, led, en, ko):
    total = len(SHOTS)
    print(f"\n[{idx + 1}/{total}] {tag}: {ko}")
    leds(led)
    for s in range(PREP_SEC, 0, -1):
        set_banner(f"[{idx + 1}/{total}] GET READY {s}", f"{tag}: {en}", (0, 220, 255))
        print(f"  준비 {s}...")
        beep(0.08)
        time.sleep(0.92)
    path = os.path.join(OUT_DIR, f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CAP_W, CAP_H))
    beep(0.4)
    with lock:
        state.update(writer=writer, frames=0)
    t0 = time.time()
    while (left := CLIP_SEC - (time.time() - t0)) > 0:
        set_banner(f"[{idx + 1}/{total}] REC {left:4.1f}s", f"{tag}: {en}", (0, 0, 255))
        time.sleep(0.1)
    with lock:
        state["writer"] = None
        frames = state["frames"]
    writer.release()
    leds(False)
    beep(0.15); time.sleep(0.1); beep(0.15)
    print(f"  저장: {os.path.basename(path)} ({frames} frames)")


def main():
    start = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False),
                     daemon=True).start()
    print(f"미리보기: http://localhost:{PORT}")
    print("클립 목록:")
    for i, (tag, led, _, ko) in enumerate(SHOTS):
        print(f"  {i + 1:2d}. {tag:18s} {'[LED]' if led else '     '} {ko}")
    set_banner("Press Enter in terminal to start", f"{len(SHOTS) - start} clips, {CLIP_SEC}s each")
    input(f"\nEnter를 누르면 {start + 1}번부터 자동으로 시작해요 (Ctrl+C 중단) ")
    for i in range(start, len(SHOTS)):
        record_shot(i, *SHOTS[i])
        if i + 1 < len(SHOTS):
            nxt = SHOTS[i + 1]
            for s in range(REST_SEC, 0, -1):
                set_banner(f"NEXT: {nxt[0]} in {s}", nxt[2], (200, 200, 200))
                time.sleep(1)
    set_banner("ALL DONE", f"{len(SHOTS)} clips saved", (0, 255, 0))
    for _ in range(3):
        beep(0.1); time.sleep(0.1)
    print(f"\n끝! 저장 위치: {OUT_DIR}")
    time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단. 다시 시작하려면: python3 ~/record_zone_auto.py <번호>")
    finally:
        leds(False)
        gpio.release()
