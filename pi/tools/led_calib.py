#!/usr/bin/env python3
import sys
import threading
import time

import cv2
import gpiod
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, request, redirect
from gpiod.line import Direction, Value
from picamera2 import Picamera2

W, H = 1280, 960
PW, PH = 640, 480
PORT = 5006
LED_PINS = (22, 17, 27, 23, 24)

cfg = {"ev": -2.0, "auto": False, "ygain": 1.0,
       "v_min": 200, "s_min": 90, "area_min": 4, "area_max": 400}

try:
    gpio = gpiod.Chip("/dev/gpiochip0").request_lines(
        config={LED_PINS: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE)},
        consumer="led_calib")
    print("LED 5개 켬")
except OSError:
    print("GPIO 사용 중이에요. 먼저 인코더를 끄세요: tmux kill-window -t glove:encoder")
    sys.exit(1)

MODEL_PATH = "/home/joshuacho/best_zone_mix.onnx"
_o = ort.SessionOptions()
_o.intra_op_num_threads = 2
sess = ort.InferenceSession(MODEL_PATH, sess_options=_o, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
SIZE = inp.shape[2] if isinstance(inp.shape[2], int) else 640
yolo = {"conf": 0.0, "kpts": None, "ms": 0.0}
SK = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
      (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
latest = {"full": None, "roi": None}


def yolo_loop():
    while True:
        full = latest["full"]
        if full is None:
            time.sleep(0.2)
            continue
        t0 = time.perf_counter()
        x0, y0, x1, y1 = latest["roi"] or (0, 0, W, H)
        img = full[y0:y1, x0:x1]
        img = cv2.convertScaleAbs(img, alpha=cfg["ygain"]) if cfg["ygain"] != 1.0 else img
        h, w = img.shape[:2]
        sc = SIZE / max(h, w)
        pad = np.full((SIZE, SIZE, 3), 114, np.uint8)
        pad[:int(h * sc), :int(w * sc)] = cv2.resize(img, (int(w * sc), int(h * sc)))
        out = sess.run(None, {inp.name: (pad.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0][0].T
        i = int(out[:, 4].argmax())
        k = out[i, 5:].reshape(21, 3).copy()
        k[:, :2] /= sc
        k[:, 0] += x0
        k[:, 1] += y0
        yolo.update(conf=float(out[i, 4]), kpts=k * [PW / W, PH / H, 1], ms=(time.perf_counter() - t0) * 1000)
        time.sleep(1.0)


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (W, H), "format": "RGB888"}, buffer_count=4, queue=False))
picam2.start()
time.sleep(1)


def apply_camera():
    picam2.set_controls({"AeEnable": True, "ExposureValue": 0.0 if cfg["auto"] else float(cfg["ev"])})
    try:
        picam2.set_controls({"AfMode": 2})
    except Exception:
        pass


apply_camera()

params = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.01
aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

lock = threading.Lock()
state = {"jpeg": None, "info": ""}


def find_blobs(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(ce[i], st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if cfg["area_min"] <= st[i, cv2.CC_STAT_AREA] <= cfg["area_max"]]


def loop():
    while True:
        full = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        latest["full"] = full
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(full, cv2.COLOR_BGR2GRAY))
        if ids is not None and len(ids) >= 4:
            pts = np.concatenate([c[0] for c in corners]).astype(np.int32)
            bx, by, bw, bh = cv2.boundingRect(pts)
            mx, my = int(bw * 0.2), int(bh * 0.2)
            latest["roi"] = (max(0, bx - mx), max(0, by - my), min(W, bx + bw + mx), min(H, by + bh + my))
        img = cv2.resize(full, (PW, PH), interpolation=cv2.INTER_AREA)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        bright = (v >= cfg["v_min"]) & (s >= cfg["s_min"])
        red = (bright & ((h <= 12) | (h >= 160))).astype(np.uint8)
        blue = (bright & (h >= 95) & (h <= 135)).astype(np.uint8)
        reds, blues = find_blobs(red), find_blobs(blue)

        vis = img.copy()
        vis[red > 0] = (0, 255, 0)
        vis[blue > 0] = (255, 0, 255)
        if ids is not None:
            for c in corners:
                cv2.polylines(vis, [(c[0] * PW / W).astype(np.int32)], True, (0, 255, 255), 2)
        for (x, y), a in reds:
            cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), 2)
        for (x, y), a in blues:
            cv2.circle(vis, (int(x), int(y)), 10, (255, 0, 255), 2)
        n_m = 0 if ids is None else len(ids)
        ok_all = n_m >= 4 and len(reds) == 4 and len(blues) == 1
        info = f"markers {n_m}/4 | red {len(reds)}/4 | blue {len(blues)}/1"
        cv2.rectangle(vis, (0, 0), (PW, 34), (0, 0, 0), -1)
        cv2.putText(vis, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if ok_all else (0, 165, 255), 2)
        if yolo["kpts"] is not None and yolo["conf"] >= 0.35:
            k = yolo["kpts"]
            for a, b in SK:
                cv2.line(vis, (int(k[a][0]), int(k[a][1])), (int(k[b][0]), int(k[b][1])), (255, 0, 255), 2)
        ycol = (0, 255, 0) if yolo["conf"] >= 0.5 else (0, 165, 255) if yolo["conf"] >= 0.35 else (0, 0, 255)
        cv2.putText(vis, f"YOLO conf {yolo['conf']:.2f} ({yolo['ms']:.0f} ms)", (PW - 300, PH - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ycol, 2)
        meta = picam2.capture_metadata()
        cam = f"EV {cfg['ev']} | exp {meta.get('ExposureTime', 0)}us | ygain {cfg['ygain']}"
        cv2.putText(vis, cam, (10, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        both = np.hstack([img, vis])
        ok, buf = cv2.imencode(".jpg", both, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            state["jpeg"] = buf.tobytes() if ok else state["jpeg"]
            state["info"] = f"{info} | {cam}"
        time.sleep(0.05)


app = Flask(__name__)


@app.route("/")
def index():
    rows = "".join(
        f'<label>{k} <input name="{k}" value="{cfg[k]}" size="6"></label> &nbsp; '
        for k in ("ev", "ygain", "v_min", "s_min", "area_min", "area_max"))
    return f'''<html><body style="background:#111;color:#eee;font-family:sans-serif;text-align:center">
    <h3>LED calibration — 목표: 위 글자 초록색 + YOLO conf 0.5 이상(초록)</h3>
    <form action="/set">{rows}
    <label>auto exposure <input type="checkbox" name="auto" {"checked" if cfg["auto"] else ""}></label>
    <button>적용</button></form>
    <p>빠른 노출: {" ".join(f'<a style="color:#6cf" href="/set?ev={e}">EV {e}</a>' for e in (0, -0.5, -1, -1.5, -2, -3))}</p>
    <img src="/stream" style="max-width:98%"><p>왼쪽 원본 | 오른쪽: 초록=빨간 LED, 보라=파란 LED, 노랑=마커</p>
    <p><a style="color:#6cf" href="/values">현재 값 보기</a></p></body></html>'''


@app.route("/set")
def set_values():
    for k in ("ev", "ygain", "v_min", "s_min", "area_min", "area_max"):
        if k in request.args and request.args[k] != "":
            cfg[k] = float(request.args[k]) if k in ("ev", "ygain") else int(float(request.args[k]))
    cfg["auto"] = "auto" in request.args
    apply_camera()
    return redirect("/")


@app.route("/values")
def values():
    with lock:
        return f"<pre>{cfg}\n{state['info']}</pre>"


@app.route("/stream")
def stream():
    def gen():
        while True:
            with lock:
                j = state["jpeg"]
            if j:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=yolo_loop, daemon=True).start()
    print(f"http://localhost:{PORT}  (터널: ssh -N -L {PORT}:localhost:{PORT} joshuacho@<pi-ip>)")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        print("최종 값:", cfg)
        gpio.release()
