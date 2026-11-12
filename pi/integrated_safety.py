#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response, request, jsonify
import threading
import time
import os
import io
import csv
import json
import collections
import rclpy
from std_msgs.msg import Float32, Bool

WIDTH, HEIGHT = 640, 480
CAP_W, CAP_H = 1280, 960
CAP_SCALE = WIDTH / CAP_W
PORT = 5005
JPEG_QUALITY = 60
RENDER_FPS = 15
CAM_FPS = 20
EXPOSURE_EV = -2.0
SETTINGS_PATH = os.path.expanduser("~/glove_ui_settings.json")
ARUCO_EVERY_N = 5

MODEL_CANDIDATES = ["/home/joshuacho/best_zone_dark_416.onnx", "/home/joshuacho/best_zone_dark.onnx"]
MODEL_PATH = next(p for p in MODEL_CANDIDATES if os.path.exists(p))
YOLO_CONF = 0.35
YOLO_KPT_CONF = 0.3
YOLO_GAIN = 1.0
YOLO_STALE_SEC = 2.0
FLOW_LOST_SEC = 0.6
FLOW_MAX_FEAT = 60
FLOW_MIN_FEAT = 8
FLOW_LK = dict(winSize=(21, 21), maxLevel=4,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
ROI_MARGIN = 0.35
ORT_THREADS = 2
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

LED_CORE_V = 240
LED_RING_S = 70
LED_AREA_MIN = 2
LED_AREA_MAX = 3000
LED_RING_PX = 11
TRACK_GATE_PX = 60
HOLD_SEC = 0.35

ARUCO_DICT = cv2.aruco.DICT_4X4_50
NUM_CORNERS = 4
MARKER_SIZE_MM = 70.0
WARN_MM = 50.0
NUM_LEVELS = 5

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
LEVEL_COLORS = [(0, 255, 0), (0, 255, 180), (0, 220, 255), (0, 165, 255), (0, 90, 255), (0, 0, 255)]

def yolo_worker(in_q, out_q, model_path, threads, conf_min):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_path, sess_options=o, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    out_q.put(("ready", size))
    while True:
        img = in_q.get()
        if img is None:
            break
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        pad = np.full((size, size, 3), 114, dtype=np.uint8)
        pad[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = (pad.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        preds = sess.run(None, {inp.name: blob})[0][0].T
        i = int(preds[:, 4].argmax())
        conf = float(preds[i, 4])
        k = None
        if conf >= conf_min:
            k = preds[i, 5:].reshape(21, 3).copy()
            k[:, :2] /= scale
        out_q.put((k, conf, (time.perf_counter() - t0) * 1000))


import multiprocessing as mp
_ctx = mp.get_context("fork")
yolo_in, yolo_out = _ctx.Queue(maxsize=1), _ctx.Queue(maxsize=1)
yolo_proc = _ctx.Process(target=yolo_worker, args=(yolo_in, yolo_out, MODEL_PATH, ORT_THREADS, YOLO_CONF), daemon=True)
yolo_proc.start()
_tag, INPUT_SIZE = yolo_out.get()
print(f"YOLO worker ready: {MODEL_PATH} (input {INPUT_SIZE}, pid {yolo_proc.pid})")

app = Flask(__name__)

rclpy.init()
ros_node = rclpy.create_node('integrated_safety')
haptic_pubs = [ros_node.create_publisher(Float32, f'/haptic_cmd_{i}', 10) for i in range(5)]
heartbeat_pub = ros_node.create_publisher(Bool, '/vision_heartbeat', 10)
gain_pubs = [ros_node.create_publisher(Float32, f'/haptic_gain_{i}', 10) for i in range(5)]
exit_pub = ros_node.create_publisher(Bool, '/zone_exit', 10)
buzzer_pub = ros_node.create_publisher(Bool, '/buzzer_enable', 10)
pause_pub = ros_node.create_publisher(Bool, '/glove_pause', 10)
imu_state = {"tremor": None, "t": 0.0, "sudden": False}


def _on_tremor(msg):
    imu_state.update(tremor=float(msg.data), t=time.monotonic())


def _on_sudden(msg):
    imu_state["sudden"] = bool(msg.data)


ros_node.create_subscription(Float32, '/tremor_rms', _on_tremor, 10)
ros_node.create_subscription(Bool, '/sudden_move_alert', _on_sudden, 10)
threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()

print("Starting camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (CAP_W, CAP_H), "format": "RGB888"}, buffer_count=6))
picam2.start()
try:
    picam2.set_controls({"AeEnable": True, "ExposureValue": EXPOSURE_EV, "AfMode": 2, "AfSpeed": 1,
                         "FrameDurationLimits": (int(1e6 / CAM_FPS), int(1e6 / CAM_FPS))})
    print(f"Exposure EV {EXPOSURE_EV}, autofocus continuous")
except Exception as e:
    print(f"Camera controls failed: {e}")
time.sleep(2)

aruco_params = cv2.aruco.DetectorParameters()
aruco_params.minMarkerPerimeterRate = 0.01
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 45
aruco_params.adaptiveThreshWinSizeStep = 8
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), aruco_params)

frame_cond = threading.Condition()
jpeg_cond = threading.Condition()
result_lock = threading.Lock()
cfg = {"ev": EXPOSURE_EV, "core_v": LED_CORE_V, "enhance": True, "gamma": 1.8, "clahe": 2.0}
shared = {"frame": None, "full": None, "gray_full": None, "kpts": None, "kpts_t": 0.0, "kpts_gray": None,
          "kpts_live": None, "kpts_live_ok": False, "yolo_ms": 0.0, "hand_conf": 0.0, "frame_id": 0, "jpeg": None, "jpeg_id": 0,
          "zone": None, "fingers": [], "blobs": ([], []), "seen": {},
          "status": ("STARTING", (200, 200, 200)), "fps": 0.0}
marker_cache = {}


def update_markers(gray):
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is not None:
        now = time.monotonic()
        for c, i in zip(corners, ids.flatten()):
            marker_cache[int(i)] = (c[0].astype(np.float32) * CAP_SCALE, now)


def corner_markers():
    recent = sorted(marker_cache.items(), key=lambda kv: kv[1][1], reverse=True)[:NUM_CORNERS]
    return {k: v[0] for k, v in recent}


def offset_polygon(poly, d):
    n = len(poly)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sgn = 1.0 if area > 0 else -1.0
    lines = []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        e = q - p
        nrm = sgn * np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-9)
        lines.append((p + nrm * d, e))
    out = []
    for i in range(n):
        (p1, e1), (p2, e2) = lines[i - 1], lines[i]
        A = np.array([e1, -e2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            out.append(p2)
            continue
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * e1)
    return np.array(out, dtype=np.float32)


def build_zone():
    markers = corner_markers()
    if len(markers) < NUM_CORNERS:
        return None
    centers = np.array([m.mean(axis=0) for m in markers.values()])
    cen = centers.mean(axis=0)
    inner = np.array([mc[np.argmin(np.linalg.norm(mc - cen, axis=1))] for mc in markers.values()],
                     dtype=np.float32)
    ang = np.arctan2(inner[:, 1] - cen[1], inner[:, 0] - cen[0])
    poly = inner[np.argsort(ang)]
    px_per_mm = np.mean([np.mean([np.linalg.norm(m[k] - m[(k + 1) % 4]) for k in range(4)])
                         for m in markers.values()]) / MARKER_SIZE_MM
    return poly, offset_polygon(poly, WARN_MM * px_per_mm), px_per_mm


def boundary_level(d_mm, warn=WARN_MM):
    if d_mm < 0:
        return NUM_LEVELS, True
    if d_mm >= warn:
        return 0, False
    step = warn / NUM_LEVELS
    return min(NUM_LEVELS, int((warn - d_mm) / step) + 1), False


def hyst_level(d_mm, prev, warn, margin):
    lv, out = boundary_level(d_mm, warn)
    if out or prev >= NUM_LEVELS:
        return lv, out
    hi, _ = boundary_level(d_mm - margin, warn)
    lo, _ = boundary_level(d_mm + margin, warn)
    if lo <= prev <= hi:
        return prev, False
    return lv, False


def detect_leds(frame):
    h, sat, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    core = (v >= cfg["core_v"]).astype(np.uint8)
    for mc in corner_markers().values():
        cv2.fillConvexPoly(core, cv2.convexHull(mc.astype(np.int32)), 0)
    core = cv2.dilate(core, np.ones((3, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(core, connectivity=8)
    thumbs, fingers = [], []
    k = np.ones((LED_RING_PX, LED_RING_PX), np.uint8)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if not (LED_AREA_MIN <= area <= LED_AREA_MAX):
            continue
        x, y, w, hh = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        pad = LED_RING_PX
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + hh + pad)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, k) - comp
        sel = (ring > 0) & (sat[y0:y1, x0:x1] >= LED_RING_S) & (v[y0:y1, x0:x1] >= 60)
        if sel.sum() < 5:
            continue
        hue = h[y0:y1, x0:x1][sel]
        red = np.mean((hue <= 12) | (hue >= 160))
        blue = np.mean((hue >= 95) & (hue <= 135))
        c = (float(ce[i][0]), float(ce[i][1]))
        if blue > 0.5 and blue > red:
            thumbs.append((area, c))
        elif red > 0.5:
            fingers.append((area, c))
    return [c for _, c in sorted(thumbs, reverse=True)], [c for _, c in sorted(fingers, reverse=True)]


def chain_order(thumb, reds):
    order, cur, left = [], thumb, list(reds)
    while left and len(order) < 4:
        j = int(np.argmin([np.hypot(cur[0] - r[0], cur[1] - r[1]) for r in left]))
        order.append(left.pop(j))
        cur = order[-1]
    return order + [None] * (4 - len(order))


def match_tracks(prev, cands, gate):
    out = [None] * len(prev)
    pairs = sorted((np.hypot(p[0] - c[0], p[1] - c[1]), i, j)
                   for i, p in enumerate(prev) if p is not None
                   for j, c in enumerate(cands))
    used = set()
    for d, i, j in pairs:
        if d <= gate and out[i] is None and j not in used:
            out[i] = cands[j]
            used.add(j)
    return out


def seed_features(gray, k):
    vis = k[k[:, 2] >= YOLO_KPT_CONF][:, :2]
    if len(vis) < 3:
        return None
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(vis.astype(np.int32)), 255)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    f = cv2.goodFeaturesToTrack(gray, FLOW_MAX_FEAT, 0.01, 5, mask=mask)
    return f if f is not None and len(f) >= FLOW_MIN_FEAT else None


def propagate(g0, g1, k, feat):
    if feat is None or len(feat) < 4:
        return k, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, feat, None, **FLOW_LK)
    good = st.reshape(-1) == 1
    if good.sum() < 4:
        return k, None
    a, b = feat[good].reshape(-1, 2), p1[good].reshape(-1, 2)
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        M[:, 2] = np.median(b - a, axis=0)
        keep = np.ones(len(b), bool)
    else:
        keep = inl.reshape(-1).astype(bool)
    k = k.copy()
    k[:, :2] = k[:, :2] @ M[:, :2].T + M[:, 2]
    nxt = b[keep].reshape(-1, 1, 2).astype(np.float32)
    return k, (nxt if len(nxt) >= 4 else None)


def yolo_loop():
    last_id = 0
    while True:
        with frame_cond:
            frame_cond.wait_for(lambda: shared["frame_id"] != last_id)
            full, small, last_id = shared["full"], shared["frame"], shared["frame_id"]
        try:
            gray_c = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            zone = build_zone()
            x0, y0, x1, y1 = 0, 0, CAP_W, CAP_H
            if zone is not None:
                bx, by, bw, bh = cv2.boundingRect((zone[0] / CAP_SCALE).astype(np.int32))
                mx, my = int(bw * ROI_MARGIN), int(bh * ROI_MARGIN)
                x0, y0 = max(0, bx - mx), max(0, by - my)
                x1, y1 = min(CAP_W, bx + bw + mx), min(CAP_H, by + bh + my)
            crop = cv2.convertScaleAbs(full[y0:y1, x0:x1], alpha=YOLO_GAIN)
            yolo_in.put(crop[:, :, ::-1].copy())
            k, conf, ms = yolo_out.get()
            if k is not None:
                k[:, 0] = (k[:, 0] + x0) * CAP_SCALE
                k[:, 1] = (k[:, 1] + y0) * CAP_SCALE
            with result_lock:
                shared.update(hand_conf=conf, yolo_ms=ms)
                if k is not None:
                    shared.update(kpts=k, kpts_t=time.monotonic(), kpts_gray=gray_c)
        except Exception as e:
            print(f"YOLO error: {e}")
            time.sleep(0.1)


def capture_loop():
    while True:
        full = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(full, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        gray_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        with frame_cond:
            shared.update(frame=frame, full=full, gray_full=gray_full, frame_t=time.monotonic())
            shared["frame_id"] += 1
            frame_cond.notify_all()


LOST_EXIT_LEVEL = 3
LOST_EXIT_SEC = 3.0
JITTER_WINDOW_SEC = 1.5
FATIGUE_HZ = 5.0
CALIB_SEC = 30.0
CALIB_MIN_SAMPLES = 40
TREMOR_FLOOR_DPS = 1.5
JITTER_FLOOR_MM = 1.0
REST_RATIO = 3.0
REST_CLEAR_RATIO = 2.0
GAUGE_MAX_RATIO = 4.0
FATIGUE_EMA = 0.05
EXIT_DEBOUNCE_SEC = 1.0
REST_SUSTAIN_SEC = 20.0
REST_CLEAR_SEC = 10.0
SNOOZE_SEC = 300.0
IMU_STALE_SEC = 1.0
CAMERA_STALE_SEC = 1.0
LOG_MAX = 20000
NOGO_WARN_MM = 30.0
TEST_PULSE_SEC = 0.45

SKELETON = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))
STATUS_TEXT = {"nogo": "NO-GO AREA", "paused": "GLOVE PAUSED", "no_zone": "SET UP ZONE", "no_hand": "PLACE HAND IN VIEW",
               "safe": "SAFE", "caution": "CAUTION - NEAR EDGE", "outside": "OUTSIDE ZONE",
               "lost": "TRACKING LOST"}
DOC_ZONE_COLORS = {"safe": (90, 200, 60), "caution": (0, 190, 255), "outside": (40, 40, 230),
                   "nogo": (40, 40, 230)}
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def load_settings():
    try:
        with open(SETTINGS_PATH) as f:
            d = json.load(f)
        return {"rotate": int(d.get("rotate", 0)) % 360, "mirror": bool(d.get("mirror", False)),
                "nogo": [[[float(u), float(v)] for u, v in q] for q in d.get("nogo", []) if len(q) >= 3]}
    except Exception:
        return {"rotate": 0, "mirror": False, "nogo": []}


def save_settings():
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"settings save failed: {e}")


def default_ctrl():
    return {"paused": False, "finger_on": [True] * 5, "gain": [1.0] * 5, "buzzer_on": True}


ctrl_lock = threading.Lock()
ctrl = default_ctrl()
settings = load_settings()
zone_state = {"locked": False}
session = {"start": time.time(), "exits": 0, "nogo_hits": 0, "setup_done": False}
test_state = {"finger": None, "buzz": False}
event_log = collections.deque(maxlen=LOG_MAX)
sample_log = collections.deque(maxlen=LOG_MAX)
out_state = {"levels": [None] * 5, "exit": None, "t": 0.0}


def log_event(kind, detail=""):
    event_log.append((time.time(), kind, detail))
    print(f"[event] {kind} {detail}")


def push_ctrl():
    with ctrl_lock:
        c = {k: (list(v) if isinstance(v, list) else v) for k, v in ctrl.items()}
    for i in range(5):
        gain_pubs[i].publish(Float32(data=float(c["gain"][i] if c["finger_on"][i] else 0.0)))
    buzzer_pub.publish(Bool(data=bool(c["buzzer_on"] and exp_feedback()[1])))
    pause_pub.publish(Bool(data=c["paused"]))
    out_state["t"] = 0.0


def send_outputs(levels, exited):
    now = time.monotonic()
    heartbeat_pub.publish(Bool(data=True))
    periodic = now - out_state["t"] > 1.0
    if not exp_feedback()[0]:
        levels = [0] * len(levels)
    if test_state["finger"] is not None:
        levels = [NUM_LEVELS if j == test_state["finger"] else 0 for j in range(5)]
        exited = False
    if test_state["buzz"]:
        levels, exited = [0] * 5, True
    with ctrl_lock:
        paused, on = ctrl["paused"], list(ctrl["finger_on"])
    for i, lv in enumerate(levels):
        eff = 0 if paused or not on[i] else lv
        if periodic or eff != out_state["levels"][i]:
            haptic_pubs[i].publish(Float32(data=eff / NUM_LEVELS))
            out_state["levels"][i] = eff
    ex = bool(exited and not paused)
    if periodic or ex != out_state["exit"]:
        exit_pub.publish(Bool(data=ex))
        out_state["exit"] = ex
    if periodic:
        out_state["t"] = now
        push_ctrl()
        out_state["t"] = now


class Fatigue:
    def __init__(self):
        self.reset()

    def reset(self):
        now = time.monotonic()
        self.calib_until = now + CALIB_SEC
        self.samples_t, self.samples_j = [], []
        self.base_t = self.base_j = None
        self.ema = 1.0
        self.high_since = self.low_since = None
        self.warn = False
        self.snooze_until = 0.0
        self.last = 0.0

    def calibrating(self, now):
        return self.base_t is None and self.base_j is None

    def update(self, now, tremor, jitter, hand):
        if now - self.last < 1.0 / FATIGUE_HZ:
            return
        self.last = now
        if self.calibrating(now):
            if hand:
                if tremor is not None:
                    self.samples_t.append(tremor)
                if jitter is not None:
                    self.samples_j.append(jitter)
            enough = max(len(self.samples_t), len(self.samples_j)) >= CALIB_MIN_SAMPLES
            if now >= self.calib_until and enough:
                if len(self.samples_t) >= CALIB_MIN_SAMPLES // 2:
                    self.base_t = max(TREMOR_FLOOR_DPS, float(np.median(self.samples_t)))
                if len(self.samples_j) >= CALIB_MIN_SAMPLES // 2:
                    self.base_j = max(JITTER_FLOOR_MM, float(np.median(self.samples_j)))
                if self.base_t is None and self.base_j is None:
                    self.calib_until = now + CALIB_SEC
                else:
                    log_event("baseline", f"tremor={self.base_t} jitter={self.base_j}")
            return
        if not hand:
            return
        ratios = []
        if tremor is not None and self.base_t:
            ratios.append(tremor / self.base_t)
        if jitter is not None and self.base_j:
            ratios.append(jitter / self.base_j)
        if not ratios:
            return
        self.ema = (1 - FATIGUE_EMA) * self.ema + FATIGUE_EMA * max(ratios)
        if self.ema > REST_RATIO:
            self.low_since = None
            self.high_since = self.high_since or now
            if (not self.warn and now - self.high_since > REST_SUSTAIN_SEC and now > self.snooze_until):
                self.warn = True
                log_event("rest_warning", f"ratio={self.ema:.2f}")
        elif self.ema < REST_CLEAR_RATIO:
            self.high_since = None
            self.low_since = self.low_since or now
            if self.warn and now - self.low_since > REST_CLEAR_SEC:
                self.warn = False
                log_event("rest_cleared", f"ratio={self.ema:.2f}")

    def snooze(self):
        self.warn = False
        self.high_since = None
        self.snooze_until = time.monotonic() + SNOOZE_SEC
        log_event("rest_dismissed")

    def summary(self):
        now = time.monotonic()
        cal = self.calibrating(now)
        return {"level": float(np.clip((self.ema - 1.0) / (GAUGE_MAX_RATIO - 1.0), 0.0, 1.0)), "ratio": round(self.ema, 2),
                "calibrating": cal, "calib_left": max(0, int(self.calib_until - now)) if cal else 0,
                "warn": self.warn, "snoozed": now < self.snooze_until,
                "base_tremor": self.base_t, "base_jitter": self.base_j}


fatigue = Fatigue()


def jitter_mm(hist, px_per_mm, now):
    vals = []
    for h in hist:
        while h and now - h[0][0] > JITTER_WINDOW_SEC:
            h.popleft()
        if len(h) < 10:
            continue
        a = np.array(h, dtype=np.float64)
        t = a[:, 0] - a[0, 0]
        res = []
        for c in (1, 2):
            fit = np.polyval(np.polyfit(t, a[:, c], 2), t)
            res.append(a[:, c] - fit)
        vals.append(float(np.sqrt(np.mean(res[0] ** 2 + res[1] ** 2))) / px_per_mm)
    return float(np.mean(vals)) if vals else None


def zone_homography(zone):
    return cv2.getPerspectiveTransform(UNIT_SQUARE, zone[0].astype(np.float32))


def nogo_polys(zone):
    if zone is None or not settings["nogo"]:
        return []
    H = zone_homography(zone)
    return [cv2.perspectiveTransform(np.array(q, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
            for q in settings["nogo"]]


def nogo_eval(p, polys, px_per_mm):
    best, inside = 0, False
    for q in polys:
        d = -cv2.pointPolygonTest(q, (float(p[0]), float(p[1])), True) / px_per_mm
        if d <= 0:
            return NUM_LEVELS, True
        if d < NOGO_WARN_MM:
            best = max(best, min(NUM_LEVELS, int((NOGO_WARN_MM - d) / (NOGO_WARN_MM / NUM_LEVELS)) + 1))
    return best, inside


def unorient(x, y):
    W, H = WIDTH, HEIGHT
    r = settings["rotate"]
    wd = H if r in (90, 270) else W
    if settings["mirror"]:
        x = wd - 1 - x
    if r == 90:
        return y, H - 1 - x
    if r == 180:
        return W - 1 - x, H - 1 - y
    if r == 270:
        return W - 1 - y, x
    return x, y


def run_test(kind):
    if kind == "buzzer":
        test_state["buzz"] = True
        time.sleep(0.6)
        test_state["buzz"] = False
        return
    for i in range(5):
        test_state["finger"] = i
        time.sleep(TEST_PULSE_SEC)
    test_state["finger"] = None


def track_loop():
    last_id = 0
    tips = [None] * 5
    tips_t = [0.0] * 5
    t_prev, fps = time.perf_counter(), 0.0
    n = 0
    trk, feat, prev_gray, src_t, flow_t = None, None, None, 0.0, 0.0
    last_lv, lost_since = [0] * 5, [None] * 5
    hist = [collections.deque(maxlen=60) for _ in range(5)]
    prev_exit, last_sample, prev_code, exit_clear_t = False, 0.0, None, 0.0
    prev_nogo = False
    x_prev_lv = 0
    while True:
        with frame_cond:
            frame_cond.wait_for(lambda: shared["frame_id"] != last_id)
            frame, gray_full, last_id = shared["frame"], shared["gray_full"], shared["frame_id"]
        try:
            if not zone_state["locked"] and (n % ARUCO_EVERY_N == 0 or len(marker_cache) < NUM_CORNERS):
                update_markers(gray_full)
            n += 1
            zone = build_zone()
            thumbs, reds = detect_leds(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.monotonic()

            new = [None] * 5
            if thumbs:
                if tips[0] is not None and len(thumbs) > 1:
                    new[0] = match_tracks([tips[0]], thumbs, TRACK_GATE_PX * 2)[0] or thumbs[0]
                else:
                    new[0] = thumbs[0]
            if new[0] is not None and len(reds) == 4:
                new[1:] = chain_order(new[0], reds)
            else:
                new[1:] = match_tracks(tips[1:], reds, TRACK_GATE_PX)
                if new[0] is not None and all(t is None for t in tips[1:]) and reds:
                    new[1:] = chain_order(new[0], reds)

            for i in range(5):
                if new[i] is not None:
                    tips[i], tips_t[i] = new[i], now
                    hist[i].append((now, new[i][0], new[i][1]))
                elif tips[i] is not None and now - tips_t[i] > HOLD_SEC:
                    tips[i] = None

            with result_lock:
                y_k, kpts_t, y_gray = shared["kpts"], shared["kpts_t"], shared["kpts_gray"]
            if y_k is not None and kpts_t != src_t:
                src_t = kpts_t
                trk, feat = propagate(y_gray, gray, y_k, seed_features(y_gray, y_k))
                if feat is not None:
                    flow_t = now
                else:
                    trk = y_k.copy()
            elif trk is not None and prev_gray is not None:
                trk, feat = propagate(prev_gray, gray, trk, feat)
                if feat is not None:
                    flow_t = now
                if feat is None or len(feat) < FLOW_MIN_FEAT:
                    feat = seed_features(gray, trk)
            prev_gray = gray
            kpts = trk
            yolo_fresh = (kpts is not None and now - kpts_t < YOLO_STALE_SEC
                          and (now - flow_t < 0.2 or now - kpts_t < FLOW_LOST_SEC))

            pts, src = list(tips), ["L" if t is not None else None for t in tips]
            if yolo_fresh:
                for i, f in enumerate(FINGERTIP_INDICES):
                    if pts[i] is None and kpts[f][2] >= YOLO_KPT_CONF:
                        pts[i], src[i] = (float(kpts[f][0]), float(kpts[f][1])), "Y"

            levels, exited, fingers, hand_out, lost_exit = [0] * 5, False, [], False, False
            dists = [None] * 5
            xm = exp["active"]
            seen = [p is not None for p in pts]
            jit, nogo_hit = None, False
            ng = nogo_polys(zone)
            if zone is not None:
                poly, _, px_per_mm = zone
                for i, p in enumerate(pts):
                    if p is None:
                        if xm:
                            if i == EXP_TRACE_FINGER:
                                x_prev_lv = 0
                            continue
                        if last_lv[i] >= LOST_EXIT_LEVEL:
                            lost_since[i] = lost_since[i] or now
                            if now - lost_since[i] < LOST_EXIT_SEC:
                                levels[i] = NUM_LEVELS
                                lost_exit = True
                            else:
                                last_lv[i] = 0
                        continue
                    lost_since[i] = None
                    d_mm = cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True) / px_per_mm
                    dists[i] = d_mm
                    if xm:
                        if i != EXP_TRACE_FINGER:
                            fingers.append((i, (int(p[0]), int(p[1])), 0, False, src[i] == "L"))
                            continue
                        lv, out = hyst_level(d_mm, x_prev_lv, EXP_WARN_MM, EXP_HYST_MM)
                        x_prev_lv = lv
                    else:
                        lv, out = boundary_level(d_mm)
                    nlv, nin = nogo_eval(p, ng, px_per_mm)
                    levels[i], last_lv[i] = max(lv, nlv), lv
                    exited = exited or out
                    nogo_hit = nogo_hit or nin
                    fingers.append((i, (int(p[0]), int(p[1])), max(lv, nlv), out or nin, src[i] == "L"))
                if yolo_fresh and not xm:
                    vis = kpts[1:][kpts[1:, 2] >= YOLO_KPT_CONF]
                    hand_out = any(cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0 for x, y, _ in vis)
                    exited = exited or hand_out
                    if ng and not nogo_hit:
                        nogo_hit = any(cv2.pointPolygonTest(q, (float(x), float(y)), False) >= 0
                                       for q in ng for x, y, _ in vis)
                exited = exited or lost_exit
                jit = jitter_mm(hist, px_per_mm, now)

            hand = bool(fingers) or yolo_fresh
            with ctrl_lock:
                paused = ctrl["paused"]
            if paused:
                code = "paused"
            elif zone is None:
                code = "no_zone"
            elif nogo_hit:
                code = "nogo"
            elif exited:
                code = "outside"
            elif not hand:
                code = "no_hand"
            elif max(levels) > 0:
                code = "caution"
            else:
                code = "safe"
            send_outputs(levels if zone is not None else [0] * 5, exited or nogo_hit)
            if nogo_hit and not prev_nogo and not paused:
                session["nogo_hits"] += 1
                log_event("nogo_entry")
            prev_nogo = nogo_hit

            if not exited and prev_exit:
                exit_clear_t = now
            if exited and not prev_exit and not paused and now - exit_clear_t > EXIT_DEBOUNCE_SEC:
                session["exits"] += 1
                log_event("zone_exit", "lost near edge" if lost_exit and not hand_out else
                          ("yolo hand" if hand_out else "fingertip"))
            prev_exit = exited
            if code != prev_code and code in ("no_zone", "paused"):
                log_event("status", code)
            prev_code = code

            tremor = imu_state["tremor"] if now - imu_state["t"] < IMU_STALE_SEC else None
            fatigue.update(now, tremor, jit, hand and zone is not None)
            exp_tick(now, exited, pts[EXP_TRACE_FINGER], dists[EXP_TRACE_FINGER], levels[EXP_TRACE_FINGER], zone, tremor,
                     src[EXP_TRACE_FINGER])
            if now - last_sample >= 1.0:
                last_sample = now
                sample_log.append((time.time(), code, max(levels), tremor, jit, fatigue.ema))

            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-3)
            t_prev = t
            with result_lock:
                shared.update(zone=zone, fingers=fingers, blobs=(thumbs, reds), code=code, levels=levels,
                              seen_f=seen, fps=fps, kpts_live=kpts, kpts_live_ok=yolo_fresh,
                              jitter=jit, hand_out=hand_out, nogo=ng,
                              seen={k: v.copy() for k, v in corner_markers().items()})
        except Exception as e:
            print(f"Track error: {e}")
            time.sleep(0.05)


_awb = {"g": np.ones(3), "n": 0, "key": None, "lut": None}
_clahe = {"clip": None, "obj": None}


def enhance(bgr):
    if _awb["n"] % 15 == 0:
        m = bgr[::8, ::8].reshape(-1, 3).mean(axis=0) + 1e-3
        g = np.clip(m.mean() / m, 0.5, 2.5)
        _awb["g"] = 0.8 * _awb["g"] + 0.2 * g
    _awb["n"] += 1
    key = tuple(np.round(_awb["g"], 2)) + (round(cfg["gamma"], 2),)
    if key != _awb["key"]:
        x = np.arange(256, dtype=np.float32) / 255.0
        lut = np.stack([np.clip(x * gc, 0, 1) ** (1.0 / cfg["gamma"]) for gc in _awb["g"]], axis=-1)
        _awb["lut"], _awb["key"] = (lut * 255).astype(np.uint8).reshape(1, 256, 3), key
    out = cv2.LUT(bgr, _awb["lut"])
    if cfg["clahe"] > 0:
        if _clahe["clip"] != cfg["clahe"]:
            _clahe["obj"], _clahe["clip"] = cv2.createCLAHE(clipLimit=cfg["clahe"], tileGridSize=(8, 8)), cfg["clahe"]
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        ycc[:, :, 0] = _clahe["obj"].apply(ycc[:, :, 0])
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return cv2.GaussianBlur(out, (3, 3), 0)


def orient_pt(x, y):
    W, H = WIDTH, HEIGHT
    r = settings["rotate"]
    if r == 90:
        x, y = H - 1 - y, x
    elif r == 180:
        x, y = W - 1 - x, H - 1 - y
    elif r == 270:
        x, y = y, W - 1 - x
    if settings["mirror"]:
        x = (H if r in (90, 270) else W) - 1 - x
    return int(x), int(y)


def orient(img):
    r = settings["rotate"]
    if r == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif r == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if settings["mirror"]:
        img = cv2.flip(img, 1)
    return img


def draw_nogo(img, polys, hit):
    if not polys:
        return
    over = img.copy()
    for q in polys:
        cv2.fillPoly(over, [q.astype(np.int32)], (40, 40, 220))
    cv2.addWeighted(over, 0.45 if hit else 0.28, img, 0.55 if hit else 0.72, 0, dst=img)
    for q in polys:
        pts = q.astype(np.int32)
        cv2.polylines(img, [pts], True, (60, 60, 255), 2, cv2.LINE_AA)
        c = pts.mean(axis=0).astype(int)


def render_doctor(frame, snap):
    disp = np.ascontiguousarray(frame[:, :, ::-1])
    if cfg["enhance"]:
        disp = enhance(disp)
    zone, code = snap["zone"], snap["code"]
    if zone is not None:
        poly = zone[0].astype(np.int32)
        col = DOC_ZONE_COLORS.get(code, (180, 180, 180))
        over = disp.copy()
        cv2.fillPoly(over, [poly], col)
        cv2.addWeighted(over, 0.16, disp, 0.84, 0, dst=disp)
        cv2.polylines(disp, [poly], True, col, 3, cv2.LINE_AA)
    draw_nogo(disp, snap.get("nogo") or [], snap["code"] == "nogo")
    with ctrl_lock:
        on = list(ctrl["finger_on"])
    for i, pt, lv, out, live in snap["fingers"]:
        if lv >= 1 and on[i]:
            cv2.circle(disp, pt, 10, LEVEL_COLORS[lv], -1, cv2.LINE_AA)
            cv2.circle(disp, pt, 10, (255, 255, 255), 2, cv2.LINE_AA)
    disp = orient(disp)
    for q in snap.get("nogo") or []:
        cx, cy = orient_pt(*q.mean(axis=0))
        cv2.putText(disp, "NO-GO", (cx - 26, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return disp


def render_engineer(frame, snap):
    frame = frame.copy()
    kpts, live_ok = snap["kpts_live"], snap["kpts_live_ok"]
    for mc in snap["seen"].values():
        cv2.polylines(frame, [mc.astype(np.int32)], True, (0, 255, 0), 2)
    if snap["zone"] is not None:
        poly, warn_poly, _ = snap["zone"]
        cv2.polylines(frame, [poly.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.polylines(frame, [warn_poly.astype(np.int32)], True, (0, 220, 255), 1)
    for q in snap.get("nogo") or []:
        cv2.polylines(frame, [q.astype(np.int32)], True, (255, 0, 255), 2)
    if kpts is not None and live_ok:
        for a, b in SKELETON:
            if kpts[a][2] >= YOLO_KPT_CONF and kpts[b][2] >= YOLO_KPT_CONF:
                cv2.line(frame, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])),
                         (160, 0, 160), 1)
    thumbs, reds = snap["blobs"]
    for b in thumbs + reds:
        cv2.circle(frame, (int(b[0]), int(b[1])), 12, (255, 255, 255), 1)
    for i, pt, lv, out, live in snap["fingers"]:
        cv2.circle(frame, pt, 8, LEVEL_COLORS[lv], -1 if live else 2)
    frame = orient(frame)
    for i, pt, lv, out, live in snap["fingers"]:
        x, y = orient_pt(*pt)
        cv2.putText(frame, FINGER_NAMES[i][0], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
    cv2.putText(frame, f"{snap['code']} | track {snap['fps']:.0f} fps | yolo {snap['yolo_ms']:.0f} ms "
                       f"({snap['hand_conf']:.2f})", (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return frame


viewers = {"doc": 0, "eng": 0}
jpegs = {"doc": (None, 0), "eng": (None, 0)}
RENDERERS = {"doc": render_doctor, "eng": render_engineer}


def render_loop():
    last_id = 0
    while True:
        with frame_cond:
            frame_cond.wait_for(lambda: shared["frame_id"] != last_id)
            frame, last_id = shared["frame"], shared["frame_id"]
        t0 = time.perf_counter()
        with result_lock:
            snap = {k: shared.get(k) for k in ("zone", "fingers", "blobs", "code", "fps", "yolo_ms", "hand_conf",
                                               "kpts_live", "kpts_live_ok", "seen", "nogo")}
        for mode, render in RENDERERS.items():
            if viewers[mode] <= 0:
                continue
            try:
                ok, buf = cv2.imencode('.jpg', render(frame, snap), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with jpeg_cond:
                        jpegs[mode] = (buf.tobytes(), jpegs[mode][1] + 1)
                        jpeg_cond.notify_all()
            except Exception as e:
                print(f"Render error ({mode}): {e}")
        shared["render_ms"] = (time.perf_counter() - t0) * 1000
        time.sleep(max(0.0, 1.0 / RENDER_FPS - (time.perf_counter() - t0)))


def generate(mode):
    viewers[mode] += 1
    try:
        last_id = jpegs[mode][1]
        while True:
            with jpeg_cond:
                ok = jpeg_cond.wait_for(lambda: jpegs[mode][1] != last_id, timeout=5.0)
                jpeg, last_id = jpegs[mode]
            if ok and jpeg is not None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
    finally:
        viewers[mode] -= 1


def state_dict():
    now = time.monotonic()
    with result_lock:
        code = shared.get("code") or "no_zone"
        levels = list(shared.get("levels") or [0] * 5)
        seen_f = list(shared.get("seen_f") or [False] * 5)
        extra = {k: shared.get(k) for k in ("fps", "yolo_ms", "hand_conf", "jitter", "hand_out", "render_ms")}
    camera_ok = now - shared.get("frame_t", 0.0) < CAMERA_STALE_SEC
    if not camera_ok:
        code = "lost"
    with ctrl_lock:
        c = {"paused": ctrl["paused"], "finger_on": list(ctrl["finger_on"]), "gain": list(ctrl["gain"]),
             "buzzer_on": ctrl["buzzer_on"]}
    tremor = imu_state["tremor"] if now - imu_state["t"] < IMU_STALE_SEC else None
    return {
        "code": code, "text": STATUS_TEXT.get(code, code), "levels": levels, "max_level": max(levels),
        "seen": seen_f, **c, "sudden": imu_state["sudden"],
        "zone_ok": shared.get("zone") is not None, "zone_locked": zone_state["locked"],
        "markers": len(marker_cache), "camera_ok": camera_ok,
        "imu_ok": tremor is not None, "tremor": tremor, "fatigue": fatigue.summary(),
        "session_sec": int(time.time() - session["start"]), "exits": session["exits"],
        "nogo_hits": session["nogo_hits"], "nogo_count": len(settings["nogo"]),
        "setup_done": session["setup_done"],
        "exp": EXP_CONDS[exp_condition()]["name"] if exp_condition() else None, "testing": test_state["finger"] is not None or test_state["buzz"],
        "events": [{"t": time.strftime("%H:%M:%S", time.localtime(t)), "kind": k, "detail": d}
                   for t, k, d in list(event_log)[-8:]][::-1],
        "rotate": settings["rotate"], "mirror": settings["mirror"],
        "cfg": dict(cfg), "num_levels": NUM_LEVELS, **{k: (round(v, 3) if isinstance(v, float) else v)
                                                        for k, v in extra.items()},
    }


def _flag(v):
    return str(v).lower() in ("1", "true", "on", "yes")


@app.route('/ctrl', methods=['GET', 'POST'])
def ctrl_route():
    a = request.values
    changed = False
    with ctrl_lock:
        if "pause" in a:
            ctrl["paused"] = _flag(a["pause"])
            changed = True
        if "finger" in a:
            i = int(a["finger"])
            if 0 <= i < 5:
                if "on" in a:
                    ctrl["finger_on"][i] = _flag(a["on"])
                if "gain" in a:
                    ctrl["gain"][i] = float(np.clip(float(a["gain"]), 0.2, 1.0))
                changed = True
        if "buzzer" in a:
            ctrl["buzzer_on"] = _flag(a["buzzer"])
            changed = True
        if a.get("session") == "new":
            ctrl.clear()
            ctrl.update(default_ctrl())
            changed = True
    if "pause" in a:
        log_event("paused" if _flag(a["pause"]) else "resumed")
    if "finger" in a:
        log_event("finger_setting", f"{FINGER_NAMES[int(a['finger'])]} on={a.get('on', '-')} gain={a.get('gain', '-')}")
    if "buzzer" in a:
        log_event("buzzer", "on" if _flag(a["buzzer"]) else "off")
    if a.get("session") == "new":
        event_log.clear()
        sample_log.clear()
        session.update(start=time.time(), exits=0, nogo_hits=0, setup_done=False)
        fatigue.reset()
        zone_state["locked"] = False
        log_event("session_start")
    zone = a.get("zone")
    if zone == "lock":
        zone_state["locked"] = True
        log_event("zone", "locked")
    elif zone == "unlock":
        zone_state["locked"] = False
        log_event("zone", "unlocked")
    elif zone == "redetect":
        marker_cache.clear()
        zone_state["locked"] = False
        log_event("zone", "re-detect")
    if "rotate" in a:
        settings["rotate"] = int(a["rotate"]) % 360
        save_settings()
    if "mirror" in a:
        settings["mirror"] = _flag(a["mirror"])
        save_settings()
    if "test" in a and test_state["finger"] is None and not test_state["buzz"]:
        threading.Thread(target=run_test, args=(a["test"],), daemon=True).start()
    if "setup" in a:
        session["setup_done"] = a["setup"] == "done"
        log_event("setup", a["setup"])
    if "calibrate" in a:
        fatigue.reset()
        log_event("baseline", "recalibrating")
    if "dismiss" in a:
        fatigue.snooze()
    if "ev" in a:
        cfg["ev"] = float(np.clip(float(a["ev"]), -8.0, 8.0))
        try:
            picam2.set_controls({"ExposureValue": cfg["ev"]})
        except Exception as e:
            print(f"EV set failed: {e}")
    if "core_v" in a:
        cfg["core_v"] = int(np.clip(float(a["core_v"]), 100, 255))
    if "enhance" in a:
        cfg["enhance"] = _flag(a["enhance"])
    if "gamma" in a:
        cfg["gamma"] = float(np.clip(float(a["gamma"]), 1.0, 3.0))
    if "clahe" in a:
        cfg["clahe"] = float(np.clip(float(a["clahe"]), 0.0, 4.0))
    if changed:
        push_ctrl()
    return jsonify(state_dict())


@app.route('/state')
def state_route():
    return jsonify(state_dict())


EXP_CONDS = {1: {"name": "Glove only", "vib": False, "buzz": False},
             2: {"name": "Glove + vibration", "vib": True, "buzz": False},
             3: {"name": "Glove + vibration + sound", "vib": True, "buzz": True}}
EXP_ORDERS = ([1, 2, 3], [2, 3, 1], [3, 1, 2])
EXP_REPS = 3
EXP_COUNTDOWN_SEC = 3.0
EXP_TIMEOUT_SEC = 120.0
EXP_TRACE_FINGER = 1
EXP_WARN_MM = 20.0
EXP_HYST_MM = 3.0
EXP_QUIET_SEC = 0.5
EXP_VALID_TRACKED_PCT = 70.0
EXP_VALID_LAP_PCT = 80.0
EXP_LAP_RAD = 2 * np.pi * 0.97
EXP_RESULTS_PATH = os.path.expanduser("~/experiment_results_v3.csv")
EXP_FIELDS = ["timestamp", "participant", "trial", "condition", "condition_name", "rep", "result", "valid",
              "duration_s", "lap_pct", "exits", "time_outside_s", "time_outside_untracked_s", "outside_pct",
              "outside_pct_tracked", "time_in_band_s", "in_band_pct", "in_band_pct_tracked",
              "mean_dist_mm", "min_dist_mm", "warnings", "warnings_from_yolo", "vibration_s",
              "finger_tracked_pct", "mean_tremor_dps", "tremor_samples"]

exp_lock = threading.Lock()
exp = {"active": False, "pid": "", "order": [], "schedule": [], "idx": 0, "phase": "off",
       "t0": 0.0, "m": None, "results": [], "last": None}


def exp_condition():
    if exp["active"] and exp["idx"] < len(exp["schedule"]):
        return exp["schedule"][exp["idx"]]
    return None


def exp_feedback():
    c = exp_condition()
    if c is None:
        return True, True
    return EXP_CONDS[c]["vib"], EXP_CONDS[c]["buzz"]


def exp_new_metrics(now):
    return {"t_run": now, "t_prev": now, "exits": 0, "prev_exit": False, "out": 0.0, "near": 0.0,
            "dsum": 0.0, "dn": 0, "dmin": None, "tracked": 0.0, "ang": 0.0, "prev_ang": None,
            "tr_sum": 0.0, "tr_n": 0, "warn_on": False, "warns": 0, "warns_y": 0, "vib": 0.0, "last_vib": -1e9,
            "last_out": False, "out_lost": 0.0, "out_tracked": 0.0}


def exp_finish(now, result):
    m = exp["m"]
    dur = max(1e-3, now - m["t_run"])
    c = exp["schedule"][exp["idx"]]
    rep = exp["schedule"][:exp["idx"] + 1].count(c)
    tracked = max(1e-3, m["tracked"])
    lap_pct = min(100.0, abs(m["ang"]) / (2 * np.pi) * 100)
    tracked_pct = 100 * m["tracked"] / dur
    valid = "yes" if tracked_pct >= EXP_VALID_TRACKED_PCT and lap_pct >= EXP_VALID_LAP_PCT else "check"
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "participant": exp["pid"], "trial": exp["idx"] + 1,
           "condition": c, "condition_name": EXP_CONDS[c]["name"], "rep": rep, "result": result, "valid": valid,
           "time_outside_untracked_s": round(m["out_lost"], 2),
           "outside_pct_tracked": round(100 * m["out_tracked"] / tracked, 1),
           "in_band_pct_tracked": round(100 * m["near"] / tracked, 1),
           "duration_s": round(dur, 2), "lap_pct": round(min(100.0, abs(m["ang"]) / (2 * np.pi) * 100), 1),
           "exits": m["exits"], "time_outside_s": round(m["out"], 2), "outside_pct": round(100 * m["out"] / dur, 1),
           "time_in_band_s": round(m["near"], 2), "in_band_pct": round(100 * m["near"] / dur, 1),
           "warnings": m["warns"], "warnings_from_yolo": m["warns_y"], "vibration_s": round(m["vib"], 2),
           "tremor_samples": m["tr_n"],
           "mean_dist_mm": round(m["dsum"] / m["dn"], 1) if m["dn"] else "",
           "min_dist_mm": round(m["dmin"], 1) if m["dmin"] is not None else "",
           "finger_tracked_pct": round(100 * m["tracked"] / dur, 1),
           "mean_tremor_dps": round(m["tr_sum"] / m["tr_n"], 2) if m["tr_n"] else ""}
    row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
    exp["results"].append(row)
    exp["last"] = row
    try:
        new = not os.path.exists(EXP_RESULTS_PATH)
        with open(EXP_RESULTS_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EXP_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"result save failed: {e}")
    log_event("trial_end", f"P{exp['pid']} trial {row['trial']} cond {c} {result} exits={row['exits']}")
    exp["idx"] += 1
    exp["phase"] = "finished" if exp["idx"] >= len(exp["schedule"]) else "result"
    exp["m"] = None


def exp_tick(now, exited, pt, d_mm, lv, zone, tremor, src=None):
    with exp_lock:
        if exp["phase"] == "countdown" and now - exp["t0"] >= EXP_COUNTDOWN_SEC:
            exp["phase"] = "running"
            exp["m"] = exp_new_metrics(now)
            log_event("trial_start", f"P{exp['pid']} trial {exp['idx'] + 1} cond {exp_condition()}")
            return
        if exp["phase"] != "running":
            return
        m = exp["m"]
        dt = min(0.2, now - m["t_prev"])
        m["t_prev"] = now
        tracked_now = pt is not None and d_mm is not None
        if tracked_now:
            is_out = bool(exited)
            m["last_out"] = is_out
        else:
            is_out = m["last_out"]
        if is_out:
            m["out"] += dt
            if tracked_now:
                m["out_tracked"] += dt
            else:
                m["out_lost"] += dt
            if not m["prev_exit"]:
                m["exits"] += 1
        m["prev_exit"] = is_out
        if d_mm is not None:
            d_mm = float(d_mm)
            if 0.0 <= d_mm < EXP_WARN_MM:
                m["near"] += dt
        warn = lv > 0
        if warn and not m["warn_on"]:
            m["warns"] += 1
            if src == "Y":
                m["warns_y"] += 1
        m["warn_on"] = warn
        with ctrl_lock:
            vib_ok = not ctrl["paused"] and ctrl["finger_on"][EXP_TRACE_FINGER]
        if warn and vib_ok and exp_feedback()[0]:
            m["vib"] += dt
            m["last_vib"] = now
        if tremor is not None and now - m["last_vib"] > EXP_QUIET_SEC:
            m["tr_sum"] += tremor
            m["tr_n"] += 1
        if pt is not None and d_mm is not None:
            m["tracked"] += dt
            if d_mm >= 0:
                m["dsum"] += d_mm
                m["dn"] += 1
            m["dmin"] = d_mm if m["dmin"] is None else min(m["dmin"], d_mm)
            if zone is not None:
                c = zone[0].mean(axis=0)
                a = float(np.arctan2(pt[1] - c[1], pt[0] - c[0]))
                if m["prev_ang"] is not None:
                    m["ang"] += (a - m["prev_ang"] + np.pi) % (2 * np.pi) - np.pi
                m["prev_ang"] = a
        if abs(m["ang"]) >= EXP_LAP_RAD:
            exp_finish(now, "lap complete")
        elif now - m["t_run"] >= EXP_TIMEOUT_SEC:
            exp_finish(now, "timeout")


def exp_state():
    with exp_lock:
        c = exp_condition()
        now = time.monotonic()
        m = exp["m"]
        run = {}
        if exp["phase"] == "running" and m:
            run = {"elapsed": round(now - m["t_run"], 1), "lap_pct": round(min(100, abs(m["ang"]) / (2 * np.pi) * 100)),
                   "exits": m["exits"]}
        summary = {}
        for k in EXP_CONDS:
            all_rows = [r for r in exp["results"] if r["condition"] == k]
            rows = [r for r in all_rows if r["valid"] == "yes"] or all_rows
            if rows:
                def avg(f):
                    v = [r[f] for r in rows if r[f] != ""]
                    return round(float(np.mean(v)), 2) if v else None
                summary[k] = {"n": len(all_rows), "n_valid": sum(r["valid"] == "yes" for r in all_rows),
                              "outside_pct_tracked": avg("outside_pct_tracked"),
                              "exits": avg("exits"), "outside_pct": avg("outside_pct"),
                              "in_band_pct": avg("in_band_pct"), "mean_dist_mm": avg("mean_dist_mm"),
                              "duration_s": avg("duration_s"), "warnings": avg("warnings")}
        return {"active": exp["active"], "pid": exp["pid"], "phase": exp["phase"], "idx": exp["idx"],
                "total": len(exp["schedule"]), "order": exp["order"], "condition": c,
                "condition_name": EXP_CONDS[c]["name"] if c else None,
                "rep": exp["schedule"][:exp["idx"] + 1].count(c) if c else None,
                "countdown": max(0.0, EXP_COUNTDOWN_SEC - (now - exp["t0"])) if exp["phase"] == "countdown" else None,
                "run": run, "last": exp["last"], "results": exp["results"], "summary": summary,
                "conds": {k: v["name"] for k, v in EXP_CONDS.items()}, "reps": EXP_REPS,
                "zone_ok": shared.get("zone") is not None, "code": shared.get("code"),
                "timeout": EXP_TIMEOUT_SEC, "band_mm": EXP_WARN_MM}


@app.route('/exp/ctrl')
def exp_ctrl():
    a = request.args.get("action", "")
    now = time.monotonic()
    with exp_lock:
        if a == "begin":
            pid = (request.args.get("pid") or "1").strip()
            digits = "".join(ch for ch in pid if ch.isdigit())
            order = EXP_ORDERS[(int(digits) - 1) % 3 if digits else 0]
            exp.update(active=True, pid=pid, order=order, schedule=[c for c in order for _ in range(EXP_REPS)],
                       idx=0, phase="ready", m=None, results=[], last=None)
            log_event("experiment_begin", f"participant {pid} order {order}")
        elif a == "start" and exp["phase"] in ("ready", "result"):
            exp.update(phase="countdown", t0=now)
        elif a == "done" and exp["phase"] == "running":
            exp_finish(now, "stopped by operator")
        elif a == "abort" and exp["phase"] in ("countdown", "running"):
            exp.update(phase="ready" if exp["idx"] == 0 or not exp["results"] else "result", m=None)
            log_event("trial_abort", f"trial {exp['idx'] + 1}")
        elif a == "redo" and exp["phase"] in ("result", "finished") and exp["results"]:
            bad = exp["results"].pop()
            exp["idx"] -= 1
            exp.update(phase="ready", last=exp["results"][-1] if exp["results"] else None)
            try:
                with open(EXP_RESULTS_PATH) as f:
                    lines = f.readlines()
                if lines and lines[-1].startswith(bad["timestamp"]):
                    with open(EXP_RESULTS_PATH, "w") as f:
                        f.writelines(lines[:-1])
            except Exception as e:
                print(f"redo file update failed: {e}")
            log_event("trial_redo", f"trial {bad['trial']} discarded")
        elif a == "end":
            exp.update(active=False, phase="off", m=None)
            log_event("experiment_end")
    push_ctrl()
    return jsonify(exp_state())


@app.route('/exp/state')
def exp_state_route():
    return jsonify(exp_state())


@app.route('/exp/results.csv')
def exp_results_csv():
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXP_FIELDS)
    w.writeheader()
    for r in exp["results"]:
        w.writerow(r)
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=experiment_P{exp['pid'] or 'x'}_{time.strftime('%Y%m%d_%H%M')}.csv"})


@app.route('/exp/all_results.csv')
def exp_all_results_csv():
    try:
        with open(EXP_RESULTS_PATH) as f:
            data = f.read()
    except FileNotFoundError:
        data = ",".join(EXP_FIELDS) + "\n"
    return Response(data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=experiment_results_all.csv"})


@app.route('/exp')
def exp_page():
    return EXP_PAGE


EXP_PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Haptic Glove - Experiment</title>
<style>
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:#0f1216;color:#e8ecf0;font-family:-apple-system,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:8px 16px;border-bottom:1px solid #2a313a;font-size:14px}
header b{font-size:15px}header a{color:#8b96a3;margin-left:auto}
#sig{flex:1;min-height:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
text-align:center;transition:background .15s;padding:16px}
.big{font-size:min(22vh,180px);font-weight:800;line-height:1}
.mid{font-size:min(6vh,44px);font-weight:700}
.small{font-size:min(3vh,20px);opacity:.85}
.s-ready{background:#1d232b}.s-countdown{background:#b7791f}.s-running{background:#1f8f45}
.s-stop{background:#c53030}.s-result{background:#1d232b}.s-finished{background:#1e3a5f}.s-off{background:#151a20}
.bar{width:min(70vw,700px);height:18px;border-radius:9px;background:rgba(0,0,0,.3);overflow:hidden}
.bar div{height:100%;background:#fff;width:0}
#ops{display:flex;gap:8px;justify-content:center;padding:10px;border-top:1px solid #2a313a;flex-wrap:wrap}
.btn{background:#2a313a;border:1px solid #3a424d;color:#e8ecf0;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer}
.btn.primary{background:#3b82f6;border-color:#3b82f6}.btn.warn{background:#7f1d1d;border-color:#7f1d1d}
.btn:disabled{opacity:.35;cursor:default}
#tables{padding:0 16px 12px;max-height:34vh;overflow:auto;font-size:13px}
table{border-collapse:collapse;width:100%;margin-top:8px}th,td{border-bottom:1px solid #2a313a;padding:4px 6px;text-align:right}
th{color:#8b96a3;font-weight:600}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
input{background:#11161b;border:1px solid #3a424d;color:#e8ecf0;border-radius:8px;padding:9px 12px;font-size:18px;width:140px}
.warnz{color:#ffb4b4;font-weight:600}
.task{max-width:700px;font-size:16px;line-height:1.5;color:#c6ced8}
</style></head><body>
<header><b>Experiment - Task 1: Edge tracing</b><span id="hinfo"></span><a href="/">Back to main app</a></header>
<div id="sig" class="s-off"></div>
<div id="ops">
  <button class="btn primary" id="bStart">Start trial (Space)</button>
  <button class="btn" id="bDone">Finished (Enter)</button>
  <button class="btn" id="bAbort">Abort trial</button>
  <button class="btn" id="bRedo">Redo last trial</button>
  <a class="btn" href="/exp/results.csv" style="text-decoration:none">Download results</a>
  <button class="btn warn" id="bEnd">End experiment</button>
</div>
<div id="tables"></div>
<script>
const $ = id => document.getElementById(id);
let E = null, audio = null, lastPhase = "", lastCount = -1, stopUntil = 0;
function beep(f, d, v) {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = f; o.connect(g); g.connect(audio.destination);
  g.gain.value = v || 0.25; o.start(); o.stop(audio.currentTime + d);
}
function initAudio() { if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)(); }
function act(a, extra) {
  initAudio();
  return fetch("/exp/ctrl?" + new URLSearchParams(Object.assign({action: a}, extra || {})), {cache: "no-store"})
    .then(r => r.json()).then(render);
}
$("bStart").onclick = () => act("start");
$("bDone").onclick = () => act("done");
$("bAbort").onclick = () => act("abort");
$("bRedo").onclick = () => { if (confirm("Discard the last trial and run it again?")) act("redo"); };
$("bEnd").onclick = () => { if (confirm("End this experiment?")) act("end"); };
document.addEventListener("keydown", e => {
  if (!E || !E.active || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); act("start"); }
  if (e.code === "Enter") { e.preventDefault(); act("done"); }
});
const fmt = v => v == null || v === "" ? "-" : v;
function setupView() {
  return `<div class="mid">New experiment</div>
  <div class="task">Task: with the gloved hand, trace the <b>inside edge</b> of the zone once around with the
  <b>index fingertip</b>, staying as close to the edge as possible <b>without going outside</b>.
  Only the index finger gives feedback in this experiment, and it starts 2 cm before the edge.
  The edge is not drawn on the table - only the 4 markers are visible.<br><br>
  3 conditions x 3 repetitions = 9 trials. The order of conditions changes with the participant number.</div>
  <div><input id="pid" placeholder="Participant #" value="1"> <button class="btn primary" id="bBegin">Begin</button></div>`;
}
function render(e) {
  E = e;
  const sig = $("sig");
  let cls = "s-" + e.phase, html = "";
  const now = Date.now();
  if (e.phase !== lastPhase) {
    if (e.phase === "running") beep(1320, 0.5, 0.35);
    if ((e.phase === "result" || e.phase === "finished") && lastPhase === "running") { beep(440, 0.9, 0.35); stopUntil = now + 1500; }
    lastPhase = e.phase; lastCount = -1;
  }
  if (e.phase === "countdown") { const c = Math.ceil(e.countdown); if (c !== lastCount && c > 0) { beep(880, 0.15); lastCount = c; } }
  $("hinfo").textContent = e.active ? `Participant ${e.pid}  |  order: ${e.order.join(" > ")}` : "";
  if (!e.active) {
    if (sig.dataset.view !== "setup") { sig.innerHTML = setupView(); sig.dataset.view = "setup";
      $("bBegin").onclick = () => act("begin", {pid: $("pid").value}); }
    sig.className = "s-off";
  } else {
    sig.dataset.view = "";
    const cond = e.condition ? `Condition ${e.condition}: ${e.condition_name}` : "";
    const trial = `Trial ${Math.min(e.idx + 1, e.total)} / ${e.total}` + (e.rep ? `  (repetition ${e.rep} of ${e.reps})` : "");
    const zw = e.zone_ok ? "" : `<div class="warnz">Zone not detected - check that all 4 markers are visible</div>`;
    if (now < stopUntil) {
      cls = "s-stop"; html = `<div class="big">STOP</div>`;
    } else if (e.phase === "ready" || e.phase === "result") {
      const last = e.phase === "result" && e.last ? `<div class="small">Last trial: ${e.last.result}, ${e.last.duration_s}s, exits ${e.last.exits}${e.last.valid === "yes" ? "" : ' - <b style="color:#f0a400">check: low tracking or lap not finished, consider Redo</b>'}</div>` : "";
      html = `<div class="small">${trial}</div><div class="mid">${cond}</div>
        <div class="small">Place your index finger at the start point. Press <b>Space</b> to start.</div>${last}${zw}`;
    } else if (e.phase === "countdown") {
      html = `<div class="small">${cond}</div><div class="big">${Math.max(1, Math.ceil(e.countdown))}</div><div class="mid">Get ready</div>`;
    } else if (e.phase === "running") {
      html = `<div class="big">GO</div><div class="mid">${e.run.elapsed || 0}s</div>
        <div class="bar"><div style="width:${e.run.lap_pct || 0}%"></div></div>
        <div class="small">lap ${e.run.lap_pct || 0}%  |  the trial ends automatically after one lap</div>${zw}`;
    } else if (e.phase === "finished") {
      html = `<div class="mid">Experiment complete</div><div class="small">Thank you! Download the results below.</div>`;
    }
    sig.innerHTML = html;
  }
  sig.className = cls;
  $("bStart").disabled = !(e.active && (e.phase === "ready" || e.phase === "result"));
  $("bDone").disabled = e.phase !== "running";
  $("bAbort").disabled = !(e.phase === "running" || e.phase === "countdown");
  $("bRedo").disabled = !(e.phase === "result" || e.phase === "finished");
  $("bEnd").disabled = !e.active;
  let t = "";
  const sk = Object.keys(e.summary);
  if (sk.length) {
    t += `<table><tr><th>Condition</th><th></th><th>Trials (valid)</th><th>Outside (%)</th><th>Outside, tracked (%)</th><th>In ${e.band_mm} mm band (%)</th><th>Exits</th><th>Warnings</th><th>Mean dist (mm)</th><th>Duration (s)</th></tr>` +
      sk.map(k => { const s = e.summary[k]; return `<tr><td>${k}</td><td>${e.conds[k]}</td><td>${s.n} (${s.n_valid})</td><td>${fmt(s.outside_pct)}</td><td>${fmt(s.outside_pct_tracked)}</td><td>${fmt(s.in_band_pct)}</td><td>${fmt(s.exits)}</td><td>${fmt(s.warnings)}</td><td>${fmt(s.mean_dist_mm)}</td><td>${fmt(s.duration_s)}</td></tr>`; }).join("") + `</table>`;
  }
  if (e.results.length) {
    t += `<table><tr><th>Trial</th><th>Condition</th><th>Rep</th><th>Result</th><th>Valid</th><th>Duration</th><th>Lap %</th><th>Outside %</th><th>In band %</th><th>Exits</th><th>Warnings (YOLO)</th><th>Vibration (s)</th><th>Mean dist</th><th>Tracked %</th></tr>` +
      e.results.slice().reverse().map(r => `<tr><td>${r.trial}</td><td>${r.condition_name}</td><td>${r.rep}</td><td>${r.result}</td><td style="color:${r.valid === "yes" ? "#2fb35a" : "#f0a400"}">${r.valid}</td><td>${r.duration_s}</td><td>${r.lap_pct}</td><td>${r.outside_pct}</td><td>${r.in_band_pct}</td><td>${r.exits}</td><td>${r.warnings} (${r.warnings_from_yolo})</td><td>${r.vibration_s}</td><td>${fmt(r.mean_dist_mm)}</td><td>${r.finger_tracked_pct}</td></tr>`).join("") + `</table>`;
  }
  $("tables").innerHTML = t;
}
setInterval(() => fetch("/exp/state", {cache: "no-store"}).then(r => r.json()).then(render).catch(() => {}), 100);
</script></body></html>'''


@app.route('/nogo', methods=['GET', 'POST'])
def nogo_route():
    a = request.values
    if "clear" in a:
        settings["nogo"] = []
        log_event("nogo", "cleared")
    elif "undo" in a and settings["nogo"]:
        settings["nogo"].pop()
        log_event("nogo", "removed last")
    elif request.is_json:
        d = request.get_json(silent=True) or {}
        zone = shared.get("zone")
        pts = d.get("points") or []
        if zone is None:
            return jsonify({"ok": False, "error": "zone not detected"}), 400
        if len(pts) < 3:
            return jsonify({"ok": False, "error": "need at least 3 points"}), 400
        frame_pts = []
        for x, y in pts:
            fx, fy = unorient(float(x), float(y))
            frame_pts.append([fx, fy])
        Hinv = cv2.getPerspectiveTransform(zone[0].astype(np.float32), UNIT_SQUARE)
        unit = cv2.perspectiveTransform(np.array(frame_pts, np.float32).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        settings["nogo"].append([[round(float(u), 4), round(float(v), 4)] for u, v in unit])
        log_event("nogo", f"added ({len(pts)} points)")
    save_settings()
    return jsonify({"ok": True, "count": len(settings["nogo"])})


@app.route('/log.csv')
def log_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "type", "status_or_event", "max_level", "tremor_dps", "jitter_mm", "fatigue_ratio"])
    rows = [(t, "event", k, d, "", "", "") for t, k, d in event_log]
    rows += [(t, "sample", c, lv, "" if tr is None else round(tr, 3), "" if j is None else round(j, 3), round(f, 3))
             for t, c, lv, tr, j, f in sample_log]
    for r in sorted(rows, key=lambda r: r[0]):
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0]))] + list(r[1:]))
    name = time.strftime("glove_session_%Y%m%d_%H%M.csv", time.localtime(session["start"]))
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@app.route('/ping')
def ping():
    return str(shared["frame_id"])


@app.route('/reset')
def reset():
    marker_cache.clear()
    zone_state["locked"] = False
    return "marker cache cleared"


@app.route('/stream')
def stream():
    mode = request.args.get("mode", "doc")
    mode = mode if mode in RENDERERS else "doc"
    return Response(generate(mode), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return PAGE


PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Haptic Glove Assist</title>
<style>
:root{--bg:#0f1216;--panel:#181d23;--line:#2a313a;--text:#e8ecf0;--mute:#8b96a3;
--safe:#2fb35a;--caution:#f0a400;--danger:#e5484d;--idle:#5b6573;--blue:#3b82f6}
*{box-sizing:border-box}html,body{height:100%;margin:0;overflow:hidden}
body{background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
font-size:13px;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:6px 14px;border-bottom:1px solid var(--line);flex:none}
header h1{font-size:15px;margin:0;font-weight:600}
.timer{color:var(--mute);font-variant-numeric:tabular-nums}
.hright{margin-left:auto;display:flex;gap:8px;align-items:center}
.modes{display:flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.modes button{background:none;border:0;color:var(--mute);padding:5px 12px;font-size:12px;cursor:pointer}
.modes button.on{background:var(--blue);color:#fff}
#banner{margin:8px 14px 0;border-radius:10px;padding:8px 16px;font-size:22px;font-weight:700;flex:none;
letter-spacing:.5px;display:flex;align-items:center;gap:12px;transition:background .15s}
#banner small{font-size:13px;font-weight:500;opacity:.9}
.b-safe{background:var(--safe)}.b-caution{background:var(--caution);color:#1b1400}
.b-outside,.b-lost,.b-nogo{background:var(--danger)}.b-paused{background:var(--blue)}
.b-no_zone,.b-no_hand{background:var(--idle)}
.b-outside,.b-nogo{animation:pulse .6s infinite alternate}@keyframes pulse{to{filter:brightness(1.35)}}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:12px;padding:10px 14px}
section{min-height:0;display:flex;flex-direction:column}
.videobox{position:relative;flex:1;min-height:0;background:#000;border-radius:10px;overflow:hidden;
display:flex;align-items:center;justify-content:center}
#video{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block}
.overlay{position:absolute;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;
gap:10px;text-align:center;padding:16px;background:rgba(0,0,0,.65);font-size:20px;font-weight:600}
.overlay.show{display:flex}
#offline{background:rgba(120,0,0,.85)}
#rest{background:rgba(20,20,30,.82)}
#rest .big{font-size:24px;color:#ffd166}
#chips{display:flex;gap:6px;margin-top:8px;flex:none}
.chip{flex:1;border-radius:8px;padding:5px 4px;text-align:center;background:var(--panel);
border:2px solid var(--line);font-weight:600;font-size:12px}
.chip .lv{font-size:11px;color:var(--mute);margin-top:2px;font-weight:500}
.chip.off{opacity:.45;border-style:dashed}.chip.off .lv{color:var(--danger)}
.side{min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.card h3{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--mute)}
#pauseBtn{width:100%;padding:10px;font-size:16px;font-weight:700;border-radius:10px;border:0;cursor:pointer;
background:#c2410c;color:#fff;flex:none}
#pauseBtn.paused{background:var(--safe)}
.row{display:grid;grid-template-columns:50px 40px 1fr 34px;align-items:center;gap:6px;margin:3px 0;font-size:12px}
.row input[type=range]{width:100%}
.tog{width:38px;height:20px;border-radius:10px;border:0;cursor:pointer;background:#3a424d;position:relative}
.tog::after{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;
background:#fff;transition:left .15s}
.tog.on{background:var(--safe)}.tog.on::after{left:20px}
.line{display:flex;align-items:center;justify-content:space-between;margin:4px 0;font-size:12px;gap:6px}
.line input[type=range]{flex:1}
.btn{background:#2a313a;border:1px solid #3a424d;color:var(--text);border-radius:7px;padding:4px 9px;
cursor:pointer;font-size:12px}
.btn:hover{background:#343c46}
.btns{display:flex;flex-wrap:wrap;gap:6px}
.gauge{height:10px;border-radius:5px;background:#2a313a;overflow:hidden}
.gauge div{height:100%;width:0;background:linear-gradient(90deg,var(--safe),var(--caution),var(--danger));
transition:width .3s}
.mute{color:var(--mute);font-size:11px}
.eng-only{display:none}body.eng .eng-only{display:block}body.eng .doc-only{display:none}
.kv{display:grid;grid-template-columns:1fr auto 1fr auto;gap:2px 8px;font-size:11px;font-variant-numeric:tabular-nums}
#events{max-height:90px;overflow-y:auto}
.ev{font-size:11px;color:var(--mute);border-top:1px solid var(--line);padding:2px 0}
body.eng main{grid-template-columns:minmax(0,1fr) 560px}
body.eng .side{display:grid;grid-template-columns:1fr 1fr;align-content:start}
body.eng #pauseBtn{grid-column:1/-1}
#draw{position:absolute;display:none;cursor:crosshair}
#viewbar{position:absolute;right:8px;bottom:8px;display:flex;gap:4px;z-index:4;opacity:.75}
#viewbar:hover{opacity:1}#viewbar .btn{background:rgba(20,24,30,.85);font-size:15px;padding:3px 9px}
#viewbar .btn.on{background:var(--blue);border-color:var(--blue)}
body.drawing #viewbar{display:none}
#drawbar{position:absolute;top:8px;left:50%;transform:translateX(-50%);display:none;gap:8px;align-items:center;
background:rgba(15,18,22,.9);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:12px;z-index:5}
body.drawing #draw{display:block}body.drawing #drawbar{display:flex}
.btn.red{border-color:#7f1d1d;background:#3b1517}
#wizard{position:fixed;inset:0;background:rgba(5,7,10,.78);display:none;align-items:center;justify-content:center;z-index:20}
#wizard.show{display:flex}
.wbox{width:min(520px,92vw);background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 22px}
.wsteps{display:flex;gap:6px;margin-bottom:14px}.wsteps div{flex:1;height:5px;border-radius:3px;background:#2a313a}
.wsteps div.done{background:var(--safe)}.wsteps div.cur{background:var(--blue)}
.wbox h2{margin:0 0 8px;font-size:20px}.wbox p{color:#c6ced8;font-size:14px;line-height:1.5;margin:6px 0}
.wstat{margin:12px 0;padding:10px 12px;border-radius:8px;background:#11161b;font-size:14px;font-weight:600}
.wstat.ok{color:var(--safe)}.wstat.wait{color:var(--caution)}
.wnav{display:flex;justify-content:space-between;align-items:center;margin-top:14px}
.wnav .btn{font-size:14px;padding:8px 16px}.btn.primary{background:var(--blue);border-color:var(--blue)}
.btn:disabled{opacity:.4;cursor:default}
@media(max-width:800px){main{grid-template-columns:1fr}html,body{overflow:auto}}
</style></head>
<body>
<header><h1>Haptic Glove Assist</h1><span class="timer" id="timer">00:00</span>
<span class="mute" id="exits"></span>
<div class="hright"><a class="btn" href="/exp" style="text-decoration:none">Experiment</a><button class="btn" id="setupBtn">Setup guide</button><button class="btn" id="newS">New session</button>
<a class="btn" href="/log.csv" style="text-decoration:none">Download log</a><button class="btn" id="sound">Enable sound alerts</button>
<div class="modes"><button id="mDoc" class="on">Doctor</button><button id="mEng">Engineer</button></div></div></header>
<div id="banner" class="b-no_zone"><span id="btext">CONNECTING...</span><small id="bsub"></small></div>
<main>
<section>
  <div class="videobox"><img id="video" alt="camera"><canvas id="draw"></canvas>
    <div id="viewbar"><button class="btn" id="rotL" title="Rotate left">&#8634;</button>
      <button class="btn" id="rotR" title="Rotate right">&#8635;</button>
      <button class="btn" id="mirror" title="Mirror left-right">&#8646;</button></div>
    <div id="drawbar"><span id="drawtxt">Click around the area to avoid</span>
      <button class="btn" id="drawUndo">Undo point</button><button class="btn primary" id="drawDone">Finish</button>
      <button class="btn" id="drawCancel">Cancel</button></div>
    <div class="overlay" id="rest"><div class="big">Hand steadiness has dropped</div>
      <div>Consider taking a short rest.</div><button class="btn" id="dismiss">Dismiss for 5 min</button></div>
    <div class="overlay" id="offline"><div>CONNECTION LOST</div>
      <div style="font-size:16px;font-weight:400">Glove guidance may be unavailable. Reconnecting...</div></div>
  </div>
  <div id="chips"></div>
</section>
<aside class="side">
  <button id="pauseBtn">PAUSE GLOVE</button>
  <div class="card"><h3>Vibration per finger</h3><div id="fingers"></div>
    <div class="line"><span>Buzzer</span><button class="tog on" id="buzzer"></button></div></div>
  <div class="card"><h3>Hand steadiness</h3>
    <div class="gauge"><div id="gauge"></div></div>
    <div class="line"><span id="ftext">-</span><span class="mute" id="ftrem"></span></div>
    <div class="btns"><button class="btn" id="calib">Recalibrate baseline</button></div></div>
  <div class="card"><h3>Zone</h3><div class="line"><span id="ztext">-</span></div>
    <div class="btns"><button class="btn" id="zlock">Lock zone</button><button class="btn" id="zredo">Re-detect markers</button></div>
    <div class="line" style="margin-top:8px"><span id="ngtext">No-go areas: 0</span></div>
    <div class="btns"><button class="btn" id="ngAdd">+ Add no-go area</button><button class="btn" id="ngUndo">Remove last</button>
    <button class="btn red" id="ngClear">Clear all</button></div></div>
  <div class="card eng-only"><h3>Event log</h3><div id="events"></div></div>
  <div class="card eng-only"><h3>Engineer</h3>
    <div class="kv" id="stats"></div>
    <div class="line"><span title="How bright the camera image is. Darker helps the glove lights stand out." style="cursor:help;border-bottom:1px dotted #666">Camera brightness</span><input type="range" id="ev" min="-4" max="0" step="0.5"><span id="evv"></span></div>
    <div class="line"><span title="How bright a spot must be to count as a glove light. Turn up if lamps or reflections are mistaken for glove lights; turn down if glove lights are missed." style="cursor:help;border-bottom:1px dotted #666">LED strictness</span><input type="range" id="corev" min="150" max="255" step="5"><span id="corevv"></span></div>
    <div class="line"><span title="Makes the Doctor screen brighter and more natural looking. Does not change tracking." style="cursor:help;border-bottom:1px dotted #666">Brighten doctor view</span><button class="tog" id="enh"></button></div>
    <div class="line"><span title="How bright the Doctor screen looks. Higher = brighter." style="cursor:help;border-bottom:1px dotted #666">View brightness</span><input type="range" id="gamma" min="1" max="3" step="0.1"><span id="gammav"></span></div>
    <div class="line"><span title="How sharp the details look on the Doctor screen. Higher = sharper but grainier. 0 = off." style="cursor:help;border-bottom:1px dotted #666">View detail</span><input type="range" id="clahe" min="0" max="4" step="0.5"><span id="clahev"></span></div>
  </div>
</aside>
</main>
<div id="wizard"><div class="wbox">
  <div class="wsteps"><div></div><div></div><div></div><div></div></div>
  <h2 id="wtitle"></h2><p id="wbody"></p><div class="wstat" id="wstat"></div>
  <div id="wextra"></div>
  <div class="wnav"><button class="btn" id="wskip">Skip setup</button>
    <div style="display:flex;gap:8px"><button class="btn" id="wback">Back</button><button class="btn primary" id="wnext">Next</button></div></div>
</div></div>
<script>
const $ = id => document.getElementById(id);
const NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const LVCOL = ["#2fb35a", "#7bc96f", "#f5c542", "#f0a400", "#f06b00", "#e5484d"];
let mode = "doc", S = null, lastOk = Date.now(), audio = null, prevCode = "", alarmT = 0;
try { mode = localStorage.getItem("glove_mode") || "doc"; } catch (e) {}

function ctrl(q) {
  return fetch("/ctrl?" + new URLSearchParams(q), {cache: "no-store"}).then(r => r.json()).then(apply).catch(() => {});
}
function setMode(m) {
  mode = m; try { localStorage.setItem("glove_mode", m); } catch (e) {}
  document.body.classList.toggle("eng", m === "eng");
  $("mDoc").classList.toggle("on", m === "doc"); $("mEng").classList.toggle("on", m === "eng");
  connect();
}
let lastFrame = Date.now(), wasDown = false;
function connect() { $("video").src = "/stream?mode=" + mode + "&t=" + Date.now(); }
$("video").onload = () => { lastFrame = Date.now(); };
$("video").onerror = () => setTimeout(connect, 1000);
$("mDoc").onclick = () => setMode("doc"); $("mEng").onclick = () => setMode("eng");

const fw = $("fingers"), chips = $("chips");
NAMES.forEach((n, i) => {
  fw.insertAdjacentHTML("beforeend", `<div class="row"><span>${n}</span>
    <button class="tog on" id="t${i}"></button>
    <input type="range" id="g${i}" min="20" max="100" step="5" value="100"><span id="gv${i}">100%</span></div>`);
  chips.insertAdjacentHTML("beforeend", `<div class="chip" id="c${i}">${n}<div class="lv" id="cl${i}">-</div></div>`);
});
NAMES.forEach((n, i) => {
  $("t" + i).onclick = () => ctrl({finger: i, on: S && S.finger_on[i] ? 0 : 1});
  let deb;
  $("g" + i).oninput = e => { $("gv" + i).textContent = e.target.value + "%"; clearTimeout(deb);
    deb = setTimeout(() => ctrl({finger: i, gain: e.target.value / 100}), 150); };
});
$("pauseBtn").onclick = () => ctrl({pause: S && S.paused ? 0 : 1});
$("buzzer").onclick = () => ctrl({buzzer: S && S.buzzer_on ? 0 : 1});
$("calib").onclick = () => ctrl({calibrate: 1});
$("dismiss").onclick = () => ctrl({dismiss: 1});
$("zlock").onclick = () => ctrl({zone: S && S.zone_locked ? "unlock" : "lock"});
$("zredo").onclick = () => ctrl({zone: "redetect"});
$("rotL").onclick = () => ctrl({rotate: ((S ? S.rotate : 0) + 270) % 360});
$("rotR").onclick = () => ctrl({rotate: ((S ? S.rotate : 0) + 90) % 360});
$("mirror").onclick = () => ctrl({mirror: S && S.mirror ? 0 : 1});
$("newS").onclick = () => { if (confirm("Start a new session? All finger settings reset to ON and the log is cleared.")) { wClosed = false; ctrl({session: "new"}); } };
$("enh").onclick = () => ctrl({enhance: S && S.cfg.enhance ? 0 : 1});
[["ev", "ev"], ["corev", "core_v"], ["gamma", "gamma"], ["clahe", "clahe"]].forEach(([id, key]) => {
  let deb; $(id).oninput = e => { $(id + "v").textContent = e.target.value; clearTimeout(deb);
    deb = setTimeout(() => ctrl({[key]: e.target.value}), 200); };
});
$("sound").onclick = () => {
  audio = new (window.AudioContext || window.webkitAudioContext)();
  $("sound").textContent = "Sound alerts on"; $("sound").disabled = true; beep(880, 0.15);
};
function beep(f, d) {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = f; o.connect(g); g.connect(audio.destination);
  g.gain.value = 0.2; o.start(); o.stop(audio.currentTime + d);
}
const fmt = s => String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");

function apply(s) {
  S = s; lastOk = Date.now();
  const b = $("banner"); b.className = "b-" + s.code; $("btext").textContent = s.text;
  let sub = "";
  if (s.code === "caution") sub = "Level " + s.max_level + " of " + s.num_levels;
  if (s.code === "no_zone") sub = "Markers found: " + s.markers + " / 4";
  if (s.code === "lost") sub = "Camera feed stalled - glove feedback is not updating";
  if (s.code === "paused") sub = "Vibration and buzzer are off";
  if (s.code === "nogo") sub = "A finger is inside a no-go area";
  if (s.exp) sub = "EXPERIMENT: " + s.exp + (sub ? "  |  " + sub : "");
  const off = s.finger_on.map((v, i) => v ? null : NAMES[i]).filter(Boolean);
  if (off.length && s.code !== "paused") sub += (sub ? "  |  " : "") + "OFF: " + off.join(", ");
  if (!s.buzzer_on && s.code !== "paused") sub += (sub ? "  |  " : "") + "Buzzer OFF";
  $("bsub").textContent = sub;
  $("timer").textContent = fmt(s.session_sec);
  $("exits").textContent = "Zone exits: " + s.exits;
  $("pauseBtn").textContent = s.paused ? "RESUME GLOVE" : "PAUSE GLOVE";
  $("pauseBtn").classList.toggle("paused", s.paused);
  $("buzzer").classList.toggle("on", s.buzzer_on);
  NAMES.forEach((n, i) => {
    $("t" + i).classList.toggle("on", s.finger_on[i]);
    const g = $("g" + i); if (document.activeElement !== g) { g.value = Math.round(s.gain[i] * 100); $("gv" + i).textContent = g.value + "%"; }
    const c = $("c" + i), lv = s.levels[i];
    c.classList.toggle("off", !s.finger_on[i]);
    c.style.borderColor = s.finger_on[i] && s.seen[i] ? LVCOL[lv] : "";
    $("cl" + i).textContent = !s.finger_on[i] ? "VIBRATION OFF" : (!s.seen[i] ? "not visible" : (lv ? "level " + lv : "OK"));
  });
  const f = s.fatigue;
  $("gauge").style.width = (f.calibrating ? 0 : Math.round(f.level * 100)) + "%";
  $("ftext").textContent = f.calibrating ? "Learning your baseline... " + f.calib_left + "s (keep hand in zone)"
    : (f.warn ? "Rest recommended" : (f.level > 0.5 ? "Elevated" : "Steady"));
  $("ftrem").textContent = s.imu_ok ? "IMU " + s.tremor.toFixed(1) + " deg/s" : "IMU not connected";
  $("rest").classList.toggle("show", f.warn);
  $("ztext").textContent = s.zone_ok ? (s.zone_locked ? "Zone locked" : "Zone active (tracking markers)") : "Markers " + s.markers + "/4";
  $("zlock").textContent = s.zone_locked ? "Unlock zone" : "Lock zone";
  $("mirror").classList.toggle("on", s.mirror);
  $("enh").classList.toggle("on", s.cfg.enhance);
  [["ev", "ev"], ["corev", "core_v"], ["gamma", "gamma"], ["clahe", "clahe"]].forEach(([id, key]) => {
    if (document.activeElement !== $(id)) { $(id).value = s.cfg[key]; $(id + "v").textContent = s.cfg[key]; } });
  if (mode === "eng") {
    $("stats").innerHTML = [["Track fps", (s.fps || 0).toFixed(1)], ["YOLO ms", Math.round(s.yolo_ms || 0)],
      ["YOLO conf", (s.hand_conf || 0).toFixed(2)], ["Render ms", Math.round(s.render_ms || 0)],
      ["Markers", s.markers], ["Jitter mm", s.jitter == null ? "-" : s.jitter.toFixed(2)],
      ["Tremor deg/s", s.tremor == null ? "-" : s.tremor.toFixed(2)], ["Fatigue ratio", f.ratio],
      ["Base tremor", f.base_tremor == null ? "-" : f.base_tremor.toFixed(2)],
      ["Base jitter", f.base_jitter == null ? "-" : f.base_jitter.toFixed(2)], ["Sudden move", s.sudden]]
      .map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
    $("events").innerHTML = s.events.map(e => `<div class="ev">${e.t} ${e.kind} ${e.detail}</div>`).join("") || '<div class="ev">-</div>';
  }
  if (s.code !== prevCode && (s.code === "lost")) beep(440, 0.4);
  if (f.warn && Date.now() - alarmT > 60000) { beep(660, 0.25); alarmT = Date.now(); }
  prevCode = s.code;
  $("ngtext").textContent = "No-go areas: " + s.nogo_count + (s.nogo_hits ? `  (entered ${s.nogo_hits}x)` : "");
  $("exits").textContent = "Zone exits: " + s.exits + (s.nogo_hits ? "  |  No-go entries: " + s.nogo_hits : "");
  if (document.body.classList.contains("drawing")) placeCanvas();
  wTick(s);
}

// ---- no-go drawing ----
let drawPts = [];
const cv = $("draw"), vid = $("video");
function placeCanvas() {
  const box = vid.parentElement.getBoundingClientRect(), r = vid.getBoundingClientRect();
  cv.style.left = (r.left - box.left) + "px"; cv.style.top = (r.top - box.top) + "px";
  cv.style.width = r.width + "px"; cv.style.height = r.height + "px";
  cv.width = r.width; cv.height = r.height;
  const g = cv.getContext("2d"); g.clearRect(0, 0, cv.width, cv.height);
  if (!drawPts.length) return;
  const sx = r.width / vid.naturalWidth, sy = r.height / vid.naturalHeight;
  g.fillStyle = "rgba(229,72,77,.25)"; g.strokeStyle = "#ff5c5c"; g.lineWidth = 2;
  g.beginPath(); drawPts.forEach(([x, y], i) => i ? g.lineTo(x * sx, y * sy) : g.moveTo(x * sx, y * sy));
  if (drawPts.length > 2) g.closePath(); g.fill(); g.stroke();
  g.fillStyle = "#fff"; drawPts.forEach(([x, y]) => { g.beginPath(); g.arc(x * sx, y * sy, 4, 0, 7); g.fill(); });
}
function startDraw() {
  if (!S || !S.zone_ok) { alert("The zone must be detected first (all 4 markers visible)."); return; }
  drawPts = []; document.body.classList.add("drawing"); updDrawTxt(); placeCanvas();
}
function endDraw() { drawPts = []; document.body.classList.remove("drawing"); placeCanvas(); }
function updDrawTxt() { $("drawtxt").textContent = drawPts.length < 3 ? `Click around the area to avoid (${drawPts.length} points, need 3+)` : `${drawPts.length} points - click Finish when done`; $("drawDone").disabled = drawPts.length < 3; }
cv.onclick = e => {
  const r = cv.getBoundingClientRect();
  drawPts.push([(e.clientX - r.left) * vid.naturalWidth / r.width, (e.clientY - r.top) * vid.naturalHeight / r.height]);
  updDrawTxt(); placeCanvas();
};
$("drawUndo").onclick = () => { drawPts.pop(); updDrawTxt(); placeCanvas(); };
$("drawCancel").onclick = endDraw;
$("drawDone").onclick = () => {
  fetch("/nogo", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({points: drawPts, mode: mode})}).then(r => r.json())
    .then(d => { if (!d.ok) alert(d.error); endDraw(); }).catch(endDraw);
};
window.addEventListener("resize", placeCanvas);
$("ngAdd").onclick = startDraw;
$("ngUndo").onclick = () => fetch("/nogo?undo=1");
$("ngClear").onclick = () => { if (confirm("Remove all no-go areas?")) fetch("/nogo?clear=1"); };

// ---- setup wizard ----
let wLastSec = 0, wStep = 0, wOpen = false, wClosed = false, wSeenSince = 0, wCalibSent = false, wTested = false;
const WSTEPS = [
  {t: "1. Place the markers", b: "Put the 4 markers at the corners of the working area so the camera can see all of them.",
   ok: s => s.zone_ok, okTxt: "All 4 markers found", wait: s => `Looking for markers... ${s.markers} / 4 found`},
  {t: "2. Put on the glove", b: "Wear the glove and place your hand flat inside the green area, fingers spread.",
   ok: s => wSeenSince && Date.now() - wSeenSince > 1500, okTxt: "Hand detected", wait: s => "Waiting for the hand..."},
  {t: "3. Learn your steadiness", b: "Keep your hand still and relaxed inside the area. This sets your personal baseline for the rest warning.",
   ok: s => wCalibSent && !s.fatigue.calibrating, okTxt: "Baseline saved",
   wait: s => `Measuring... ${s.fatigue.calib_left}s left (keep hand in the area)`},
  {t: "4. Check the feedback", b: "Press Test vibration: each finger buzzes once, thumb to pinky. Then test the buzzer.",
   ok: s => wTested, okTxt: "Feedback checked - ready to start", wait: s => s.testing ? "Testing..." : "Press Test vibration"},
];
function wRender() {
  const st = WSTEPS[wStep], ok = S && st.ok(S);
  document.querySelectorAll(".wsteps div").forEach((d, i) => d.className = i < wStep ? "done" : (i === wStep ? "cur" : ""));
  $("wtitle").textContent = st.t; $("wbody").textContent = st.b;
  $("wstat").className = "wstat " + (ok ? "ok" : "wait");
  $("wstat").textContent = ok ? "✓ " + st.okTxt : (S ? st.wait(S) : "Connecting...");
  $("wback").disabled = wStep === 0;
  $("wnext").textContent = wStep === WSTEPS.length - 1 ? "Start" : "Next";
  $("wnext").disabled = !ok;
  $("wextra").innerHTML = wStep === 3 ? `<div class="btns"><button class="btn" id="wtv">Test vibration</button>
    <button class="btn" id="wtb">Test buzzer</button></div>
    <label style="display:block;margin-top:10px;font-size:13px"><input type="checkbox" id="wlock" checked> Lock the zone when starting</label>` : "";
  if (wStep === 3) {
    $("wtv").onclick = () => { ctrl({test: "fingers"}); setTimeout(() => { wTested = true; }, 2400); };
    $("wtb").onclick = () => ctrl({test: "buzzer"});
  }
}
function wShow(step) { wOpen = true; wStep = step || 0; wTested = false; wCalibSent = false; $("wizard").classList.add("show"); wRender(); }
function wHide() { wOpen = false; wClosed = true; $("wizard").classList.remove("show"); }
$("wskip").onclick = () => { ctrl({setup: "done"}); wHide(); };
$("wback").onclick = () => { if (wStep > 0) { wStep--; wRender(); } };
$("wnext").onclick = () => {
  if (wStep < WSTEPS.length - 1) {
    wStep++;
    if (wStep === 2) { ctrl({calibrate: 1}); wCalibSent = true; }
    wRender();
  } else {
    const lock = $("wlock") && $("wlock").checked;
    ctrl(lock ? {setup: "done", zone: "lock"} : {setup: "done"}); wHide();
  }
};
$("setupBtn").onclick = () => { ctrl({setup: "restart"}); wShow(0); };
function wTick(s) {
  if (s.seen.some(Boolean)) { wSeenSince = wSeenSince || Date.now(); } else wSeenSince = 0;
  if (s.session_sec + 5 < wLastSec) wClosed = false;
  wLastSec = s.session_sec;
  if (!s.setup_done && !wOpen && !wClosed) wShow(0);
  if (wOpen) wRender();
}

setInterval(() => {
  fetch("/state", {cache: "no-store"}).then(r => r.json()).then(apply).catch(() => {});
  const down = Date.now() - lastOk > 2000;
  $("offline").classList.toggle("show", down);
  if (down) { if (Date.now() - alarmT > 1500) { beep(330, 0.3); alarmT = Date.now(); } wasDown = true; }
  else if (wasDown) { wasDown = false; connect(); }
}, 300);
setMode(mode);
</script></body></html>'''


if __name__ == '__main__':
    for target in (capture_loop, track_loop, yolo_loop, render_loop):
        threading.Thread(target=target, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True)
    finally:
        for p in haptic_pubs:
            p.publish(Float32(data=0.0))
        exit_pub.publish(Bool(data=False))
        ros_node.destroy_node()
        rclpy.shutdown()
