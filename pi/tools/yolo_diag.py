import time
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

MODELS = {"mix": "/home/joshuacho/best_zone_mix.onnx", "dark": "/home/joshuacho/best_zone_dark.onnx"}
W, H = 1280, 960
MARGIN = 0.35

sessions = {}
for name, path in MODELS.items():
    o = ort.SessionOptions()
    o.intra_op_num_threads = 3
    s = ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])
    sessions[name] = (s, s.get_inputs()[0].name)


def infer(sess, name, img):
    h, w = img.shape[:2]
    sc = 640 / max(h, w)
    pad = np.full((640, 640, 3), 114, np.uint8)
    pad[:int(h * sc), :int(w * sc)] = cv2.resize(img, (int(w * sc), int(h * sc)))
    out = sess.run(None, {name: (pad.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0][0].T
    i = int(out[:, 4].argmax())
    k = out[i, 5:].reshape(21, 3)
    return float(out[i, 4]), float((k[:, 2] > 0.3).mean())


cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (W, H), "format": "RGB888"}, buffer_count=4))
cam.start()
cam.set_controls({"AeEnable": True, "ExposureValue": -1.0, "AfMode": 2})
time.sleep(3)

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
pts = {}
for _ in range(20):
    f = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    c, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if ids is not None:
        for cc, i in zip(c, ids.flatten()):
            pts[int(i)] = cc[0]
    if len(pts) >= 4:
        break
print(f"markers: {sorted(pts)}")

if pts:
    allp = np.concatenate(list(pts.values()))
    bx, by, bw, bh = cv2.boundingRect(allp.astype(np.int32))
    mx, my = int(bw * MARGIN), int(bh * MARGIN)
    x0, y0, x1, y1 = max(0, bx - mx), max(0, by - my), min(W, bx + bw + mx), min(H, by + bh + my)
else:
    x0, y0, x1, y1 = 0, 0, W, H

print("\nKeep your OPEN hand inside the zone...\n")
print(f"{'EV':>5} {'order':>8} {'gain':>5} | {'mix conf':>8} {'kpt%':>5} | {'dark conf':>9} {'kpt%':>5}")
for ev in (-1.0, -2.0, -3.0):
    cam.set_controls({"ExposureValue": ev})
    time.sleep(2.5)
    raw = cam.capture_array()
    for order, img_full in (("pipeline", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)), ("swapped", raw)):
        crop = img_full[y0:y1, x0:x1]
        if order == "pipeline":
            cv2.imwrite(f"/home/joshuacho/diag_ev{int(ev)}.jpg", crop[:, :, ::-1])
        for gain in (1.0, 2.5):
            img = cv2.convertScaleAbs(crop, alpha=gain) if gain != 1.0 else crop
            r = [infer(s, n, img) for s, n in sessions.values()]
            print(f"{ev:>5} {order:>8} {gain:>5} | {r[0][0]:>8.2f} {r[0][1]:>5.0%} | {r[1][0]:>9.2f} {r[1][1]:>5.0%}")
cam.stop()
