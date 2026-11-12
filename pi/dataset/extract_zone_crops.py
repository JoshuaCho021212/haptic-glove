#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np

VIDEO_DIR = os.path.expanduser("~/hand_dataset/videos_zone")
OUT_DIR = os.path.expanduser("~/hand_dataset/zone_crops")
EVERY_N = 3
ROI_MARGIN = 0.35
NUM_CORNERS = 4

params = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.01
aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


def zone_roi(cache, w, h):
    recent = sorted(cache.items(), key=lambda kv: kv[1][1], reverse=True)[:NUM_CORNERS]
    if len(recent) < NUM_CORNERS:
        return None
    markers = [v[0] for _, v in recent]
    cen = np.mean([m.mean(axis=0) for m in markers], axis=0)
    inner = np.array([m[np.argmin(np.linalg.norm(m - cen, axis=1))] for m in markers], dtype=np.float32)
    bx, by, bw, bh = cv2.boundingRect(inner.astype(np.int32))
    mx, my = int(bw * ROI_MARGIN), int(bh * ROI_MARGIN)
    return max(0, bx - mx), max(0, by - my), min(w, bx + bw + mx), min(h, by + bh + my)


os.makedirs(OUT_DIR, exist_ok=True)
total = 0
for path in sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))):
    tag = os.path.basename(path).rsplit("_", 2)[0]
    cap = cv2.VideoCapture(path)
    cache, idx, saved, skipped = {}, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                cache[int(i)] = (c[0].astype(np.float32), idx)
        if idx % EVERY_N == 0:
            roi = zone_roi(cache, frame.shape[1], frame.shape[0])
            if roi is None:
                skipped += 1
            else:
                x0, y0, x1, y1 = roi
                cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_{saved:04d}.jpg"), frame[y0:y1, x0:x1],
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
        idx += 1
    cap.release()
    total += saved
    print(f"{tag:20s} frames={idx:4d} crops={saved:4d} skipped(no markers)={skipped}")
print(f"\ntotal crops: {total} -> {OUT_DIR}")
