"""CPU latency benchmark for one or more ONNX models (640x640 input)."""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def bench(path, n, warmup=3):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(warmup):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, {name: x})
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    for m in args.models:
        ms = bench(m, args.n)
        print(f"{os.path.basename(m):24s} {os.path.getsize(m) / 1e6:5.1f} MB  "
              f"{ms:7.1f} ms  ({1000 / ms:5.2f} FPS)")


if __name__ == "__main__":
    main()
