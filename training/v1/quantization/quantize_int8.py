"""INT8 static post-training quantization for the YOLOv8n-pose hand model.

The output head (/model.22/) stays in FP32: it packs box coordinates (0-640)
and confidences (0-1) into one tensor, and quantizing it collapses mAP to 0.
The result is saved with ONNX IR version 9 so it loads on onnxruntime 1.18.1 (Pi 4).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

HEAD_PREFIX = "/model.22/"
SIZE = 640


def letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def preprocess(path):
    img = letterbox(cv2.imread(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None]


class CalibReader(CalibrationDataReader):
    def __init__(self, model_path, paths):
        self.name = ort.InferenceSession(model_path).get_inputs()[0].name
        self.it = iter(paths)

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.name: preprocess(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/best.onnx")
    ap.add_argument("--calib", required=True, help="folder of calibration images (training split)")
    ap.add_argument("--out", default="weights/best_int8.onnx")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.calib, "*.jpg")) +
                   glob.glob(os.path.join(args.calib, "*.png")))
    print(f"calibration images: {len(paths)}")

    pre = args.out.replace(".onnx", "_pre.onnx")
    tmp = args.out.replace(".onnx", "_tmp.onnx")
    quant_pre_process(args.model, pre)

    head = [n.name for n in onnx.load(pre).graph.node if HEAD_PREFIX in n.name]
    print(f"keeping {len(head)} head nodes in FP32")

    quantize_static(
        pre, tmp, CalibReader(pre, paths),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=head,
    )

    m = onnx.load(tmp)
    m.ir_version = min(m.ir_version, 9)
    onnx.save(m, args.out)
    os.remove(pre)
    os.remove(tmp)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
