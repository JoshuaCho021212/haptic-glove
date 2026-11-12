import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def make_dark_val(src, dst, factor):
    for sub in ("images/val", "labels/val"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted((src / "images" / "val").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        dark = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dst / "images" / "val" / img_path.name), dark, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lbl = src / "labels" / "val" / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dst / "labels" / "val" / lbl.name)
        n += 1
    return n


def write_yaml(path, root, train, val):
    path.write_text(
        f"path: {root.as_posix()}\n"
        f"train: {train}\n"
        f"val: {val}\n\n"
        "kpt_shape: [21, 3]\n"
        "flip_idx: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]\n\n"
        "names:\n  0: hand\n"
    )


def evaluate(weights, data, tag, workers):
    m = YOLO(str(weights)).val(data=str(data), imgsz=640, batch=8, workers=workers, plots=False,
                               name=f"dark_cmp_{tag}", exist_ok=True, verbose=False)
    return m.box.map50, m.pose.map50, m.pose.map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\j8cho\yolo_dataset_zone")
    ap.add_argument("--weights", default=r"C:\Users\j8cho\runs\pose\zone_ft_mix\weights\best.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dark", type=float, default=0.125)
    args = ap.parse_args()

    root = Path(args.data)
    n_train = len(list((root / "images" / "train").glob("*.jpg")))
    n_val = len(list((root / "images" / "val").glob("*.jpg")))
    print(f"yesterday-only data: train {n_train} | val {n_val}")

    normal_yaml = root / "data_zone_only.yaml"
    write_yaml(normal_yaml, root, "images/train", "images/val")

    dark_root = root.parent / (root.name + "_darkval")
    print(f"dark val ({args.dark:.3f}x brightness): {make_dark_val(root, dark_root, args.dark)} images -> {dark_root}")
    dark_yaml = dark_root / "data_dark.yaml"
    write_yaml(dark_yaml, dark_root, (root / "images" / "train").as_posix(), "images/val")

    print("\n=== baseline (zone_ft_mix) ===")
    base_n = evaluate(args.weights, normal_yaml, "base_normal", args.workers)
    base_d = evaluate(args.weights, dark_yaml, "base_dark", args.workers)

    model = YOLO(args.weights)
    model.train(
        data=str(normal_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.9,
        degrees=30.0,
        scale=0.5,
        translate=0.15,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        patience=25,
        plots=False,
        name="zone_dark",
        exist_ok=True,
    )
    best = Path(model.trainer.best)

    print("\n=== new (zone_dark) ===")
    new_n = evaluate(best, normal_yaml, "new_normal", args.workers)
    new_d = evaluate(best, dark_yaml, "new_dark", args.workers)

    onnx = YOLO(str(best)).export(format="onnx", imgsz=640)
    dst = Path.home() / "Downloads" / "best_zone_dark.onnx"
    shutil.copy(onnx, dst)

    print("\n================ RESULT (val clips: edges_trace, led_on_center) ================")
    print(f"{'model':16s} {'normal box50':>12s} {'normal pose50':>14s} | {'DARK box50':>11s} {'DARK pose50':>12s}")
    for tag, (bn, pn, _), (bd, pd, _) in (("zone_ft_mix (old)", base_n, base_d), ("zone_dark (new)", new_n, new_d)):
        print(f"{tag:16s} {bn:12.3f} {pn:14.3f} | {bd:11.3f} {pd:12.3f}")
    print(f"\nbest: {best}\nonnx: {dst}")


if __name__ == "__main__":
    main()
