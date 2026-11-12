import argparse
import io
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

VAL_CLIPS = {"edges_trace", "led_on_center"}
NEW_OVERSAMPLE = 3


def clip_of(name):
    return Path(name).stem.rsplit("_", 1)[0]


def load_coco(exports_dir):
    anns = {}
    for z in sorted(Path(exports_dir).glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            js = [n for n in zf.namelist() if n.endswith(".json")]
            if not js:
                print(f"  skip {z.name}: no json")
                continue
            coco = json.loads(zf.read(js[0]))
        imgs = {im["id"]: im for im in coco["images"]}
        seen = set()
        for im in coco["images"]:
            anns.setdefault(Path(im["file_name"]).name, [])
        for a in coco["annotations"]:
            im = imgs[a["image_id"]]
            kp = a.get("keypoints", [])
            if len(kp) != 63:
                continue
            anns[Path(im["file_name"]).name].append((a["bbox"], kp, im["width"], im["height"]))
            seen.add(clip_of(im["file_name"]))
        print(f"  {z.name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations, clips={sorted(seen)}")
    return anns


def iter_crops(crops_dir):
    for p in Path(crops_dir).rglob("*.jpg"):
        yield p.name, p.read_bytes()
    for z in Path(crops_dir).rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".jpg"):
                    yield Path(n).name, zf.read(n)
                elif n.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(zf.read(n))) as inner:
                        for m in inner.namelist():
                            if m.lower().endswith(".jpg"):
                                yield Path(m).name, inner.read(m)


def yolo_line(bbox, kp, w, h):
    x, y, bw, bh = bbox
    vals = [0, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
    for i in range(21):
        px, py, v = kp[3 * i: 3 * i + 3]
        if v == 0:
            vals += [0.0, 0.0, 0]
        else:
            vals += [min(max(px / w, 0), 1), min(max(py / h, 0), 1), int(v)]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True, help="folder with CVAT COCO Keypoints export zips")
    ap.add_argument("--crops", required=True, help="folder with the clip zips / crop jpgs")
    ap.add_argument("--negatives", default=None, help="folder with no_hand jpgs")
    ap.add_argument("--old", default=r"C:\Users\j8cho\yolo_dataset")
    ap.add_argument("--weights", default=r"N:\hand_dataset\runs\pose\train-7\weights\best.pt")
    ap.add_argument("--out", default=r"C:\Users\j8cho\yolo_dataset_zone")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print("reading CVAT exports...")
    anns = load_coco(args.exports)
    labeled_clips = {clip_of(n) for n in anns}
    print(f"labeled clips: {sorted(labeled_clips)}")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    for name, data in iter_crops(args.crops):
        if name not in anns:
            continue
        split = "val" if clip_of(name) in VAL_CLIPS else "train"
        (out / "images" / split / name).write_bytes(data)
        lines = [yolo_line(*a) for a in anns[name]]
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
        counts[(split, "hand" if lines else "empty")] += 1

    if args.negatives:
        for p in sorted(Path(args.negatives).glob("*.jpg")):
            split = "val" if hash(p.name) % 5 == 0 else "train"
            shutil.copy(p, out / "images" / split / ("neg_" + p.name))
            (out / "labels" / split / ("neg_" + p.stem + ".txt")).write_text("")
            counts[(split, "negative")] += 1
    print("new data:", dict(counts))

    new_train = sorted(str(p) for p in (out / "images" / "train").glob("*.jpg"))
    old_train = sorted(str(p) for p in (Path(args.old) / "images" / "train").glob("*.jpg"))
    train_list = old_train + new_train * NEW_OVERSAMPLE
    random.seed(0)
    random.shuffle(train_list)
    (out / "train_list.txt").write_text("\n".join(train_list))
    val_new = sorted(str(p) for p in (out / "images" / "val").glob("*.jpg"))
    (out / "val_list.txt").write_text("\n".join(val_new))
    print(f"train: {len(old_train)} old + {len(new_train)} new x{NEW_OVERSAMPLE} = {len(train_list)} | val (new clips): {len(val_new)}")

    yaml_path = out / "data_zone.yaml"
    yaml_path.write_text(
        f"path: {out.as_posix()}\n"
        f"train: {(out / 'train_list.txt').as_posix()}\n"
        f"val: {(out / 'val_list.txt').as_posix()}\n\n"
        "kpt_shape: [21, 3]\n"
        "flip_idx: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]\n\n"
        "names:\n  0: hand\n"
    )

    print("\n=== baseline: old model on new val clips ===")
    old_m = YOLO(args.weights).val(data=str(yaml_path), imgsz=640, batch=args.batch, workers=args.workers,
                                    name="zone_val_old", exist_ok=True, plots=False)

    model = YOLO(args.weights)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        degrees=30.0,
        scale=0.5,
        translate=0.15,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        patience=30,
        name="zone_ft",
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    new_m = YOLO(str(best)).val(data=str(yaml_path), imgsz=640, batch=args.batch, workers=args.workers,
                                 name="zone_val_new", exist_ok=True, plots=True)

    onnx = YOLO(str(best)).export(format="onnx", imgsz=640)
    dst = Path(__file__).resolve().parent / "best_zone.onnx"
    shutil.copy(onnx, dst)

    print("\n=========== RESULT (new val clips: %s) ===========" % ", ".join(sorted(VAL_CLIPS)))
    print(f"{'model':6s} {'pose mAP50':>11s} {'pose mAP50-95':>14s} {'box mAP50':>10s}")
    for tag, m in (("old", old_m), ("new", new_m)):
        print(f"{tag:6s} {m.pose.map50:11.3f} {m.pose.map:14.3f} {m.box.map50:10.3f}")
    print(f"\nbest weights: {best}\nonnx: {dst}")


if __name__ == "__main__":
    main()
