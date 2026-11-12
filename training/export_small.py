import shutil
from pathlib import Path
from ultralytics import YOLO


def main():
    best = Path(r"C:\Users\j8cho\runs\pose\zone_dark\weights\best.pt")
    data = r"C:\Users\j8cho\yolo_dataset_zone\data_zone_only.yaml"
    dark = r"C:\Users\j8cho\yolo_dataset_zone_darkval\data_dark.yaml"
    out_dir = Path.home() / "Downloads"
    rows = []
    for size in (640, 416, 320):
        m = YOLO(str(best))
        r = m.val(data=data, imgsz=size, workers=0, plots=False, verbose=False)
        rd = None
        if Path(dark).exists():
            rd = YOLO(str(best)).val(data=dark, imgsz=size, workers=0, plots=False, verbose=False)
        rows.append((size, r.box.map50, r.pose.map50,
                     rd.box.map50 if rd else float("nan"), rd.pose.map50 if rd else float("nan")))
        if size != 640:
            onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=size))
            dst = out_dir / f"best_zone_dark_{size}.onnx"
            shutil.copy(onnx, dst)
            print(f"exported {dst}")
    print("\n================ RESULT (zone_dark by input size) ================")
    print(f"{'imgsz':>6} {'box50':>7} {'pose50':>7} | {'DARK box50':>10} {'DARK pose50':>11}")
    for s, b, p, db, dp in rows:
        print(f"{s:>6} {b:>7.3f} {p:>7.3f} | {db:>10.3f} {dp:>11.3f}")


if __name__ == "__main__":
    main()
