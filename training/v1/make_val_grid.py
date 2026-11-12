import os, glob, argparse, tempfile
from collections import Counter
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

DATA = r"C:\Users\j8cho\yolo_dataset\images\val"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS, TILE_W, N_CAND = 3, 400, 24


def load_pool():
    imgs = sorted(glob.glob(os.path.join(DATA, "*.*")))
    gloved = [p for p in imgs if os.path.basename(p).startswith("JOB")]
    pool = gloved if gloved else imgs
    print(f"val images: {len(imgs)}, gloved: {len(gloved)}")
    for job, n in sorted(Counter(os.path.basename(p).split("_frame")[0] for p in pool).items()):
        print(f"  {job}: {n} frames")
    return pool


def candidates(pool):
    step = len(pool) / N_CAND
    return [pool[int(i * step + step / 2)] for i in range(N_CAND)]


def to_tile(im):
    return im.resize((TILE_W, int(im.height * TILE_W / im.width)))


def make_grid(tiles, cols):
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, t in enumerate(tiles):
        grid.paste(t.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    return grid


ap = argparse.ArgumentParser(description="Build a prediction grid from held-out validation frames")
ap.add_argument("--preview", action="store_true", help="save numbered candidate frames to pick from")
ap.add_argument("--picks", default="", help="comma-separated candidate numbers, e.g. 0,5,9,13,17,22")
args = ap.parse_args()

pool = load_pool()
cands = candidates(pool)

if args.preview:
    font = ImageFont.load_default(size=36)
    tiles = []
    for i, p in enumerate(cands):
        t = to_tile(Image.open(p).convert("RGB"))
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 64, 46], fill="black")
        d.text((8, 2), str(i), fill="yellow", font=font)
        tiles.append(t)
        print(f"{i:2d}: {os.path.basename(p)}")
    out = os.path.join(tempfile.gettempdir(), "val_candidates.jpg")
    make_grid(tiles, 6).save(out, quality=85)
    print("saved:", out)
else:
    idx = [int(x) for x in args.picks.split(",")] if args.picks else [2, 6, 10, 14, 18, 22]
    picks = [cands[i] for i in idx]
    for p in picks:
        print("pick:", os.path.basename(p))

    model = YOLO(os.path.join(REPO, "weights", "best.pt"))
    results = model.predict(picks, imgsz=640, conf=0.25, verbose=False)
    tiles = [to_tile(Image.fromarray(r.plot()[:, :, ::-1])) for r in results]

    out = os.path.join(REPO, "results", "val_predictions_grid.jpg")
    make_grid(tiles, COLS).save(out, quality=88)
    print("saved:", out)
