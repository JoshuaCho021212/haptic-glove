"""
Combine:
  - 12 CVAT COCO-Keypoints exports (gloved hand, manually labeled)
  - bare_labels.json (bare hand, MediaPipe auto-labeled)
into one YOLOv8-pose dataset.
"""

import json
import os
import shutil
import random
import cv2

# ---------------- CONFIG ----------------
CVAT_EXPORTS_DIR = r"N:\hand_dataset\cvat_exports_v2"
BARE_FRAMES_DIR = r"N:\hand_dataset\bare\frames"
BARE_LABELS_JSON = r"N:\hand_dataset\bare_labels.json"
OUTPUT_DIR = r"C:\Users\j8cho\yolo_dataset"

NUM_KEYPOINTS = 21
BARE_VAL_FRACTION = 0.1
BBOX_PADDING_FRAC = 0.15
RANDOM_SEED = 42
USE_BARE = False
# -----------------------------------------

random.seed(RANDOM_SEED)


def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


def bbox_from_keypoints(xs, ys, img_w, img_h):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * BBOX_PADDING_FRAC
    pad_y = h * BBOX_PADDING_FRAC
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img_w, x_max + pad_x)
    y_max = min(img_h, y_max + pad_y)
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    return cx, cy, bw, bh


def write_yolo_label(path, cx, cy, bw, bh, kpts_norm):
    parts = [0, cx, cy, bw, bh]
    line = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in parts)
    for x, y, v in kpts_norm:
        line += f" {x:.6f} {y:.6f} {int(v)}"
    with open(path, "a") as f:
        f.write(line + "\n")


def process_cvat_exports():
    job_dirs = [d for d in os.listdir(CVAT_EXPORTS_DIR) if d.startswith("JOB #")]
    print(f"Found {len(job_dirs)} CVAT job folders")

    total_written = 0
    for job in sorted(job_dirs):
        job_path = os.path.join(CVAT_EXPORTS_DIR, job)
        ann_dir = os.path.join(job_path, "annotations")
        json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        if not json_files:
            print(f"  [skip] {job}: no json found")
            continue
        json_path = os.path.join(ann_dir, json_files[0])

        if "Train" in json_files[0]:
            split = "train"
        elif "Validation" in json_files[0]:
            split = "val"
        else:
            split = "train"

        with open(json_path) as f:
            data = json.load(f)

        images_by_id = {img["id"]: img for img in data["images"]}
        job_slug = job.replace(" ", "_").replace("#", "")

        for ann in data["annotations"]:
            img_info = images_by_id.get(ann["image_id"])
            if img_info is None:
                continue
            img_w, img_h = img_info["width"], img_info["height"]
            file_name = img_info["file_name"]

            subset_folder = "Train" if split == "train" else "Validation"
            src_img_path = os.path.join(job_path, "images", subset_folder, file_name)
            if not os.path.exists(src_img_path):
                found = None
                for root, _, files in os.walk(os.path.join(job_path, "images")):
                    if file_name in files:
                        found = os.path.join(root, file_name)
                        break
                if found is None:
                    print(f"  [warn] image not found: {job}/{file_name}")
                    continue
                src_img_path = found

            kpts_flat = ann["keypoints"]
            if len(kpts_flat) != NUM_KEYPOINTS * 3:
                print(f"  [warn] unexpected keypoint count in {job}/{file_name}")
                continue

            xs, ys, kpts_norm = [], [], []
            for i in range(NUM_KEYPOINTS):
                x, y, v = kpts_flat[i * 3], kpts_flat[i * 3 + 1], kpts_flat[i * 3 + 2]
                if v > 0:
                    xs.append(x)
                    ys.append(y)
                kpts_norm.append((max(0, x / img_w), max(0, y / img_h), v))

            if not xs:
                continue

            cx, cy, bw, bh = bbox_from_keypoints(xs, ys, img_w, img_h)

            new_name = f"{job_slug}_{file_name}"
            dst_img_path = os.path.join(OUTPUT_DIR, "images", split, new_name)
            shutil.copy2(src_img_path, dst_img_path)

            label_name = os.path.splitext(new_name)[0] + ".txt"
            label_path = os.path.join(OUTPUT_DIR, "labels", split, label_name)
            write_yolo_label(label_path, cx, cy, bw, bh, kpts_norm)
            total_written += 1

    print(f"CVAT (gloved) frames written: {total_written}")


def process_bare_labels():
    with open(BARE_LABELS_JSON) as f:
        bare_labels = json.load(f)

    keys = list(bare_labels.keys())
    random.shuffle(keys)
    val_count = int(len(keys) * BARE_VAL_FRACTION)
    val_keys = set(keys[:val_count])

    total_written = 0
    for key, landmarks in bare_labels.items():
        src_img_path = os.path.join(BARE_FRAMES_DIR, key)
        if not os.path.exists(src_img_path):
            print(f"  [warn] bare image not found: {key}")
            continue

        img = cv2.imread(src_img_path)
        if img is None:
            print(f"  [warn] could not read: {key}")
            continue
        img_h, img_w = img.shape[:2]

        if len(landmarks) != NUM_KEYPOINTS:
            print(f"  [warn] unexpected landmark count for {key}")
            continue

        xs, ys, kpts_norm = [], [], []
        for lm in landmarks:
            x_norm, y_norm = lm["x"], lm["y"]
            x_px, y_px = x_norm * img_w, y_norm * img_h
            xs.append(x_px)
            ys.append(y_px)
            kpts_norm.append((x_norm, y_norm, 2))

        cx, cy, bw, bh = bbox_from_keypoints(xs, ys, img_w, img_h)

        split = "val" if key in val_keys else "train"
        safe_name = key.replace("/", "_").replace("\\", "_")
        dst_img_path = os.path.join(OUTPUT_DIR, "images", split, safe_name)
        shutil.copy2(src_img_path, dst_img_path)

        label_name = os.path.splitext(safe_name)[0] + ".txt"
        label_path = os.path.join(OUTPUT_DIR, "labels", split, label_name)
        write_yolo_label(label_path, cx, cy, bw, bh, kpts_norm)
        total_written += 1

    print(f"Bare-hand frames written: {total_written}")


def write_data_yaml():
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

kpt_shape: [{NUM_KEYPOINTS}, 3]
flip_idx: [{",".join(str(i) for i in range(NUM_KEYPOINTS))}]

names:
  0: hand
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'data.yaml')}")


if __name__ == "__main__":
    ensure_dirs()
    process_cvat_exports()
    if USE_BARE:
        process_bare_labels()
    write_data_yaml()

    for split in ["train", "val"]:
        n_img = len(os.listdir(os.path.join(OUTPUT_DIR, "images", split)))
        n_lbl = len(os.listdir(os.path.join(OUTPUT_DIR, "labels", split)))
        print(f"{split}: {n_img} images, {n_lbl} labels")
