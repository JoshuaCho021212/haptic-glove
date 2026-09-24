import cv2
import os

def extract(video_path, output_dir, every_n=3):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            cv2.imwrite(os.path.join(output_dir, f'frame_{saved:04d}.jpg'), frame)
            saved += 1
        count += 1
    cap.release()
    print(f'{video_path}: {saved} frames saved to {output_dir}')

VIDEOS_DIR = '/home/joshuacho/hand_dataset/videos'

for fname in sorted(os.listdir(VIDEOS_DIR)):
    if not fname.endswith('.mp4'):
        continue

    tag = fname.rsplit('_', 2)[0]  # strip the _YYYYMMDD_HHMMSS timestamp
    video_path = os.path.join(VIDEOS_DIR, fname)

    # route bare-hand clips into bare/, everything else into gloved/
    category = 'bare' if tag.startswith('bare') else 'gloved'
    out_dir = f'/home/joshuacho/hand_dataset/{category}/frames/{tag}'

    extract(video_path, out_dir)
