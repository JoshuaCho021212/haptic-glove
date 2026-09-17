# Haptic Glove Project

Custom hand-pose detection pipeline for a haptic feedback glove (Raspberry Pi 4).

## Pipeline
1. Recorded gloved-hand video clips on the Pi
2. Labeled 21-point hand keypoints using **self-hosted CVAT v2.71.0** (official repo, default docker-compose setup — no custom config)
3. Exported labels in COCO Keypoints 1.0 format
4. \	raining/build_yolo_dataset.py\ — converts CVAT exports into YOLOv8-pose dataset format
5. Trained YOLOv8n-pose on the dataset (see \	raining/data.yaml\)
6. \	raining/plot_yolo_training_results*.py\ — visualizes training metrics

## Notes
- Labeling/preprocessing scripts (auto_label, frame extraction) currently live on a separate laptop — to be migrated
- Deployed model runs as ONNX (320x320 or 640x640) on the Pi via onnxruntime
