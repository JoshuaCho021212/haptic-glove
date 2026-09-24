# Haptic Glove

A wearable haptic feedback glove that tracks the user's hand in real time and delivers per-finger vibration cues, with built-in safety monitoring. Runs entirely on a Raspberry Pi 4 with ROS2.

<!-- TODO: add a photo or GIF of the glove here -->
<!-- ![Haptic glove](docs/glove.gif) -->

## Features

- **Per-finger haptic feedback**: 5 ERM vibration motors (thumb to pinky) driven through DRV2605L drivers in real-time playback mode
- **Custom gloved-hand tracking**: MediaPipe fails on the gloved hand because wires and the perfboard break the bare-hand silhouette, so I trained a custom **YOLOv8n-pose** model (21 keypoints) and deployed it on the Pi with ONNX Runtime
- **Safety monitoring**: IMU-based tremor detection (`/tremor_alert`) and a sudden-movement safety stop (`/sudden_move_alert`)
- **Marker-defined work zone**: an ArUco marker defines the treatment/work area, and the camera uses it to set the safety zone

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi 4 (4GB), Ubuntu Server 24.04, ROS2 Jazzy |
| Haptics | 5x DRV2605L (0x5A) + ERM motors, one per finger |
| I2C mux | TCA9548A (0x70): channels 0-4 for the haptic drivers, channel 5 for the IMU |
| IMU | MPU-6050 (0x68) |
| Camera | Raspberry Pi NoIR camera (imx708) |
| Enclosure | TPU finger caps with motor and LED pockets, SolidWorks forearm shell for the electronics |

All five haptic drivers share one I2C address, so they sit behind the TCA9548A multiplexer. IMU reads are consolidated into the haptic encoder node and guarded with a `threading.Lock()` to avoid bus conflicts.

## System Pipeline

```
Camera   -> finger_tracker -> zone_manager -> haptic_encoder -> DRV2605L -> ERM motors
                                                   ^
MPU-6050 -> probe_tracker (IMU) -------------------+-> tremor / sudden-movement alerts
```

| Package | Role |
|---|---|
| `finger_tracker` | Detects finger positions from the camera feed |
| `zone_manager` | Checks each finger against the defined zone |
| `haptic_encoder` | Maps zone state to motor intensity, reads the IMU, raises safety alerts |
| `probe_tracker` | MPU-6050 IMU publisher |
| `haptic_launch` | Launches the full pipeline |

## Repository Structure

```
ros2_ws/src/   ROS2 packages (see table above)
pi/vision/     Standalone Pi scripts: YOLO hand tracking, ArUco zone detection, integrated safety
pi/dataset/    Dataset recording and frame extraction
pi/tools/      Hardware test utilities (LEDs)
training/      YOLO dataset building and training plots
weights/       Trained model (best.pt, best.onnx)
results/       Training metrics and validation images
```

## Running

```bash
# ROS2 pipeline
cd ros2_ws
colcon build
source install/setup.bash
ros2 launch haptic_launch haptic.launch.py

# Standalone YOLO hand tracking (live stream on port 5004)
cp weights/best.onnx ~/best.onnx
python3 pi/vision/yolo_hand_live.py
```

## Hand Tracking Model

### Data pipeline

1. Recorded 18 gloved-hand clips on the Pi with a fixed overhead camera (varied angles, lighting, speed, and poses including fist, grab, and pinch)
2. Labeled 21-point hand keypoints (MediaPipe topology) in **self-hosted CVAT v2.71.0** using keyframes and interpolation
3. Exported labels in COCO Keypoints 1.0 format
4. `training/build_yolo_dataset.py` converts the exports into YOLOv8-pose format
5. Trained YOLOv8n-pose (see `training/data.yaml`)

### Results

YOLOv8n-pose trained on 3,332 frames for 150 epochs with HSV augmentation.

| Metric | Score |
|---|---|
| Pose mAP50-95 | 0.698 |
| Box mAP50-95 | 0.876 |

![Training results](results/results.png)
![Confusion matrix](results/confusion_matrix_normalized.png)

**Validation: ground truth vs predictions**

![Ground truth](results/val_batch0_labels.jpg)
![Predictions](results/val_batch0_pred.jpg)

### Deployment notes

- Deployed as a **640x640 ONNX** model. The 320x320 export produced very low confidence, so it is not used.
- `onnxruntime==1.18.1` is required; 1.28 causes a bus error on the Pi 4.
- A 2GB swap file is needed.
- Inference runs every 3rd frame for near-real-time performance.
- Do **not** apply NoIR color correction before inference. It creates a domain gap between training data and the live feed.
