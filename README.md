# Haptic Glove

A wearable haptic feedback glove that tracks the user's hand in real time and delivers per-finger vibration cues, with built-in safety monitoring. Runs entirely on a Raspberry Pi 4 with ROS2.

<p align="center"><img src="docs/glove_front.webp" width="380" alt="Haptic glove"></p>

## Features

- **Per-finger haptic feedback**: 5 ERM vibration motors (thumb to pinky) driven through DRV2605L drivers in real-time playback mode
- **Custom gloved-hand tracking**: MediaPipe fails on the gloved hand because wires and the perfboard break the bare-hand silhouette, so I trained a custom **YOLOv8n-pose** model (21 keypoints) and deployed it on the Pi with ONNX Runtime
- **Safety monitoring**: IMU-based tremor detection (`/tremor_alert`) and a sudden-movement safety stop (`/sudden_move_alert`)
- **Marker-defined work zone**: an ArUco marker defines the treatment/work area, and the camera uses it to set the safety zone

<p align="center"><img src="docs/zone_demo.webp" width="450" alt="Zone-based feedback demo"></p>
<p align="center"><em>Fingertip LEDs indicate each finger's state as the hand moves over the ArUco-marked work area.</em></p>

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

<p align="center"><img src="docs/glove_side.webp" width="320" alt="First hardware prototype"></p>
<p align="center"><em>First hardware prototype: all components connected with jumper wires, before soldering.</em></p>

## System Pipeline

```mermaid
flowchart LR
    CAM["NoIR camera"] --> FT["finger_tracker<br/>YOLOv8n-pose"]
    FT --> ZM["zone_manager"]
    ZM --> HE["haptic_encoder"]
    IMU["MPU-6050 IMU"] --> HE
    HE --> MUX["TCA9548A I2C mux"]
    MUX --> DRV["5x DRV2605L"]
    DRV --> ERM["ERM motors"]
    HE --> ALERT["Tremor / sudden-movement alerts"]
```

| Package | Role |
|---|---|
| `finger_tracker` | Detects finger positions from the camera feed |
| `zone_manager` | Checks each finger against the defined zone |
| `haptic_encoder` | Maps zone state to motor intensity, reads the IMU, raises safety alerts |
| `probe_tracker` | MPU-6050 IMU publisher |
| `haptic_launch` | Launches the full pipeline |

## Challenges & Solutions

| Challenge | Root cause | Solution |
|---|---|---|
| MediaPipe could not track the gloved hand | Wires, fingertip caps, and the perfboard break the bare-hand silhouette MediaPipe is trained on | Recorded 18 clips, labeled 21 keypoints in self-hosted CVAT, and trained a custom YOLOv8n-pose model on 3,332 frames |
| Five haptic drivers on one I2C bus | Every DRV2605L has the same fixed address (0x5A) | Placed each driver on its own TCA9548A multiplexer channel (0-4), with the IMU on channel 5 |
| Intermittent I2C read/write errors | The IMU and haptic drivers were accessed from separate nodes, so multiplexer channel switches interleaved | Consolidated IMU reads into the `haptic_encoder` node and guarded each channel switch and transaction with a `threading.Lock()` |
| Model crashed on the Pi 4 | onnxruntime 1.28 fails with a bus error on the Pi 4 | Pinned `onnxruntime==1.18.1` and added a 2GB swap file |
| Very low confidence from the 320x320 model | Input resolution far below the 640x640 training size | Exported at 640x640 and ran inference every 3rd frame to stay near real-time |
| INT8 dynamic quantization made the model slower (0.71x) | Dynamic quantization recomputes activation ranges every run and maps convolutions to less-optimized `ConvInteger` kernels, which suits Transformers, not CNNs | Switched to static quantization with a 100-image calibration set drawn from the training split |
| Static INT8 model dropped to 0 mAP | The YOLO output head packs box coordinates (0-640) and confidences (0-1) into one tensor, so 8-bit quantization rounded every confidence to zero | Kept the output head (`/model.22/`) in FP32 and quantized the rest of the network |
| Accuracy drop on the live feed | Color-correcting the NoIR camera output made live frames look different from the training data | Fed raw frames to the model, matching the training data |
| Risk of inflated validation scores | Neighboring video frames are nearly identical | Held out entire CVAT jobs (clips) for validation instead of random frames |

## Repository Structure

```
ros2_ws/src/   ROS2 packages (see table above)
pi/vision/     Standalone Pi scripts: YOLO hand tracking, ArUco zone detection, integrated safety
pi/dataset/    Dataset recording and frame extraction
pi/tools/      Hardware test utilities (LEDs)
training/      YOLO dataset building, training plots, and INT8 quantization scripts
weights/       Trained model (best.pt, best.onnx, best_int8.onnx)
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

YOLOv8n-pose trained on 3,332 frames for 150 epochs with HSV augmentation. Gloved-hand validation data is held out as entire CVAT jobs rather than random frames, so near-identical neighboring frames never appear in both the training and validation sets.

| Metric | Score |
|---|---|
| Pose mAP50-95 | 0.698 |
| Box mAP50-95 | 0.876 |

![Training results](results/results.png)
![Confusion matrix](results/confusion_matrix_normalized.png)

**Predictions on held-out validation frames** (selected to cover different poses and lighting)

![Validation predictions](results/val_predictions_grid.jpg)

### INT8 quantization for the Pi

The FP32 model runs at roughly 0.5-0.6 s per frame on the Pi 4, so the model was quantized to INT8 with post-training static quantization (`training/quantization/`).

| | FP32 | INT8 (static) |
|---|---|---|
| Model size | 13.9 MB | **7.3 MB** (-47%) |
| Latency on Raspberry Pi 4, run 1 | 627.6 ms | **491.2 ms** (-22%) |
| Latency on Raspberry Pi 4, run 2 | 518.1 ms | **381.5 ms** (-26%) |
| Pose mAP50 | 0.995 | **0.995** |
| Pose mAP50-95 | 0.703 | **0.700** |

Latency is measured with `onnxruntime==1.18.1` on the Pi CPU at 640x640 input. Accuracy is measured with Ultralytics validation on the held-out frames. The FP32/INT8 ratio held across two runs even though absolute timings varied with background load.

How it got there:

1. **Dynamic quantization** was tried first and made the model **slower** (0.71x on a desktop CPU), because it recomputes activation ranges on every run and suits Transformer matrix multiplies rather than convolutions.
2. **Static quantization** (QDQ, per-channel INT8 weights, calibrated on 100 training frames) collapsed accuracy to **0 mAP**. The output head packs box coordinates (0-640) and confidences (0-1) into one tensor, and a single 8-bit scale over that range rounds every confidence to zero.
3. Keeping the **output head (`/model.22/`) in FP32** and quantizing the rest restored full accuracy while keeping the speedup, since most of the compute is in the backbone.
4. The quantized model is saved with ONNX IR version 9 so it loads on the Pi's pinned `onnxruntime==1.18.1`.

On a desktop x86 CPU the two models ran at the same speed; the gain appears on the Pi's ARM CPU, which is where the model is deployed.

```bash
python training/quantization/quantize_int8.py --model weights/best.onnx --calib <calibration_images_dir> --out weights/best_int8.onnx
python training/quantization/benchmark.py weights/best.onnx weights/best_int8.onnx
```

### Deployment notes

- 640x640 ONNX model running on `onnxruntime==1.18.1` (FP32 `best.onnx` or INT8 `best_int8.onnx`)
- Inference every 3rd frame
- 2GB swap file required
- Raw (uncorrected) NoIR frames are fed to the model
