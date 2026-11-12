# v1 prototype (archived)

The first version of the glove, kept for reference. It is not used by the current system.

- Perfboard electronics wired to the Pi with jumper wires
- One large ArUco marker defined the work area
- Hand tracking used YOLOv8n-pose only (no fingertip LEDs)
- Separate ROS2 nodes for finger tracking, zone checking and the IMU

| Folder | Contents |
|---|---|
| `pi/vision/` | v1 live scripts: YOLO hand tracking, single-marker zone, first integrated safety script |
| `pi/dataset/` | v1 dataset recording and frame extraction |
| `ros2_ws/src/` | v1 ROS2 nodes: `finger_tracker`, `zone_manager`, `probe_tracker`, `haptic_launch` |

The v1 training scripts, results and INT8 quantization work are in [`training/v1/`](../../training/v1), and the v1 weights are in [`weights/v1/`](../../weights/v1).
