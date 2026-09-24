import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import subprocess
import time
import threading
from flask import Flask, Response

MODEL_PATH = '/home/joshuacho/hand_landmarker.task'
FINGER_IDS = [4, 8, 12, 16, 20]
IR_THRESHOLD = 200
SAFE_ZONE_RADIUS = 80

flask_app = Flask(__name__)
latest_debug_frame = None
debug_lock = threading.Lock()

def gen_frames():
    while True:
        with debug_lock:
            if latest_debug_frame is None:
                time.sleep(0.03)
                continue
            frame = latest_debug_frame.copy()
        _, jpg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.03)

@flask_app.route('/')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

class FingerTracker(Node):
    def __init__(self):
        super().__init__('finger_tracker')
        self.publisher = self.create_publisher(Float32MultiArray, '/finger_positions', 10)
        self.zone_pub = self.create_publisher(Point, '/safe_zone', 10)
        self.proc = subprocess.Popen([
            'rpicam-vid', '-t', '0', '--inline', '--nopreview',
            '--codec', 'mjpeg', '--output', 'udp://127.0.0.1:8554',
            '--width', '320', '--height', '240', '--framerate', '30'
        ])
        time.sleep(2)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(
            'udp://127.0.0.1:8554?fifo_size=1000&overrun_nonfatal=1&flags=low_delay',
            cv2.CAP_FFMPEG
        )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.smooth_zone = None
        self.alpha = 0.1
        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()
        flask_thread = threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=5001, threaded=True), daemon=True)
        flask_thread.start()
        self.get_logger().info('Debug stream at http://10.37.86.18:5001')
        self.timer = self.create_timer(0.05, self.process_frame)
        self.get_logger().info('Finger Tracker started!')

    def _read_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame

    def detect_ir_zone(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, IR_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) > 50]
        if not valid:
            return None
        cx_list, cy_list = [], []
        h, w = frame.shape[:2]
        for c in valid:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx_list.append(M['m10'] / M['m00'] / w)
                cy_list.append(M['m01'] / M['m00'] / h)
        if not cx_list:
            return None
        return (sum(cx_list) / len(cx_list), sum(cy_list) / len(cy_list))

    def process_frame(self):
        global latest_debug_frame
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        zone = self.detect_ir_zone(frame)
        if zone:
            if self.smooth_zone is None:
                self.smooth_zone = zone
            else:
                self.smooth_zone = (
                    self.alpha * zone[0] + (1 - self.alpha) * self.smooth_zone[0],
                    self.alpha * zone[1] + (1 - self.alpha) * self.smooth_zone[1]
                )
            msg = Point()
            msg.x = self.smooth_zone[0]
            msg.y = self.smooth_zone[1]
            msg.z = 0.0
            self.zone_pub.publish(msg)
            self.get_logger().info(f'Safe zone: ({self.smooth_zone[0]:.2f}, {self.smooth_zone[1]:.2f})')

        b, g, r = cv2.split(frame)
        r = cv2.add(r, 60)
        b = cv2.subtract(b, 40)
        frame = cv2.merge([b, g, r])
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        debug_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw safe zone
        if self.smooth_zone:
            zx = int(self.smooth_zone[0] * w)
            zy = int(self.smooth_zone[1] * h)
            cv2.circle(debug_frame, (zx, zy), SAFE_ZONE_RADIUS, (0, 255, 0), 2)
            cv2.putText(debug_frame, 'SAFE ZONE', (zx - 40, zy - SAFE_ZONE_RADIUS - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            msg = Float32MultiArray()
            data = []
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (0, 255, 0)
                if i in FINGER_IDS:
                    idx = FINGER_IDS.index(i)
                    data.extend([lm.x, lm.y, lm.z])
                    # Check if finger is outside safe zone
                    if self.smooth_zone:
                        zx = int(self.smooth_zone[0] * w)
                        zy = int(self.smooth_zone[1] * h)
                        dist = ((cx - zx)**2 + (cy - zy)**2)**0.5
                        if dist > SAFE_ZONE_RADIUS:
                            color = (0, 0, 255)  # Red = outside
                            cv2.line(debug_frame, (cx, cy), (zx, zy), (0, 0, 255), 1)
                        else:
                            color = (0, 255, 0)  # Green = inside
                cv2.circle(debug_frame, (cx, cy), 4, color, -1)
            msg.data = data
            self.publisher.publish(msg)
            cv2.putText(debug_frame, 'DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(debug_frame, 'NO HAND', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        with debug_lock:
            latest_debug_frame = debug_frame

    def destroy_node(self):
        self.cap.release()
        self.proc.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FingerTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
