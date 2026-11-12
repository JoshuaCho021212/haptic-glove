import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point
import smbus2
import time
import threading
import math
import collections
import gpiod

TCA9548A_ADDR = 0x70
DRV2605L_ADDR = 0x5A
MPU6050_ADDR = 0x68

REG_MODE = 0x01
REG_LIBRARY = 0x03
REG_RTP = 0x02
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
LED_PINS = [22, 17, 27, 23, 24]
BUZZER_PIN = 18
BUZZER_FREQ = 2000

IMU_HZ = 100.0
TREMOR_BAND = (4.0, 12.0)
TREMOR_WINDOW_SEC = 2.0
TREMOR_PUB_HZ = 10.0
TREMOR_ALERT_DPS = 6.0
SUDDEN_MOVE_THRESHOLD = 2.5
MIN_INTENSITY = 0.5
MUX_SETTLE_SEC = 0.0005
PATTERN_HZ = 100.0
PULSE_ON_SEC = 0.1
PULSE_PERIOD_SEC = {1: 0.9, 2: 0.6, 3: 0.4, 4: 0.22}
HEARTBEAT_TIMEOUT_SEC = 1.0
STARTUP_GRACE_SEC = 30.0
LOST_PATTERN_SEC = 1.5
LOST_PULSES = ((0.0, 0.12), (0.25, 0.37))
LOST_AMPLITUDE = 0.8


class Biquad:
    def __init__(self, fs, f0, q):
        w0 = 2 * math.pi * f0 / fs
        alpha = math.sin(w0) / (2 * q)
        a0 = 1 + alpha
        self.b = (alpha / a0, 0.0, -alpha / a0)
        self.a = (-2 * math.cos(w0) / a0, (1 - alpha) / a0)
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def __call__(self, x):
        y = self.b[0] * x + self.b[1] * self.x1 + self.b[2] * self.x2 - self.a[0] * self.y1 - self.a[1] * self.y2
        self.x2, self.x1, self.y2, self.y1 = self.x1, x, self.y1, y
        return y


class HapticEncoder(Node):
    def __init__(self):
        super().__init__('haptic_encoder')
        self.bus = smbus2.SMBus(1)
        self.i2c_lock = threading.Lock()
        self.cur_channel = None

        chip = gpiod.Chip('/dev/gpiochip0')
        self.led_request = chip.request_lines(
            config={tuple(LED_PINS): gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT,
                                                        output_value=gpiod.line.Value.ACTIVE)},
            consumer="haptic_led")
        self.buzzer_request = chip.request_lines(
            config={(BUZZER_PIN,): gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT,
                                                      output_value=gpiod.line.Value.INACTIVE)},
            consumer="haptic_buzzer")
        self.get_logger().info('LEDs + buzzer ready')

        for ch in range(5):
            try:
                self._select_channel(ch, settle=0.02)
                self._init_drv2605l()
                self.get_logger().info(f'channel {ch} DRV2605L ready')
            except Exception as e:
                self.get_logger().warn(f'channel {ch} missing: {e}')

        self.imu_ok = False
        try:
            self._select_channel(5, settle=0.02)
            self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
            time.sleep(0.1)
            self.bus.write_byte_data(MPU6050_ADDR, SMPLRT_DIV, int(1000 / IMU_HZ) - 1)
            self.bus.write_byte_data(MPU6050_ADDR, CONFIG, 3)
            self.bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0)
            self.bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, 0)
            val = self.bus.read_byte_data(MPU6050_ADDR, 0x75)
            self.imu_ok = True
            self.get_logger().info(f'MPU6050 ready (WHO_AM_I: {hex(val)})')
        except Exception as e:
            self.get_logger().warn(f'MPU6050 init failed: {e}')

        self.intensity = [0.0] * 5
        self.level = [0] * 5
        self.level_t0 = [0.0] * 5
        self.rtp_out = [None] * 5
        self.start_t = time.monotonic()
        self.last_hb = None
        self.lost = False
        self.lost_t0 = 0.0
        self.gain = [1.0] * 5
        self.paused = False
        self.buzzer_enabled = True
        self.sudden_move_active = False
        self.zone_exit_active = False
        self.tremor_rms = 0.0
        self.last_pose = (0.0, 0.0, 0.0)

        self.running = True
        self.buzzer_thread = threading.Thread(target=self._buzzer_loop, daemon=True)
        self.buzzer_thread.start()

        self.subs = []
        for i in range(5):
            self.subs.append(self.create_subscription(
                Float32, f'/haptic_cmd_{i}', lambda msg, idx=i: self.encode_haptic(msg, idx), 10))
            self.subs.append(self.create_subscription(
                Float32, f'/haptic_gain_{i}', lambda msg, idx=i: self.on_gain(msg, idx), 10))
        self.subs.append(self.create_subscription(Bool, '/zone_exit', self.on_zone_exit, 10))
        self.subs.append(self.create_subscription(Bool, '/buzzer_enable', self.on_buzzer_enable, 10))
        self.subs.append(self.create_subscription(Bool, '/glove_pause', self.on_pause, 10))
        self.subs.append(self.create_subscription(Bool, '/vision_heartbeat', self.on_heartbeat, 10))
        self.lost_pub = self.create_publisher(Bool, '/vision_lost', 10)

        self.imu_pub = self.create_publisher(Point, '/probe_pose', 10)
        self.tremor_pub = self.create_publisher(Bool, '/tremor_alert', 10)
        self.tremor_rms_pub = self.create_publisher(Float32, '/tremor_rms', 10)
        self.sudden_pub = self.create_publisher(Bool, '/sudden_move_alert', 10)
        self.create_timer(1.0 / TREMOR_PUB_HZ, self.publish_status)

        self.imu_thread = threading.Thread(target=self._imu_loop, daemon=True)
        self.imu_thread.start()
        self.pattern_thread = threading.Thread(target=self._pattern_loop, daemon=True)
        self.pattern_thread.start()
        self.get_logger().info('Haptic Encoder started!')

    def _select_channel(self, channel, settle=MUX_SETTLE_SEC):
        if channel != self.cur_channel:
            self.bus.write_byte(TCA9548A_ADDR, 1 << channel)
            self.cur_channel = channel
            time.sleep(settle)

    def _init_drv2605l(self):
        self.bus.write_byte_data(DRV2605L_ADDR, REG_MODE, 0x80)
        time.sleep(0.01)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_MODE, 0x00)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_LIBRARY, 0x01)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_RTP, 0x00)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_MODE, 0x05)

    def _tone(self, duration):
        half = 0.5 / BUZZER_FREQ
        end = time.monotonic() + duration
        while time.monotonic() < end and self.running:
            self.buzzer_request.set_value(BUZZER_PIN, gpiod.line.Value.ACTIVE)
            time.sleep(half)
            self.buzzer_request.set_value(BUZZER_PIN, gpiod.line.Value.INACTIVE)
            time.sleep(half)

    def _buzzer_loop(self):
        while self.running:
            if (self.buzzer_enabled and not self.paused
                    and (self.sudden_move_active or self.zone_exit_active)):
                self._tone(0.1)
                time.sleep(0.1)
            else:
                time.sleep(0.05)

    def _imu_loop(self):
        if not self.imu_ok:
            return
        fs = IMU_HZ
        f0 = math.sqrt(TREMOR_BAND[0] * TREMOR_BAND[1])
        q = f0 / (TREMOR_BAND[1] - TREMOR_BAND[0])
        filters = [[Biquad(fs, f0, q), Biquad(fs, f0, q)] for _ in range(3)]
        window = collections.deque(maxlen=int(TREMOR_WINDOW_SEC * fs))
        period = 1.0 / fs
        nxt = time.perf_counter()
        while self.running:
            nxt += period
            try:
                with self.i2c_lock:
                    self._select_channel(5)
                    d = self.bus.read_i2c_block_data(MPU6050_ADDR, ACCEL_XOUT_H, 14)
            except Exception as e:
                self.get_logger().warn(f'MPU read error: {e}')
                time.sleep(0.1)
                nxt = time.perf_counter()
                continue
            raw = [((d[i] << 8) | d[i + 1]) for i in range(0, 14, 2)]
            raw = [v - 65536 if v >= 0x8000 else v for v in raw]
            ax, ay, az = (v / 16384.0 for v in raw[0:3])
            gyro = [v / 131.0 for v in raw[4:7]]
            band = []
            for g, (f1, f2) in zip(gyro, filters):
                band.append(f2(f1(g)))
            window.append(band[0] ** 2 + band[1] ** 2 + band[2] ** 2)
            if len(window) == window.maxlen:
                self.tremor_rms = math.sqrt(sum(window) / len(window))
            self.last_pose = (ax, ay, az)
            mag = math.sqrt(ax * ax + ay * ay + az * az)
            self.sudden_move_active = abs(mag - 1.0) > SUDDEN_MOVE_THRESHOLD
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def publish_status(self):
        ax, ay, az = self.last_pose
        self.imu_pub.publish(Point(x=ax, y=ay, z=az))
        self.tremor_rms_pub.publish(Float32(data=float(self.tremor_rms)))
        self.tremor_pub.publish(Bool(data=self.tremor_rms > TREMOR_ALERT_DPS))
        self.sudden_pub.publish(Bool(data=self.sudden_move_active))

    def on_zone_exit(self, msg):
        if msg.data and not self.zone_exit_active:
            self.get_logger().warn('ZONE EXIT')
        self.zone_exit_active = msg.data

    def on_buzzer_enable(self, msg):
        self.buzzer_enabled = bool(msg.data)

    def on_pause(self, msg):
        if bool(msg.data) != self.paused:
            self.paused = bool(msg.data)
            self.get_logger().info('GLOVE PAUSED' if self.paused else 'GLOVE RESUMED')

    def on_heartbeat(self, msg):
        self.last_hb = time.monotonic()

    def on_gain(self, msg, idx):
        self.gain[idx] = max(0.0, min(1.0, msg.data))

    def encode_haptic(self, msg, finger_idx):
        i = max(0.0, min(1.0, msg.data))
        lv = int(round(i * 5))
        now = time.monotonic()
        if lv > self.level[finger_idx]:
            self.level_t0[finger_idx] = now
        self.intensity[finger_idx] = i
        self.level[finger_idx] = lv

    def _vision_lost(self, now):
        if self.last_hb is None:
            return now - self.start_t > STARTUP_GRACE_SEC
        return now - self.last_hb > HEARTBEAT_TIMEOUT_SEC

    def _amplitude(self, ch, now):
        if self.paused:
            return 0.0
        if self.lost:
            ph = (now - self.lost_t0) % LOST_PATTERN_SEC
            return LOST_AMPLITUDE if any(a <= ph < b for a, b in LOST_PULSES) else 0.0
        lv = self.level[ch]
        if lv <= 0:
            return 0.0
        amp = MIN_INTENSITY + (1.0 - MIN_INTENSITY) * self.intensity[ch]
        if lv >= 5:
            return 1.0
        ph = (now - self.level_t0[ch]) % PULSE_PERIOD_SEC[lv]
        return amp if ph < PULSE_ON_SEC else 0.0

    def _pattern_loop(self):
        period = 1.0 / PATTERN_HZ
        while self.running:
            now = time.monotonic()
            lost = self._vision_lost(now)
            if lost != self.lost:
                self.lost = lost
                self.lost_t0 = now
                self.lost_pub.publish(Bool(data=lost))
                if lost:
                    self.get_logger().warn('VISION LOST - tracking alert on glove')
                else:
                    self.get_logger().info('vision heartbeat back')
                    self.level = [0] * 5
            for ch in range(5):
                amp = self._amplitude(ch, now)
                gain = 1.0 if self.lost else self.gain[ch]
                rtp = int(gain * amp * 127) if amp > 0 and gain > 0 else 0
                if rtp != self.rtp_out[ch]:
                    self._write_rtp(ch, rtp)
            time.sleep(period)

    def _write_rtp(self, channel, rtp):
        try:
            with self.i2c_lock:
                self._select_channel(channel)
                self.bus.write_byte_data(DRV2605L_ADDR, REG_RTP, rtp)
            self.rtp_out[channel] = rtp
        except Exception as e:
            self.get_logger().warn(f'I2C error ch{channel}: {e}')
            time.sleep(0.05)

    def destroy_node(self):
        self.running = False
        self.buzzer_thread.join(timeout=1.0)
        self.pattern_thread.join(timeout=1.0)
        for ch in range(5):
            self._write_rtp(ch, 0)
        self.buzzer_request.set_value(BUZZER_PIN, gpiod.line.Value.INACTIVE)
        self.buzzer_request.release()
        for pin in LED_PINS:
            self.led_request.set_value(pin, gpiod.line.Value.INACTIVE)
        self.led_request.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HapticEncoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
