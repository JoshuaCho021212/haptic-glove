import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point
import smbus2
import time
import threading
import math
import gpiod

TCA9548A_ADDR = 0x70
DRV2605L_ADDR = 0x5A
MPU6050_ADDR  = 0x68

REG_MODE      = 0x01
REG_LIBRARY   = 0x03
REG_FEEDBACK  = 0x1A
REG_CONTROL3  = 0x1D
REG_RTP       = 0x02
PWR_MGMT_1    = 0x6B
ACCEL_XOUT_H  = 0x3B

FINGER_NAMES = ['엄지', '검지', '중지', '약지', '새끼']
LED_PINS = [22, 27, 17, 23, 24]  # 엄지=GPIO22, 검지=27, 중지=17, 약지=23, 소지=24

TREMOR_WINDOW = 20
TREMOR_ACCEL_THRESHOLD = 0.15
SUDDEN_MOVE_THRESHOLD = 2.5

class HapticEncoder(Node):
    def __init__(self):
        super().__init__('haptic_encoder')
        self.bus = smbus2.SMBus(1)
        self.i2c_lock = threading.Lock()

        # LED init
        chip = gpiod.Chip('/dev/gpiochip0')
        config = gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT,
            output_value=gpiod.line.Value.INACTIVE
        )
        self.led_request = chip.request_lines(
            config={tuple(LED_PINS): config},
            consumer="haptic_led"
        )
        self.get_logger().info('LED 5개 초기화 완료')

        # DRV init (channels 0-4)
        for ch in range(5):
            try:
                self._select_channel_raw(ch)
                self._init_drv2605l()
                self.get_logger().info(f'채널 {ch} DRV2605L 초기화 완료')
            except Exception as e:
                self.get_logger().warn(f'채널 {ch} 없음: {e}')

        # MPU init (channel 5)
        try:
            self._select_channel_raw(5)
            self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
            time.sleep(0.1)
            val = self.bus.read_byte_data(MPU6050_ADDR, 0x75)
            self.get_logger().info(f'MPU6050 초기화 완료 (WHO_AM_I: {hex(val)})')
        except Exception as e:
            self.get_logger().warn(f'MPU6050 초기화 실패: {e}')

        # Tremor state
        self.accel_history = []
        self.tremor_active = False
        self.sudden_move_active = False

        # Subscriptions
        self.subs = []
        for i in range(5):
            sub = self.create_subscription(
                Float32, f'/haptic_cmd_{i}',
                lambda msg, idx=i: self.encode_haptic(msg, idx), 10
            )
            self.subs.append(sub)

        # Publishers
        self.imu_pub = self.create_publisher(Point, '/probe_pose', 10)
        self.tremor_pub = self.create_publisher(Bool, '/tremor_alert', 10)
        self.sudden_pub = self.create_publisher(Bool, '/sudden_move_alert', 10)

        # IMU timer at 10Hz
        self.imu_timer = self.create_timer(0.1, self.publish_imu)

        self.get_logger().info('Haptic Encoder started!')

    def _select_channel_raw(self, channel):
        self.bus.write_byte(TCA9548A_ADDR, 1 << channel)
        time.sleep(0.02)

    def _init_drv2605l(self):
        self.bus.write_byte_data(DRV2605L_ADDR, REG_MODE, 0x05)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_FEEDBACK, 0x80)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_CONTROL3, 0xA0)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_LIBRARY, 0x02)
        self.bus.write_byte_data(DRV2605L_ADDR, REG_RTP, 0x00)

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(MPU6050_ADDR, reg)
        low  = self.bus.read_byte_data(MPU6050_ADDR, reg + 1)
        val  = (high << 8) + low
        if val >= 0x8000:
            val -= 65536
        return val

    def set_led(self, finger_idx, on):
        pin = LED_PINS[finger_idx]
        value = gpiod.line.Value.ACTIVE if on else gpiod.line.Value.INACTIVE
        self.led_request.set_value(pin, value)

    def detect_tremor(self, ax, ay, az):
        magnitude = math.sqrt(ax*ax + ay*ay + az*az)
        self.accel_history.append(magnitude)
        if len(self.accel_history) > TREMOR_WINDOW:
            self.accel_history.pop(0)
        if len(self.accel_history) < TREMOR_WINDOW:
            return False
        mean = sum(self.accel_history) / len(self.accel_history)
        variance = sum((x - mean)**2 for x in self.accel_history) / len(self.accel_history)
        return variance > TREMOR_ACCEL_THRESHOLD

    def detect_sudden_movement(self, ax, ay, az):
        magnitude = math.sqrt(ax*ax + ay*ay + az*az)
        dynamic_accel = abs(magnitude - 1.0)
        return dynamic_accel > SUDDEN_MOVE_THRESHOLD

    def publish_imu(self):
        acquired = self.i2c_lock.acquire(timeout=0.05)
        if not acquired:
            return
        try:
            self._select_channel_raw(5)
            ax = self._read_word_2c(ACCEL_XOUT_H)     / 16384.0
            ay = self._read_word_2c(ACCEL_XOUT_H + 2) / 16384.0
            az = self._read_word_2c(ACCEL_XOUT_H + 4) / 16384.0

            msg = Point()
            msg.x = ax
            msg.y = ay
            msg.z = az
            self.imu_pub.publish(msg)

            # Tremor
            tremor = self.detect_tremor(ax, ay, az)
            if tremor and not self.tremor_active:
                self.get_logger().warn('⚠️ 손 떨림 감지!')
                self.tremor_active = True
            elif not tremor and self.tremor_active:
                self.get_logger().info('손 떨림 해소')
                self.tremor_active = False
            tremor_msg = Bool()
            tremor_msg.data = tremor
            self.tremor_pub.publish(tremor_msg)

            # Sudden movement
            sudden = self.detect_sudden_movement(ax, ay, az)
            if sudden and not self.sudden_move_active:
                self.get_logger().warn('🚨 급격한 움직임 감지! 안전 정지!')
                self.sudden_move_active = True
            elif not sudden and self.sudden_move_active:
                self.get_logger().info('움직임 정상화')
                self.sudden_move_active = False
            sudden_msg = Bool()
            sudden_msg.data = sudden
            self.sudden_pub.publish(sudden_msg)

        except Exception as e:
            self.get_logger().warn(f'MPU 읽기 에러: {e}')
        finally:
            self.i2c_lock.release()

    def set_intensity(self, channel, intensity):
        with self.i2c_lock:
            self._select_channel_raw(channel)
            rtp_val = int(intensity * 127)
            self.bus.write_byte_data(DRV2605L_ADDR, REG_RTP, rtp_val)

    def encode_haptic(self, msg, finger_idx):
        intensity = max(0.0, min(1.0, msg.data))
        try:
            self.set_intensity(finger_idx, intensity)
            # LED on when motor active, off when idle
            self.set_led(finger_idx, intensity > 0.3)
        except Exception as e:
            self.get_logger().warn(f'I2C 에러: {e}')
        if intensity > 0.0:
            self.get_logger().info(
                f'{FINGER_NAMES[finger_idx]}: intensity={intensity:.2f}'
            )

    def destroy_node(self):
        for pin in LED_PINS:
            self.led_request.set_value(pin, gpiod.line.Value.INACTIVE)
        self.led_request.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HapticEncoder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
