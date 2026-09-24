import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import smbus2
import time

TCA9548A_ADDR = 0x70
MPU6050_ADDR  = 0x68
MPU_CHANNEL   = 5  # TCA9548A 채널 5

class MPU6050Node(Node):
    def __init__(self):
        super().__init__('mpu6050')
        self.publisher = self.create_publisher(Vector3, '/probe_pose', 10)
        self.bus = smbus2.SMBus(1)

        # TCA 채널 5 활성화 후 MPU 깨우기
        self.select_channel()
        self.bus.write_byte_data(MPU6050_ADDR, 0x6B, 0)
        time.sleep(0.1)

        self.timer = self.create_timer(0.05, self.read_imu)
        self.get_logger().info('MPU6050 Node started!')

    def select_channel(self):
        self.bus.write_byte(TCA9548A_ADDR, 1 << MPU_CHANNEL)
        time.sleep(0.01)

    def read_imu(self):
        try:
            self.select_channel()
            data = self.bus.read_i2c_block_data(MPU6050_ADDR, 0x3B, 6)
            ax = (data[0] << 8 | data[1]) / 16384.0
            ay = (data[2] << 8 | data[3]) / 16384.0
            az = (data[4] << 8 | data[5]) / 16384.0

            msg = Vector3()
            msg.x = ax
            msg.y = ay
            msg.z = az
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'IMU 읽기 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MPU6050Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
