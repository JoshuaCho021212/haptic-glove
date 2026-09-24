import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import smbus2

MPU6050_ADDR = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
TCA_ADDR     = 0x70

class MPU6050Publisher(Node):
    def __init__(self):
        super().__init__('mpu6050_publisher')
        self.publisher = self.create_publisher(Point, '/probe_pose', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)
        self.bus = smbus2.SMBus(1)
        self.bus.write_byte_data(TCA_ADDR, 0, 1 << 5)
        self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
        self.get_logger().info('MPU6050 Publisher started!')

    def read_word_2c(self, reg):
        high = self.bus.read_byte_data(MPU6050_ADDR, reg)
        low  = self.bus.read_byte_data(MPU6050_ADDR, reg + 1)
        val  = (high << 8) + low
        if val >= 0x8000:
            val -= 65536
        return val

    def publish_pose(self):
        self.bus.write_byte_data(TCA_ADDR, 0, 1 << 5)
        ax = self.read_word_2c(ACCEL_XOUT_H)     / 16384.0
        ay = self.read_word_2c(ACCEL_XOUT_H + 2) / 16384.0
        az = self.read_word_2c(ACCEL_XOUT_H + 4) / 16384.0
        msg = Point()
        msg.x = ax
        msg.y = ay
        msg.z = az
        self.publisher.publish(msg)
        self.get_logger().info(f'IMU: x={ax:.2f}g, y={ay:.2f}g, z={az:.2f}g')

def main(args=None):
    rclpy.init(args=args)
    node = MPU6050Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
