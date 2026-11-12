import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Point
import math

class ZoneManager(Node):
    def __init__(self):
        super().__init__('zone_manager')

        self.subscription = self.create_subscription(
            Float32MultiArray, '/finger_positions', self.check_zone, 10)
        self.zone_sub = self.create_subscription(
            Point, '/safe_zone', self.update_zone, 10)

        self.haptic_pubs = [
            self.create_publisher(Float32, f'/haptic_cmd_{i}', 10)
            for i in range(5)
        ]

        # 기본 safe zone 중심 (IR 감지 전까지)
        self.zone_cx = 0.5
        self.zone_cy = 0.5
        self.safe_radius = 0.3    # 이 거리 안은 진동 없음
        self.danger_radius = 0.6  # 이 거리 밖은 최대 진동

        self.get_logger().info('Zone Manager started!')

    def update_zone(self, msg):
        self.zone_cx = msg.x
        self.zone_cy = msg.y
        self.get_logger().info(f'Zone updated: ({self.zone_cx:.2f}, {self.zone_cy:.2f})')

    def check_zone(self, msg):
        data = msg.data
        if len(data) < 15:
            return

        for i in range(5):
            x = data[i*3]
            y = data[i*3 + 1]
            distance = math.sqrt((x - self.zone_cx)**2 + (y - self.zone_cy)**2)

            # 안에 있으면 진동 없음, 벗어날수록 강해짐
            if distance <= self.safe_radius:
                intensity = 0.0
            elif distance >= self.danger_radius:
                intensity = 1.0
            else:
                intensity = (distance - self.safe_radius) / (self.danger_radius - self.safe_radius)

            haptic_cmd = Float32()
            haptic_cmd.data = intensity
            self.haptic_pubs[i].publish(haptic_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ZoneManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
