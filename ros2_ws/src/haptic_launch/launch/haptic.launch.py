from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='finger_tracker',
            executable='finger_tracker',
            name='finger_tracker',
            output='screen'
        ),
        Node(
            package='zone_manager',
            executable='zone_manager',
            name='zone_manager',
            output='screen'
        ),
        Node(
            package='haptic_encoder',
            executable='haptic_encoder',
            name='haptic_encoder',
            output='screen'
        ),
    ])
