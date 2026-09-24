from setuptools import find_packages, setup
package_name = 'probe_tracker'
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joshua',
    maintainer_email='joshua@todo.todo',
    description='MPU6050 IMU node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mpu6050 = probe_tracker.mpu6050_publisher:main',
        ],
    },
)
