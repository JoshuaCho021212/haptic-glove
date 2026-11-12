from setuptools import find_packages, setup

package_name = 'finger_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joshuacho',
    maintainer_email='joshuacho@todo.todo',
    description='Finger tracking with MediaPipe',
    license='MIT',
    entry_points={
        'console_scripts': [
            'finger_tracker = finger_tracker.finger_tracker_node:main',
        ],
    },
)
