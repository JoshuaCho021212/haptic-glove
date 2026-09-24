from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'haptic_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joshuacho',
    maintainer_email='joshuacho@todo.todo',
    description='Haptic glove launch package',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
