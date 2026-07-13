from setuptools import setup
import os
from glob import glob

package_name = 'tracer_nav2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='er',
    maintainer_email='weijun.xie@elephantrobotics.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                             'vel_filter_v3_node=tracer_nav2.vel_filter_v3_node:main',
                             'camera_node = tracer_nav2.camera_node:main',
                             'aruco_align_node = tracer_nav2.aruco_align_node:main',
                             'mission_brain_node = tracer_nav2.mission_brain_node:main',
                             'auto_initial_pose_node = tracer_nav2.auto_initial_pose_node:main',
                             'agv_align_bridge_node = tracer_nav2.agv_align_bridge_node:main',
                             'nav2_route_runner_node = tracer_nav2.nav2_route_runner_node:main',
                             'vel_filter_node = tracer_nav2.vel_filter_node:main',
        ],
    },
)
