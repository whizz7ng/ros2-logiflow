from setuptools import setup
import os
from glob import glob

package_name = 'tracer'

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
        'console_scripts': [ 'tracer_node = tracer.tracer_node:main',
                             'vel_filter_node=tracer.vel_filter_node:main',
                             'rack_wall_debug_node = tracer.rack_wall_debug_node:main',
                             'goal_manager = tracer.goal_manager:main',
                             'debug_route_runner = tracer.debug_route_runner: main',
                             'nav_debug_logger = tracer.nav_debug_logger:main',
                             'camera_node = tracer.camera_node:main',
                             'mission_save_node = tracer.mission_save_node:main',
                             'aruco_align_node = tracer.aruco_align_node:main',
                             'primitive_route_runner = tracer.primitive_route_runner:main',
                             'mission_brain_node = tracer.mission_brain_node:main',
                             'auto_initial_pose_node = tracer.auto_initial_pose_node:main',
                             'agv_align_bridge_node = tracer.agv_align_bridge_node:main',
        ],
    },
)
