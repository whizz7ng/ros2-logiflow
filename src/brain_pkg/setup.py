from setuptools import setup

package_name = 'brain_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzz',
    maintainer_email='zzz@todo',
    description='brain + vision node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brain_node = brain_pkg.brain_node:main',
            'vision_node = brain_pkg.vision_node:main',
        ],
    },
)
