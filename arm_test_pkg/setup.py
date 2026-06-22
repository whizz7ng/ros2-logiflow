from setuptools import setup

package_name = "arm_test_pkg"
setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="er",
    maintainer_email="er@todo",
    description="myCobot 280 Pi pick test",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pick_node = arm_test_pkg.pick_node:main",
        ],
    },
)
