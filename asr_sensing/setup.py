from setuptools import setup

package_name = "asr_sensing"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kim",
    maintainer_email="ksj100848@gmail.com",
    description="Sensor preprocessing nodes for Adaptive Sensor Reliability AI",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "encoder_odom = asr_sensing.encoder_odom_node:main",
            "lidar_odom = asr_sensing.lidar_odom_node:main",
            "ground_truth = asr_sensing.ground_truth_node:main",
            "driver = asr_sensing.driver_node:main",
            "fault_injector = asr_sensing.fault_injector_node:main",
        ],
    },
)
