from setuptools import setup

package_name = "asr_core"

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
    description="ROS-free core algorithms for Adaptive Sensor Reliability AI",
    license="Apache-2.0",
    tests_require=["pytest"],
)
