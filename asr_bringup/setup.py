import os
from glob import glob

from setuptools import setup

package_name = "asr_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kim",
    maintainer_email="ksj100848@gmail.com",
    description="Bringup and demo for Adaptive Sensor Reliability AI",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dashboard = asr_bringup.dashboard_node:main",
        ],
    },
)
