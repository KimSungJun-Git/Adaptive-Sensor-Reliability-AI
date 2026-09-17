import os
from glob import glob

from setuptools import setup

package_name = "asr_reliability_estimator"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "model"), glob("model/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kim",
    maintainer_email="ksj100848@gmail.com",
    description="Reliability AI estimator, Adaptive EKF and covariance injector",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "reliability = asr_reliability_estimator.reliability_node:main",
            "adaptive_ekf = asr_reliability_estimator.adaptive_ekf_node:main",
            "covariance_injector = asr_reliability_estimator.covariance_injector:main",
        ],
    },
)
