from setuptools import setup

package_name = "asr_evaluation"

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
    description="Dataset pipeline, training and evaluation for ASR",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "record_runs = asr_evaluation.record_runs:main",
            "extract_bag = asr_evaluation.extract_bag:main",
            "build_dataset = asr_evaluation.build_dataset:main",
            "train = asr_evaluation.train:main",
            "compare = asr_evaluation.compare:main",
            "external_check = asr_evaluation.external_check:main",
            "analyze_live = asr_evaluation.analyze_live:main",
            "calibrate = asr_evaluation.calibrate:main",
            "smoke_test_live = asr_evaluation.smoke_test_live:main",
        ],
    },
)
