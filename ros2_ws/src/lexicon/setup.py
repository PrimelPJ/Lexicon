from setuptools import find_packages, setup
import os
from glob import glob

package_name = "lexicon"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Primel Jayawardana",
    maintainer_email="you@primelj.dev",
    description="Open-vocabulary language-grounded navigation (VLM + Nav2)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = lexicon.detector_node:main",
            "find_object_server = lexicon.find_object_server:main",
        ],
    },
)
