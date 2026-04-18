import os
from glob import glob

from setuptools import find_packages, setup

package_name = "controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "params"), glob("params/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
        (os.path.join("share", package_name, "data"), glob("data/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mo Chen",
    maintainer_email="mochen@cs.sfu.ca",
    description="Controller package, including LQR, HJ reachability, and CBFs",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "controller = controller.controller_node:main",
        ],
    },
)
