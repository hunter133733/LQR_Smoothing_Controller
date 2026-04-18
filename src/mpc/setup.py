import os
from glob import glob

from setuptools import find_packages, setup

package_name = "mpc"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name, "maps"), glob("maps/*")),
    (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
    (os.path.join("share", package_name, "rviz"), glob("rviz/*")),
    (os.path.join("share", package_name, "launch"), glob("launch/*")),
    (os.path.join("share", package_name, "params"), glob("params/*")),
]

for directory in glob("models/*"):
    files = glob(directory + "/*")
    data_files.append((os.path.join("share", package_name, directory), files))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mo Chen",
    maintainer_email="mochen@cs.sfu.ca",
    description="ROS2 planner frontend for MPC and MPPI backends.",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "robot_pose_publisher = mpc.robot_pose_publisher:main",
            "mpc = mpc.mpc_planner:main",
        ],
    },
)
