import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    controller_params = LaunchConfiguration("controller_params")
    pose_params = LaunchConfiguration("pose_params")

    controller_node = Node(
        package="controller",
        executable="controller",
        parameters=[controller_params],
    )

    pub_robot_pose_node = Node(
        package="mpc",
        executable="robot_pose_publisher",
        parameters=[pose_params],
    )

    return [controller_node, pub_robot_pose_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "controller_params",
                default_value=os.path.join(
                    get_package_share_directory("controller"),
                    "params",
                    "lqr_base_test.yaml",
                ),
                description="Controller parameters file to use",
            ),
            DeclareLaunchArgument(
                "pose_params",
                default_value=os.path.join(
                    get_package_share_directory("mpc"),
                    "params",
                    "pose_publisher.yaml",
                ),
                description="Pose publisher parameters file to use",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )