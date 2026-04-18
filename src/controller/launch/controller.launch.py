import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    controller_params = LaunchConfiguration("controller_params")
    planner_params = LaunchConfiguration("planner_params")

    planner_node = Node(
        package="mpc",
        executable="mpc",
        parameters=[planner_params],
        remappings=[
            ("/cmd_vel", "/cmd_vel_nom"),
        ],
    )
    controller_node = Node(
        package="controller",
        executable="controller",
        parameters=[controller_params],
    )
    pub_robot_pose_node = Node(
        package="mpc",
        executable="robot_pose_publisher",
        parameters=[planner_params],
    )
    return [planner_node, controller_node, pub_robot_pose_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "controller_params",
                default_value=os.path.join(
                    get_package_share_directory("controller"),
                    "params",
                    "controller_params.yaml",
                ),
                description="Controller parameters file to use",
            ),
            DeclareLaunchArgument(
                "planner_params",
                default_value=os.path.join(
                    get_package_share_directory("mpc"),
                    "params",
                    "mpc_planner_params.yaml",
                ),
                description="Planner parameters file to use",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
