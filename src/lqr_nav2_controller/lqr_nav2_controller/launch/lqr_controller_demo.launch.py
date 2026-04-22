from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    params = PathJoinSubstitution([
        FindPackageShare("lqr_nav2_controller"),
        "config",
        "lqr_controller_params.yaml"
    ])

    return LaunchDescription([

        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[params],
        ),

        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_controller",
            output="screen",
            parameters=[{
                "autostart": True,
                "node_names": ["controller_server"]
            }]
        ),
    ])