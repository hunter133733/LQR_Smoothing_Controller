import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Arguments ────────────────────────────────────────────────
    # Pass  planner:=smac  (or navfn / thetastar) on the command line.
    # The argument just controls which commented block you activate in
    # the YAML — or you can wire it to a different params file per planner.
    pkg_share = FindPackageShare("nav2_lqr_bringup")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "nav2_params.yaml"]),
        description="Full path to the nav2 parameter file",
    )

    map_file_arg = DeclareLaunchArgument(
        "map",
        default_value=PathJoinSubstitution([pkg_share, "maps", "map.yaml"]),
        description="Full path to the map yaml file",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    params_file    = LaunchConfiguration("params_file")
    map_file       = LaunchConfiguration("map")
    use_sim_time   = LaunchConfiguration("use_sim_time")

    # ── Nodes ─────────────────────────────────────────────────────
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[params_file, {"yaml_filename": map_file,
                                   "use_sim_time": use_sim_time}],
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
    )

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
    )

    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        # Remaps cmd_vel so a velocity smoother or mux can sit in between.
        # Remove the remap if you want to drive the robot directly.
        remappings=[("cmd_vel", "cmd_vel_nav")],
    )

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager",
        output="screen",
        parameters=[
            {"autostart": True},
            {"use_sim_time": use_sim_time},
            {"node_names": [
                "map_server",
                "amcl",
                "planner_server",
                "controller_server",
                "bt_navigator",
            ]},
        ],
    )

    return LaunchDescription([
        params_file_arg,
        map_file_arg,
        use_sim_time_arg,
        map_server,
        amcl,
        planner_server,
        controller_server,
        bt_navigator,
        lifecycle_manager,
    ])