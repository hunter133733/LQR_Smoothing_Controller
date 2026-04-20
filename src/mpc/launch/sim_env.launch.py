import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_PARAMS_FILES = {
    "nav2_params_file": os.path.join(
        get_package_share_directory("mpc"),
        "params",
        "nav2_params.yaml",
    ),
    "mpc_params_file": os.path.join(
        get_package_share_directory("mpc"),
        "params",
        "mpc_planner_params.yaml",
    ),
    "rviz_settings": os.path.join(
        get_package_share_directory("mpc"),
        "rviz",
        "default_mpc_settings.rviz",
    ),
    "map_file": os.path.join(
        get_package_share_directory("mpc"),
        "maps",
        "empty_room.yaml",
    ),
    "world_file": os.path.join(
        get_package_share_directory("mpc"), "worlds", "empty_room.world"
    ),
}


def launch_setup(context, *args, **kwargs):
    mpc_dir = get_package_share_directory("mpc")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    nav2_bringup_launch_dir = os.path.join(nav2_bringup_dir, "launch")
    mpc_models_dir = os.path.join(mpc_dir, "models")
    tb3_models_dir = os.path.join(tb3_gazebo_dir, "models")

    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path = ":".join(
        p for p in [mpc_models_dir, tb3_models_dir, existing_model_path] if p
    )

    nav2_params_file = LaunchConfiguration("nav2_params_file")
    rviz_settings_file = LaunchConfiguration("rviz_settings_file").perform(context)

    map_file = LaunchConfiguration("map_file").perform(context)

    world = LaunchConfiguration("world")

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        output="screen",
        parameters=[{"use_sim_time": True}, {"yaml_filename": map_file}],
    )

    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager",
        output="screen",
        emulate_tty=True,  # https://github.com/ros2/launch/issues/188
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {"node_names": [
                "map_server", 
                "amcl",
                "controller_server",
                "planner_server",
                "behavior_server",
                "bt_navigator",
            ]},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        parameters=[
            {"use_sim_time": True},
        ],
        arguments=["-d" + rviz_settings_file],
        output={"both": "log"},
    )

    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        respawn_delay=2.0,
        parameters=[
            nav2_params_file,
            {
                "save_pose_rate": 5.0,
                "update_min_a": 0.0,
                "update_min_d": 0.0,
                "tf_broadcast": True,
                "set_initial_pose": True,
                "initial_pose.x": -4.0,
                "initial_pose.y": 3.5,
                "initial_pose.z": 0.0,
                "initial_pose.yaw": 0.0,
                "use_sim_time": True,
            },
        ],
    )

    urdf = os.path.join(tb3_gazebo_dir, "urdf", "turtlebot3_burger.urdf")
    with open(urdf, "r") as infp:
        robot_description = infp.read()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
    )

    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver",
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
            world,
        ],
        cwd=[nav2_bringup_launch_dir],
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
        output="screen",
    )

    gazebo_client = ExecuteProcess(
        cmd=["gzclient"],
        cwd=[nav2_bringup_launch_dir],
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
        output="screen",
    )

    pose = {
        "x": LaunchConfiguration("x_pose", default="-4.0"),
        "y": LaunchConfiguration("y_pose", default="3.5"),
        "z": LaunchConfiguration("z_pose", default="0.01"),
        "R": LaunchConfiguration("roll", default="0.0"),
        "P": LaunchConfiguration("pitch", default="0.0"),
        "Y": LaunchConfiguration("yaw", default="0.0"),
    }

    gazebo_spawner = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",
            "tb3_burger",
            "-file",
            os.path.join(tb3_gazebo_dir, "models", "turtlebot3_burger", "model.sdf"),
            "-x",
            pose["x"],
            "-y",
            pose["y"],
            "-z",
            pose["z"],
            "-R",
            pose["R"],
            "-P",
            pose["P"],
            "-Y",
            pose["Y"],
        ],
    )

    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_launch_dir, "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": nav2_params_file.perform(context),
            "autostart": "false",  # lifecycle_manager above handles activation
        }.items(),
    )

    return [
        gazebo_server,
        gazebo_client,
        gazebo_spawner,
        robot_state_publisher,
        map_server_node,
        lifecycle_manager_node,
        amcl_node,
        nav2_navigation,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "nav2_params_file",
                default_value=DEFAULT_PARAMS_FILES["nav2_params_file"],
                description="Nav2 parameters file to use",
            ),
            DeclareLaunchArgument(
                "run_rviz",
                default_value="True",
                description="Whether to use rviz",
                choices=["True", "False"],
            ),
            DeclareLaunchArgument(
                "rviz_settings_file",
                default_value=DEFAULT_PARAMS_FILES["rviz_settings"],
                description="Rviz settings file to use",
            ),
            DeclareLaunchArgument(
                "map_file",
                default_value=DEFAULT_PARAMS_FILES["map_file"],
                description="Map file to use",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=DEFAULT_PARAMS_FILES["world_file"],
                description="Full path to world model file to load",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
