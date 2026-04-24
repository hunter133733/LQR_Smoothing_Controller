# terminal 1
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
# terminal 2
ros2 launch nav2_bringup tb3_simulation_launch.py
# terminal 3
ros2 launch nav2_bringup navigation_launch.py   params_file:=src/nav2_lqr_bringup/config/nav2_params.yaml   use_sim_time:=true