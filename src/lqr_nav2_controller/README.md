# lqr_nav2_controller

A finite-horizon **Linear Quadratic Regulator (LQR)** controller plugin for [Nav2](https://nav2.ros.org/), targeting ROS 2. The controller tracks a global path by solving a time-varying LQR problem over a receding horizon, linearising the unicycle model at each reference point.

---

## ⚠️ Important: This is a Nav2 plugin, not a full navigation stack

This repository only provides a **Nav2 Controller plugin (LQRController)**.

It does NOT include a complete Nav2 system.

To run this plugin, you still need a working Nav2 setup providing:

- A robot state publisher (`robot_state_publisher`)
- TF tree: map -> odom -> base_link
- A localization system OR SLAM system:
- `slam_toolbox` OR
- `amcl` + prebuilt map
- A full Nav2 stack (minimum):
- `controller_server` (provided via this plugin or external Nav2 bringup)
- `planner_server`
- `bt_navigator`
- `behavior_server`
- `waypoint_follower` (optional)

### Why this matters

This package only defines **how the robot computes velocity commands (LQR control)**.

It does NOT:
- Generate maps
- Provide localization
- Publish TF transforms
- Run full navigation pipelines

Without the above components, the controller will fail with errors such as:
- missing `/map` frame
- missing TF transforms (`map → base_link`)
- inactive costmap layers

---

## Repository layout

```
lqr_nav2_controller/
├── include/lqr_nav2_controller/
│   └── lqr_controller.hpp
├── src/
│   └── lqr_controller.cpp
├── config/
│   └── lqr_controller_params.yaml
├── launch/
│   └── lqr_controller_demo.launch.py
├── lqr_nav2_controller_plugins.xml
├── CMakeLists.txt
└── package.xml
```

---

## Dependencies

All dependencies are standard ROS 2 / Nav2 packages available via `rosdep`:

| Package | Purpose |
|---|---|
| `rclcpp` | ROS 2 C++ client library |
| `nav2_core` | Controller plugin interface |
| `nav2_util` | Nav2 utilities |
| `nav2_costmap_2d` | Costmap interface |
| `pluginlib` | Plugin loading |
| `geometry_msgs` / `nav_msgs` | Message types |
| `tf2_ros` | Transform lookups |
| `angles` | Angle wrapping helpers |
| `Eigen3` | Matrix maths (LQR solve) |

---

## Installation

### 1. Clone into your workspace

```bash
cd ~/ros2_ws/src
git clone https://github.com/<your-username>/lqr_nav2_controller.git
```

### 2. Install dependencies

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build

```bash
colcon build --packages-select lqr_nav2_controller
source install/setup.bash
```

---

## Minimal working setup (recommended)

To test this controller, run it with:

- `slam_toolbox` (for mapping + localization)
- `nav2_bringup` (base Nav2 stack)
- this plugin replacing the default controller

Example:

```bash
ros2 launch nav2_bringup slam_launch.py
```
Then modify the Nav2 params file:
```yaml
controller_server:
  FollowPath:
    plugin: "lqr_nav2_controller::LQRController"
```
---

## References

- ROS 2 Navigation Working Group. *Nav2 Controller Plugin Development Tutorial*.  
  https://navigation.ros.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html  
  Accessed: 2026-02-15
- ROS 2 Navigation Working Group. *nav2_core (Navigation2 Humble branch)*  
  https://github.com/ros-navigation/navigation2/tree/humble/nav2_core  
  Accessed: 2026-02-20
