# lqr_nav2_controller

A finite-horizon **Linear Quadratic Regulator (LQR)** controller plugin for [Nav2](https://nav2.ros.org/), targeting ROS 2 (tested on Humble / Iron). The controller tracks a global path by solving a time-varying LQR problem over a receding horizon, linearising the unicycle model at each reference point.

---

## How it works

The robot is modelled as a differential-drive unicycle:

```
ẋ = v·cos(θ)
ẏ = v·sin(θ)
θ̇ = ω
```

At each control step the plugin:

1. **Samples a reference trajectory** — picks `horizon` equally-spaced waypoints ahead on the global path and computes reference velocities `(v_ref, ω_ref)` from consecutive pose differences.
2. **Linearises** the unicycle dynamics at each reference point to produce time-varying matrices `A_k`, `B_k`.
3. **Solves a backward Riccati recursion** to obtain a sequence of feedback gains `K_k`.
4. **Applies the first gain** to the current state error to produce `(v, ω)` commands, clamped to the configured velocity limits.

---

## ⚠️ Important: This is a Nav2 plugin, not a full navigation stack

This repository only provides a **Nav2 Controller plugin (LQRController)**.

It does NOT include a complete Nav2 system.

To run this plugin, you still need a working Nav2 setup providing:

- A robot state publisher (`robot_state_publisher`)
- TF tree: map -> odom -> base_link
- - A localization system OR SLAM system:
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

## Usage

### Option A — drop-in replacement in your existing Nav2 stack

Add the plugin to your existing Nav2 params file under `controller_server`:

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "lqr_nav2_controller::LQRController"
      # LQR-specific params (see Parameters section below)
      dt:       0.1
      horizon:  25
      cruise_v: 0.3
      v_max:    0.5
      w_max:    1.2
      Q_x:      5.0
      Q_y:      5.0
      Q_theta:  2.0
      L_x:      10.0
      L_y:      10.0
      L_theta:  5.0
      R_v:      0.5
      R_w:      0.2
```

### Option B — standalone demo launch

Launches only `controller_server` + `lifecycle_manager` for quick testing:

```bash
ros2 launch lqr_nav2_controller lqr_controller_demo.launch.py
```

To override the params file:

```bash
ros2 launch lqr_nav2_controller lqr_controller_demo.launch.py \
  params_file:=/path/to/your/params.yaml
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dt` | `double` | `0.1` | Controller timestep (s). Keep equal to `1 / controller_frequency`. |
| `horizon` | `int` | `25` | Number of look-ahead steps for the Riccati recursion. |
| `cruise_v` | `double` | `0.3` | Reference forward speed along the path (m/s). |
| `v_min` | `double` | `0.0` | Minimum linear velocity (m/s). |
| `v_max` | `double` | `0.5` | Maximum linear velocity (m/s). |
| `w_min` | `double` | `-1.2` | Minimum angular velocity (rad/s). |
| `w_max` | `double` | `1.2` | Maximum angular velocity (rad/s). |
| `max_linear_vel` | `double` | `0.5` | Hard clamp on output linear velocity (m/s). |
| `max_angular_vel` | `double` | `1.2` | Hard clamp on output angular velocity (rad/s). |
| `Q_x` | `double` | `5.0` | State cost weight — x position error. |
| `Q_y` | `double` | `5.0` | State cost weight — y position error. |
| `Q_theta` | `double` | `2.0` | State cost weight — heading error. |
| `L_x` | `double` | `10.0` | Terminal cost weight — x position error. |
| `L_y` | `double` | `10.0` | Terminal cost weight — y position error. |
| `L_theta` | `double` | `5.0` | Terminal cost weight — heading error. |
| `R_v` | `double` | `0.5` | Control cost weight — linear velocity effort. |
| `R_w` | `double` | `0.2` | Control cost weight — angular velocity effort. |

### Parameter tuning guide

**Start here:**

1. Set `R_v` and `R_w` high (e.g. `1.0`) and `Q`/`L` low. Confirm the robot moves without oscillation.
2. Gradually increase `Q_x` and `Q_y` to tighten path tracking.
3. Increase `Q_theta` relative to `Q_x`/`Q_y` if the robot cuts corners or arrives at waypoints with wrong heading.
4. Set `L` (terminal weights) to 2–3× the corresponding `Q` values to ensure the robot commits to reaching each waypoint.
5. Increase `horizon` (and correspondingly reduce `dt`, or accept slower solve times) for smoother behaviour on high-curvature paths.

**Keep `dt = 1 / controller_frequency`.** If these diverge, the linearised dynamics matrices will be inconsistent with the actual control rate and tracking will degrade.

---

## License

TODO — add your chosen license here.