# LQR and Augmented-State LQR Smoothing for Mobile Robot Trajectory Tracking

This repository contains the implementation of an augmented state LQR
smoothing controller for mobile robot trajectory tracking, evaluated on a
TurtleBot3 Burger in Gazebo simulation.

The smoothing controller augments the robot state with the previously applied
control input and solves for the control increment $\Delta u_k$ directly. A
penalty matrix $S$ on $\Delta u_k$ is embedded into the Riccati recursion,
producing smoother velocity commands without altering the standard LQR pipeline.
Setting $S = 0$ recovers the baseline LQR formulation.

We compare the baseline LQR controller and the augmented smoothing controller
across two trajectory types: a point-to-point task and a figure-8 trajectory.
On the figure-8, the smoothing controller reduces peak angular jerk by up to
74% and average angular jerk by 59% with negligible cost to tracking accuracy.

## Authors

Nick Smart, Morgan Hindy, Hershey Batore


## Setup

The repository is tested on Ubuntu 22.04 with ROS2 Humble. We recommend running
inside the provided dev container.

### Dependencies

- ROS2 Humble
- Gazebo Classic 11
- TurtleBot3 simulation packages: `turtlebot3_gazebo`, `turtlebot3_msgs`
- Python 3.10+ with `numpy`, `matplotlib`
- Custom message package: `nav_helpers_msgs` (included)

### Build

Clone the repository into your workspace and build with `colcon`:

```bash
git clone https://github.com/your-org/lqr-smoothing-controller.git
cd lqr-smoothing-controller
colcon build --packages-select controller --symlink-install
source install/setup.bash
```

Set the TurtleBot3 model:

```bash
export TURTLEBOT3_MODEL=burger
```

## Running the Experiments

Each experiment requires three terminals: one for Gazebo, one for the controller
node, and one for the trajectory publisher (figure-8 only).

### Experiment 1 — Point-to-Point

The robot starts at $(-4.0, 3.5)$ and navigates to a goal at $(-2.0, -4.0)$.

```bash
# Terminal 1 — Gazebo
ros2 launch mpc sim_env.launch.py

# Terminal 2 — Controller (swap YAML for each smoothing setting)
ros2 launch controller controller_ref_test.launch.py \
  controller_params:=src/controller/params/lqr_exp1_baseline.yaml \
  pose_params:=src/mpc/params/pose_publish.yaml
```

After each run, save the CSV log:

```bash
cp results/metrics/current_run.csv results/metrics/exp1_baseline.csv
```

### Experiment 2 — Figure-8

The robot starts at $(-4.0, 3.5)$, drives to the crossing point $(0, 0)$, and
executes two figure-8 lobes of radius $r = 1.2\,\mathrm{m}$ at $v = 0.5\,\mathrm{m/s}$.

```bash
# Terminal 1 — Gazebo
ros2 launch mpc sim_env.launch.py

# Terminal 2 — Controller
ros2 launch controller controller_ref_test.launch.py \
  controller_params:=src/controller/params/lqr_exp_fig8_baseline.yaml \
  pose_params:=src/mpc/params/pose_publish.yaml

# Terminal 3 — Figure-8 reference publisher
ros2 run controller figure8_publisher
```

After each run, save the CSV log:

```bash
cp results/metrics/current_run.csv results/metrics/fig8_baseline.csv
```

The seven smoothing settings used are: $(0,0)$, $(0.5,1)$, $(1,2)$, $(3,5)$,
$(10,20)$, $(30,60)$, $(50,100)$ for $(s_v, s_\omega)$.

## Analysis

Generate all plots from the logged CSV files:

```bash
python3 analysis/exp1_generate_plots.py
python3 analysis/exp2_generate_plots.py
```

Outputs land in `exp1_plots/` and `exp2_plots/`.


## Key Files

| File | Purpose |
|---|---|
| `lqr_algorithm.py` | Baseline finite-horizon LQR with backward Riccati recursion |
| `lqr_smoothing_augmented.py` | Augmented-state LQR; setting `du_v_cost = du_w_cost = 0` recovers baseline |
| `controller_node.py` | ROS2 frontend; subscribes to `/robot_pose` and `/traj`, publishes `/cmd_vel` |
| `figure8_publisher.py` | Pre-computes the figure-8 reference and broadcasts it on `/traj` |

## Results Summary

### Figure-8 (primary result)

| Run | RMS $\Delta\omega$ | Max $\Delta\omega$ | TV $\omega$ | Path (m) |
|---|---|---|---|---|
| Baseline (0,0) | 0.0630 | 1.515 | 21.61 | 29.06 |
| Light (0.5,1) | 0.0478 | 1.006 | 15.59 | 29.22 |
| Medium (1,2) | 0.0430 | 0.803 | 14.95 | 29.22 |
| Medium2 (3,5) | 0.0340 | 0.611 | 11.03 | 29.13 |
| Heavy (10,20) | 0.0259 | 0.400 | 8.94 | 29.56 |
| XHeavy (30,60) | 0.0195 | 0.332 | 6.39 | 29.36 |
| XXHeavy (50,100) | 0.0174 | 0.333 | 5.65 | 29.84 |

All seven smoothness metrics decrease monotonically with $\lambda$. Path length
varies by less than $0.8\,\mathrm{m}$ over $\sim 29.5\,\mathrm{m}$, indicating
negligible tracking cost. Diminishing returns emerge above $\lambda = 20$, identifying
Heavy $(10, 20)$ as the recommended operating point for this robot and trajectory.

## Acknowledgements

This project builds on starter code from Assignment 3 of CMPT 720 (Spring 2026)
at Simon Fraser University, taught by Prof. Mo Chen. The original assignment
provided the ROS2 frontend scaffolding (`controller_node.py`), the Dubins car
dynamics module (`dubins3d_2ctrls.py`), the reference trajectory generator,
and the baseline `LQRController` structure that we extended with the
augmented-state smoothing controller (`lqr_smoothing_augmented.py`) for this
project.

The augmented LQR formulation follows the lecture notes of P. Abbeel
(UC Berkeley CS 287) and the textbook treatment in Anderson and Moore,
*Optimal Control: Linear Quadratic Methods* (Prentice-Hall, 1990). Built on
top of the TurtleBot3 simulation stack and ROS2 Humble.
