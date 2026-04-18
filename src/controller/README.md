# Controller

This package contains controllers that take as input the result of planners to produce controls controls executed by a robot.

The ROS2 frontend is `ControllerNode` and uses one of the following backends:

- `controller.lqr_algorithm:LQRController`
- `controller.lqr_algorithm:HJController`
- `controller.lqr_algorithm:CBFController`

Backends must implement `ControllerBackend.get_action(observation, traj)` and return at least a control vector `[v, omega]`.

The package builds and runs, but the control logic is intentionally incomplete.

Complete only code inside sections marked:

- `# STUDENT CODE START`
- `# STUDENT CODE END`

Do not modify code outside these regions.

## Quickstart

1. Build the package in your ROS2 workspace:

```bash
colcon build --symlink-install --packages-select controller mpc nav_helpers nav_helpers_msgs
colcon build --symlink-install --base-paths py # for ROS2 to recognize OptimizedDP
source install/setup.bash
```

2. Run the simulation environment:
```bash
ros2 launch mpc sim_env.launch.py
```

3. Set the desired backend and parameters in `controller_params.yaml`.

4. Launch the controller launch file, which will launch NMPC as the planner that publishes a `StateActionTrajectory` (try `ros2 interface show nav_helpers_msgs/msg/StateActionTrajectory`) which will be an input to the `ControllerNode`:

```bash
ros2 launch controller controller.launch.py
```

## LQR controller
### Code Layout

The files that matter for this question are:

- `controller/lqr_algorithm.py`
  Finite-horizon time-varying LQR gain computation, tracking rollout, and receding-horizon action selection.
- `controller/controller_node.py`
  ROS2 node that subscribes to `/robot_pose` and publishes `/cmd_vel`.

### Part A: Backend Control Logic

Implement the missing logic in the following functions.

The controller is a discrete-time finite-horizon time-varying LQR tracker around a nominal Dubins trajectory:

- nominal states `z_ref[t]`
- nominal controls `u_ref[t]`
- linearized error dynamics `delta_z[t+1] = A_t delta_z[t] + B_t delta_u[t]`
- tracking law `u_t = u_ref[t] - K_t delta_z[t]`

The finite-horizon quadratic cost is:

- stage cost `delta_z[t]^T Q delta_z[t] + delta_u[t]^T R delta_u[t]`
- terminal cost `delta_z[N]^T L delta_z[N]`

#### 1. `LQRAlgorithm.compute_gains(...)`

Task:
- implement backward finite-horizon Riccati recursion
- use the terminal cost matrix `L`
- return a feedback matrix `K_t` for every step in the horizon

Use the standard discrete-time recursion from lecture:

- `P_N = L`
- `K_t = (R + B_t^T P_{t+1} B_t)^(-1) B_t^T P_{t+1} A_t`
- `P_t = Q + A_t^T P_{t+1} A_t - A_t^T P_{t+1} B_t K_t`

#### 2. `LQRAlgorithm.solve(...)`

Inputs:

- initial state `z_0`
- start time `t_0`
- reference states `z_ref`
- reference controls `u_ref`

Outputs:

- `z_sol`
- `u_sol`
- `tau_sol`

Task:
- compute the tracking error relative to the nominal trajectory
- use `delta_z[t] = z_t - z_ref[t]`
- wrap the heading error on the periodic angle dimension
- apply the time-varying LQR tracking law `u_t = u_ref[t] - K_t @ delta_z`

#### 3. `LQRController.get_action(...)`

Task:
- extract the current reference window
- solve the finite-horizon tracking problem
- return the first control in the solved sequence
- advance the internal horizon index for receding-horizon control

The provided helper pads the reference window with the terminal sample when the horizon extends beyond the end of the nominal trajectory.

### Part B: ROS2 Integration

Complete the missing ROS2 wiring in `controller/controller_node.py`.

Your node must:

1. subscribe to `/robot_pose` and save the latest state as `[x, y, yaw]`
2. subscribe to `/traj` and ave the latest trajectory as a `StateActionTrajectory`
3. call the backend every timer tick to compute the control, and publish `Twist` on `/cmd_vel` as a `Twist` message

### Common Pitfalls
- mixing continuous-time and discrete-time Jacobians
- forgetting the terminal condition `P_N = L`
- using incorrect matrix shapes in the Riccati recursion
- forgetting that `K_t` has shape `(2, 3)`
- failing to wrap angle error near `pi` and `-pi`
- returning the wrong action from the receding-horizon solve
- publishing the wrong `Twist` fields in ROS2

## Control Barrier Function Safety Filter

This package also contains the implementation of a control barrier function (CBF) safety filter for a Dubins car moving through a 2D corridor with a circular obstacle.

The provided code includes a nominal nonlinear MPC controller that drives the robot toward the goal while respecting the corridor and input constraints. However, the nominal controller does **not** account for the obstacle. In the starter code, the nominal control is already provided to the safety filter at each time step. Your task is to implement a CBF-based safety filter that modifies this nominal control only when necessary for obstacle avoidance.

### Code Layout

| File | Role |
|---|---|
| `controller/cbf_algorithm.py` | Main file for this question. |

### What you need to implement

Complete the marked blocks in `cbf_algorithm.py`:

1. Design your own barrier using the look-ahead point and compute the corresponding quantities `h(x)`, `a(x)`, and `b(x)`,
2. Implement the CBF-QP safety filter.
3. Complete the `get_action` interface to the ROS2 node

You can run the provided simulation and visualize the resulting trajectory and barrier-related plots.

### Implementation notes

- Many parameters, including control bounds, weights, and CBF parameters, are found in the dictionary `params`. For example, `gamma` corresponds to `params["gamma_cbf"]`.
- The corridor parameters are specified in `corridor_params`.
- The obstacle parameters are specified in `obstacle`.
- The goal state is specified by `goal`.
- The look-ahead distance corresponds to `params["lookahead_distance"]`.
- The additional buffer corresponds to `params["center_margin_buffer"]`.

### Hints

- Make sure to use the state ordering `[px, py, theta]` and control ordering `[v, omega]`.
- The nominal control input is already provided to the safety filter in the starter code.
- You must define your barrier using the look-ahead point, not the robot center.
- Your barrier must be designed so that steering appears in the affine CBF constraint.
- Your `cbf_terms(...)` function must return three finite scalar values: `h`, `a`, and `b`.
- Your safety filter should always return a valid bounded control input.
- The safety filter should stay close to the nominal control when the nominal control is already safe.
- A correct implementation should keep the robot away from the obstacle while still allowing it to make progress toward the goal.


## Hamilton-Jacobi (HJ) Reachability Based Safety Filter

This question includes two parts. The first part involves solving the HJ PDE to obtain the value function. The second part will make use of this value function to ensure safety of our robot through a *Least Restrictive Switching Filter* logic. 

### Installing optimized_dp
Please follow the instructions [here]{https://github.com/SFU-MARS/optimized_dp/tree/master} to install `conda`, necessary solver and its dependent components. If you encounter error about `libtinfo5` not found in your docker container, try running `sudo apt-get update` first before running `sudo apt install libtinfo5`.

**Notes**: We also want to use some pure pythonic components of the package outside of the `conda` environment when running `ros` nodes (`ROS` cannot run inside a `conda` env) . Open up another terminal and make sure `conda` is deactivated, go to the pulled `optimized_dp` repo and run `pip install -e .` to make all the modules visible to any python process. You will use these components when implementing the *Least Restrictive Switching Filter* inside the `ROS` controller later.  

### Switching between the environment
After following the above steps, the package `odp` will be available anywhere to be imported in your python process. When computing the HJ value function, make sure it is done inside the installed `conda` environment. In other words, once you completed your implementation for `py/compute_hj_value_function.py`, run it inside the `odp` environment. We store the results from this section to use it online in our `ros` controller in the next section.


### Code Layout

| File | Role | Conda (yes or no)
|---|---|---|
| `py/compute_hj_value_function.py` | Solve the HJ pde that computes the value function V and also the spatial derivatives for use in the safety filter | Yes (odp)
| `src/controller/controller/hj_algorithm.py` | Loading the pre-computed value function to implement Least Restrictive Switching Filter | No

### What you need to implement

Complete the marked `TODO` blocks in `compute_hj_value_function.py` and `hj_algorithm.py`:

1. Correctly initialize the problem that's required to be solved. Please refer to this [example]{https://github.com/SFU-MARS/optimized_dp/blob/master/examples/pursuit_evasion_game.py} to check how to specify an example.
2. Using the precomputed value function to implement the Least Restrictive Switching Safety filter inside `HJController.get_action()`.

### Implementation notes
Unlike previous section on CBF, all the necessary parameters have to be determined and specified during the offline computation of the computation. The only parameter that's needed to be specified is just the `epsilon` used in the switching filter.


### Hints
- Your will find the python member functions of class `Grid` and `DubinsCar2` useful for implementing the safety filter switch logic
- It's important to notice that `theta` is periodic, make sure that you can specify this when initializing the grid in `compute_hj_value_function.py`. This will ensure the solver can correctly handle the periodicity of the state space.
- To plot the 2D slice the zero level-set surface, you can set `plotDims=[0, 1]` and input in index for the third dimension in `slicesCut`.

## Real-World Demonstration with TurtleBot3
See `robot_readme.md`