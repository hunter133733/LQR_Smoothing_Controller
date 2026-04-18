# MPC / MPPI Planner (ROS2)

MPC planner with ROS2 frontend `MPCPlanner` the following backends:

- **MPPI backend:** `mpc.mppi_algorithm:MPPIController`
- **NMPC backend (CasADi):** `mpc.nmpc_algorithm:NMPCController`

Backends must implement the `ControllerBackend` interface (see `mpc/controller_base.py`) and provide:

- `get_action(observation: np.ndarray) -> np.ndarray` returning a 2D control `[v, omega]`.

## Quickstart for environment setup
Build the packages and run the simulation in ROS2 and Gazebo as follows.

1. Build your ROS2 workspace according to instructions in assignment 0, starting from cloning the `cmpt720_sp26` branch of hte repo.
```bash
git clone -b cmpt720_sp26 git@github.com:SFU-MARS/ros2_ws.git
```
2. Unzip package into the `src/` folder
3. Build and source the workspace

```bash
cd /workspaces/ros2_ws
colcon build --symlink-install --packages-select mpc
source install/setup.bash
```

2. Start simulation environment in Gazebo
```bash
ros2 launch mpc sim_env.launch.py
```

3. Run MPC node, with the backend specified in `src/mpc/params/mpc_planner_params.yaml`
```bash
ros2 run mpc mpc --ros-args --params-file src/mpc/params/mpc_planner_params.yaml
```

## Completing the Assignment
Although the package builds and node runs, the implementation of the node and the backend algorithm that it depends on are not complete.

The implementation should be completed by filling in sections marked as `# STUDENT CODE START` and `# STUDENT CODE END`. 
Do not make modifications anywhere else.

### Repository layout (what matters for the assignments)

| File                                            | Role                                                                                                          |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `src/mpc/mpc/nmpc_algorithm.py` | (Part A) **NMPC backend (CasADi)** (you implement Dubins step + constraints/objective here). Includes a Python-only demo. |
| `src/mpc/mpc/mppi_algorithm.py` | (Part A) **MPPI backend** (you implement core MPPI here). Includes a Python-only demo.                        |
| `src/mpc/mpc/mpc_planner.py`    | (Part B) ROS2 frontend node. Subscribes to `/robot_pose`, publishes `/cmd_vel`, calls backend each timer tick.            |

### Part A: Completing the backend

#### NMPC backend (CasADi)

Finish the implementation of the backend in  `mpc/nmpc_algorithm.py` by completing `dubins_step(...)` and `solve_mpc(...)`.

##### Implementation notes
- `dubins_step(...)` (symbolic dynamics) implements Forward-Euler Dubins dynamics for symbolic CasADi variables:
  - State: `x = [px, py, theta]`
  - Control: `u = [v, omega]`
  - Step: `dt`
  - Rules:
    - Use CasADi trig functions: `ca.cos`, `ca.sin` (not NumPy).
    - For simplicity in this assignment, do **not** wrap angles.

- `solve_mpc(...)` (constraints + objective)
  - Decision variables:
    - `X` shape `(3, N+1)` (CasADi), columns are states `k=0..N`
    - `U` shape `(2, N)` (CasADi), columns are controls `k=0..N-1`
  - Output:
    - `X_out`: NumPy array shape `(N+1, 3)`
    - `U_out`: NumPy array shape `(N, 2)`

Common pitfalls
- Corridor constraints must apply for **all** `k=0..N` (include terminal).
- Be careful with shapes on output: the solver gives `X` as `(3, N+1)`; you must return `(N+1, 3)`.
- Do not introduce NumPy operations on symbolic variables.

##### Python-only demo
For convenience, a python-only demo is provided to plot the solutions. It can be run via
```bash
python3 src/mpc/mpc/nmpc_algorithm.py # from /workspaces/ros2_ws
```

#### MPPI backend
Finish the implementation of the backend in  `mpc/mppi_algorithm.py` by completing `MPPI.rollout(...)` and  `MPPI.get_action(...)`.

##### Implementation notes
- `MPPI.rollout(self, observation, actions) -> costs`
  - Inputs:
    - `observation`: shape `(3,)` representing `x0 = [px, py, theta]`
    - `actions`: shape `(K, H, 2)` representing sampled action sequences `u^(k)_t = [v, omega]`
  - Output:
    - `costs`: shape `(K, H)` where `costs[k, t]` is the running cost at time `t`
  - Must include smoothness control transition penalty penalty for `t >= 1`.
  - Use the provided dynamics callback:
    - `self.dynamics_func(state, action) -> (next_state, cost)`
    - Here `cost` corresponds to the base running cost (tracking + effort + corridor penalty).

- `MPPI.get_action(self, observation) -> best_action`
  - Must implement the MPPI loop:
    1. Sample Gaussian noise `eps ~ N(0, Sigma)` of shape `(K, H, 2)`
    2. Perturb nominal action sequence `self.actions` and **clip** to bounds
    3. Roll out all trajectories and compute total trajectory cost `S_k = sum_t cost[k,t]`
    4. Compute weights.
    5. Update nominal sequence by weighted averaging:
    6. Receding horizon shift:
       - return `u_0`
       - shift sequence left and repeat the last element

- Required behavior / contracts
  - `get_action(...)` returns a NumPy array of shape `(2,)` ordered as `[v, omega]`.
  - `rollout(...)` returns a NumPy array of shape `(K, H)`.
  - Your implementation must be **deterministic** given the provided `np.random.Generator` (i.e., no calls to global RNG).
  - Use `np.clip(...)` for:
    - perturbed actions (per-sample),
    - updated nominal sequence,
    - final returned action (the wrapper also clips, but do it correctly anyway).
  - Use `epsilon` for numerical stability in denominators.

##### Python-only demo
For convenience, a python-only demo is provided to plot the solutions. It can be run via
```bash
python3 src/mpc/mpc/mppi_algorithm.py # from /workspaces/ros2_ws
```

### Part B: ROS2 Integration
1. After the backend implementations, you will proceed to implement the ROS2 node frontend module using the backend code at `mpc/mpc_planner.py`.
   - The node must 
       - subscribe to `/robot_pose` and extract the robot state from that message
       - publish to `/cmd_vel` at a rate of `control_rate` with the speed v being `Twist.linear.x` and the angular speed w being `Twist.angular.z`
   - It is helpful to refer to the previous assignment for the ROS2 node pattern, as well as the official ROS2 documentation and tutorials from assignment 0.

2. After this, you should have a fully runnable ROS2 controller ready to use in simulation or on real hardware.

3. With either backend (NMPC or MPPI), you should observe the robot moving to its goal location and hovering around it once it is close enough.

#### Implementation notes
- Default parameters are in `src/mpc/params/mpc_planner_params.yaml`

- To select backend implementation, set ROS parameter `backend_class` using `module.path:ClassName`. (Backends must implement `ControllerBackend`.)

- Depending on the computer hardware you are using, our `mppi_algorithm.py` is purely in Python and not optimized for high-frequency control. 
Since the planner must update at a rate higher than the control loop, to ensure the MPPI planner doesn't choke the ROS2 `cmd_vel`, consider reducing the MPPI `n_traj` and `horizon` to make it run faster.

- Similarly, for NMPC, setting the number of future steps `N` higher will help produce better trajectories, but setting it too high will slow down the algorithm.