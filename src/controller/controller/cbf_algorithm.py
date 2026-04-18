from typing import Any, Dict, Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from controller.controller_base import ControllerBackend
from nav_helpers.trajectory import StateActionTrajectory

# ============================================================
# Constants / Defaults
# ============================================================

DEFAULT_PARAMS: Dict[str, Any] = {
    # Nominal MPC parameters
    "N": 80,
    "dt": 0.1,
    "v_min": 0.0,
    "v_max": 1.0,
    "omega_max": 1.2,
    "w_pos": 2.0,
    "w_theta": 0.1,
    "w_u": 0.05,
    "w_du": 0.2,
    "w_pos_T": 50.0,
    "w_theta_T": 0.2,
    "solver_opts": {
        "print_level": 0,
        "max_iter": 300,
        "tol": 1e-6,
        "constr_viol_tol": 1e-4,
    },
    # CBF filter parameters
    "gamma_cbf": 2.0,
    "lookahead_distance": 0.8,
    "center_margin_buffer": 0.05,
    "w_cbf_v": 1.0,
    "w_cbf_omega": 0.05,
    # Simulation
    "goal_tolerance": 0.20,
    "sim_steps": 150,
}

DEFAULT_CORRIDOR: Dict[str, Any] = {
    "x_knots": [0.0, 2.0, 4.0, 6.0, 8.0],
    "y_low_knots": [-1.0, -1.4, -1.6, -1.6, -1.4],
    "y_high_knots": [1.0, 1.4, 1.6, 1.6, 1.4],
}

DEFAULT_X0 = np.array([0.5, 0.0, 0.0], dtype=float)
DEFAULT_GOAL = np.array([7.5, 0.0, 0.0], dtype=float)

DEFAULT_OBSTACLE: Dict[str, Any] = {
    "center": np.array([4.3, 0.15], dtype=float),
    "radius": 0.50,
    "safety_margin": 0.20,
}


# ============================================================
# Helper functions
# ============================================================


def dubins_step_numpy(x: NDArray, u: NDArray, dt: float) -> NDArray:
    """
    Forward Euler discretization:
      x = [px, py, theta], u = [v, omega]
    """
    x = np.asarray(x, dtype=float).reshape(3)
    u = np.asarray(u, dtype=float).reshape(2)

    px, py, th = float(x[0]), float(x[1]), float(x[2])
    v, om = float(u[0]), float(u[1])

    return np.array(
        [
            px + v * np.cos(th) * dt,
            py + v * np.sin(th) * dt,
            th + om * dt,
        ],
        dtype=float,
    )


def dubins_step(xk: ca.MX, uk: ca.MX, dt: float) -> ca.MX:
    """
    Symbolic forward Euler discretization:
      x = [px, py, theta], u = [v, omega]
    """
    px_next = xk[0] + uk[0] * ca.cos(xk[2]) * dt
    py_next = xk[1] + uk[0] * ca.sin(xk[2]) * dt
    th_next = xk[2] + uk[1] * dt
    return ca.vertcat(px_next, py_next, th_next)


def pwl_clamped(x, x_knots, y_knots):
    xk = list(map(float, x_knots))
    yk = list(map(float, y_knots))
    assert len(xk) == len(yk) and len(xk) >= 2, "invalid knot arrays"

    y = yk[-1]
    y = ca.if_else(x <= xk[0], yk[0], y)

    for i in range(len(xk) - 1):
        x0, x1 = xk[i], xk[i + 1]
        y0, y1 = yk[i], yk[i + 1]
        seg = ca.logic_and(x >= x0, x < x1)
        alpha = (x - x0) / (x1 - x0)
        yi = y0 + alpha * (y1 - y0)
        y = ca.if_else(seg, yi, y)

    y = ca.if_else(x >= xk[-1], yk[-1], y)
    return y


def y_low(px, corridor_params: Dict[str, Any]):
    return pwl_clamped(px, corridor_params["x_knots"], corridor_params["y_low_knots"])


def y_high(px, corridor_params: Dict[str, Any]):
    return pwl_clamped(px, corridor_params["x_knots"], corridor_params["y_high_knots"])


# ============================================================
# Nominal MPC
# ============================================================


def solve_mpc(
    x0: NDArray,
    goal: NDArray,
    corridor_params: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[NDArray, NDArray]:
    """
    solve_mpc(x0, goal, corridor_params, params) -> (X, U)
    """
    N = int(params["N"])
    dt = float(params["dt"])

    v_min = float(params["v_min"])
    v_max = float(params["v_max"])
    omega_max = float(params["omega_max"])

    w_pos = float(params["w_pos"])
    w_theta = float(params["w_theta"])
    w_u = float(params["w_u"])
    w_du = float(params["w_du"])
    w_pos_T = float(params["w_pos_T"])
    w_theta_T = float(params["w_theta_T"])

    x0 = np.asarray(x0, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)

    gx, gy, gth = float(goal[0]), float(goal[1]), float(goal[2])

    x_min = float(corridor_params["x_knots"][0])
    x_max = float(corridor_params["x_knots"][-1])

    x0_dm = ca.DM(x0)

    opti = ca.Opti()
    X = opti.variable(3, N + 1)
    U = opti.variable(2, N)

    J = ca.MX(0)

    # Warm start
    v_guess = (gx - float(x0[0])) / max(N * dt, 1e-9)
    v_guess = float(np.clip(v_guess, v_min, v_max))
    u_guess = np.array([v_guess, 0.0], dtype=float)

    X_init = np.zeros((3, N + 1), dtype=float)
    X_init[:, 0] = x0
    for k in range(N):
        X_init[:, k + 1] = dubins_step_numpy(X_init[:, k], u_guess, dt)
        X_init[0, k + 1] = float(np.clip(X_init[0, k + 1], x_min, x_max))

    U_init = np.zeros((2, N), dtype=float)
    U_init[0, :] = v_guess
    U_init[1, :] = 0.0

    opti.set_initial(X, X_init)
    opti.set_initial(U, U_init)

    # ============================================================
    # Nominal MPC constraints + objective
    # ============================================================

    opti.subject_to(X[:, 0] == x0_dm)

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        opti.subject_to(X[:, k + 1] == dubins_step(xk, uk, dt))
        opti.subject_to(opti.bounded(v_min, uk[0], v_max))
        opti.subject_to(opti.bounded(-omega_max, uk[1], omega_max))

        ep = ca.vertcat(xk[0] - gx, xk[1] - gy)
        eth = xk[2] - gth
        J += w_pos * ca.sumsqr(ep) + w_theta * (eth ** 2) + w_u * ca.sumsqr(uk)

    for k in range(N + 1):
        px = X[0, k]
        py = X[1, k]
        opti.subject_to(opti.bounded(x_min, px, x_max))
        opti.subject_to(py >= y_low(px, corridor_params))
        opti.subject_to(py <= y_high(px, corridor_params))

    xN = X[:, N]
    epN = ca.vertcat(xN[0] - gx, xN[1] - gy)
    ethN = xN[2] - gth
    J += w_pos_T * ca.sumsqr(epN) + w_theta_T * (ethN ** 2)

    for k in range(1, N):
        du = U[:, k] - U[:, k - 1]
        J += w_du * ca.sumsqr(du)

    opti.minimize(J)

    p_opts = {"expand": True, "print_time": False}
    s_opts = dict(params["solver_opts"])
    s_opts.setdefault("max_iter", 2000)
    s_opts.setdefault("acceptable_tol", 1e-3)
    s_opts.setdefault("acceptable_constr_viol_tol", 1e-4)
    s_opts.setdefault("acceptable_iter", 10)
    s_opts.setdefault("mu_strategy", "adaptive")

    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        X_out = sol.value(X).T
        U_out = sol.value(U).T
    except RuntimeError:
        print("Warning: nominal MPC solve failed. Returning zeros.")
        X_out = np.full((N + 1, 3), np.nan)
        U_out = np.zeros((N, 2), dtype=float)

    return X_out, U_out


# ============================================================
# Provided helper + student functions
# ============================================================
def get_nominal_control(
    x: NDArray,
    goal: NDArray,
    corridor_params: Dict[str, Any],
    params: Dict[str, Any],
) -> NDArray:
    """
    Provided helper: compute the nominal control using the provided MPC
    controller and return the first control input of the MPC plan.
    """
    _, U_nom = solve_mpc(
        x0=np.asarray(x, dtype=float).reshape(3),
        goal=np.asarray(goal, dtype=float).reshape(3),
        corridor_params=corridor_params,
        params=params,
    )

    if U_nom.shape[0] == 0 or np.any(np.isnan(U_nom[0])):
        return np.array([0.0, 0.0], dtype=float)

    return np.asarray(U_nom[0], dtype=float).reshape(2)


# ----------------------------
# Student implementation starts here
# ----------------------------


def cbf_terms(
    x: NDArray,
    obstacle: Dict[str, Any],
    lookahead_distance: float,
    center_margin_buffer: float,
) -> Tuple[float, float, float]:
    """
    Return barrier-related quantities h, a, b such that the chosen
    student-designed CBF constraint can be written as

        a(x) * v + b(x) * omega + gamma * h(x) >= 0.

    Design a distance-based barrier using a look-ahead point.
    Using the robot center will result in a relative degree of 2 with respect
    to the steering input, causing a loss of control authority. The design
    should correctly inflate the obstacle radius to account for this look-ahead distance.

    Inputs:
      x: robot state [px, py, theta]
      obstacle: dictionary containing obstacle center, radius, and safety margin
      lookahead_distance: parameter that should be used for the offset
      center_margin_buffer: parameter that may be used to pad the radius

    Outputs:
      h: scalar barrier value
      a: coefficient multiplying v in the affine CBF condition
      b: coefficient multiplying omega in the affine CBF condition
    """
    h = 0.0
    a = 0.0
    b = 0.0

    # ============================================================
    # STUDENT TODO:
    # Design and implement your barrier function for the circular
    # obstacle using a look-ahead point, and compute h(x), a(x), and b(x)
    # so that your chosen CBF condition can be enforced in the QP.
    #
    # You MUST use the look-ahead point rather than the robot center to
    # ensure your safety filter maintains steering authority.
    #
    # Your returned values must make the constraint
    #
    #     a(x) * v + b(x) * omega + gamma * h(x) >= 0
    #
    # meaningful for your chosen barrier.
    # ============================================================
    # STUDENT CODE START
    x = np.asarray(x, dtype=float).reshape(3)

    px = float(x[0])
    py = float(x[1])
    th = float(x[2])

    cx = float(obstacle["center"][0])
    cy = float(obstacle["center"][1])
    r_obs = float(obstacle["radius"])
    d_safe = float(obstacle["safety_margin"])

    L = float(lookahead_distance)
    eps = float(center_margin_buffer)

    pLx = px + L * np.cos(th)
    pLy = py + L * np.sin(th)

    dx = pLx - cx
    dy = pLy - cy

    r_eff = r_obs + d_safe + L  + eps

    h = dx * dx + dy * dy - r_eff * r_eff

    a = 2.0 * (dx * np.cos(th) + dy * np.sin(th))
    b = 2.0 * L * (-dx * np.sin(th) + dy * np.cos(th))
    # STUDENT CODE END

    return h, a, b


def solve_cbf_qp(
    x: NDArray,
    u_nom: NDArray,
    obstacle: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
) -> NDArray:
    """
    Solve the one-step safety-filter QP:

        min_u 0.5 * [ w_v (v-v_nom)^2 + w_omega (omega-omega_nom)^2 ]
        s.t.  a(x) * v + b(x) * omega + gamma * h(x) >= 0
              v_min <= v <= v_max
             -omega_max <= omega <= omega_max

    The barrier-related quantities h, a, b are produced by the
    student-designed cbf_terms(...) function.
    """
    u_safe = np.zeros(2, dtype=float)  # dummy

    # ============================================================
    # STUDENT TODO:
    # Implement the CBF-QP safety filter using your chosen barrier
    # design from cbf_terms(...).
    # ============================================================
    # STUDENT CODE START
    x = np.asarray(x, dtype=float).reshape(3)
    u_nom = np.asarray(u_nom, dtype=float).reshape(2)

    if obstacle is None:
        obstacle = DEFAULT_OBSTACLE
    if params is None:
        params = DEFAULT_PARAMS
    
    gamma = float(params["gamma_cbf"])
    lookahead_distance = float(params["lookahead_distance"])
    center_margin_buffer = float(params["center_margin_buffer"])

    v_min = float(params["v_min"])
    v_max = float(params["v_max"])
    omega_max = float(params["omega_max"])

    w_v = float(params["w_cbf_v"])
    w_omega = float(params["w_cbf_omega"])

    u_nom_clip = np.array(
        [
            np.clip(u_nom[0], v_min, v_max),
            np.clip(u_nom[1], -omega_max, omega_max),
        ],
        dtype=float,
    )

    h, a, b = cbf_terms(
        x=x,
        obstacle=obstacle,
        lookahead_distance = lookahead_distance,
        center_margin_buffer = center_margin_buffer,
    )

    lhs_nom = a * u_nom_clip[0] + b * u_nom_clip[1] + gamma * h
    if np.isfinite(lhs_nom) and lhs_nom >= 0.0:
        return u_nom_clip

    opti = ca.Opti()

    v = opti.variable()
    omega = opti.variable()

    J = 0.5 * (
        w_v * (v - float(u_nom_clip[0])) ** 2
        + w_omega * (omega - float(u_nom_clip[1])) ** 2
    )

    opti.minimize(J)

    opti.subject_to(opti.bounded(v_min, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    opti.subject_to(a * v + b * omega + gamma * h >+ 0.0)

    opti.set_initial(v, float(u_nom_clip[0]))
    opti.set_initial(omega, float(u_nom_clip[1]))

    p_opts = {"expand": True, "print_time": False}
    s_opts = {
        "print_level": 0,
        "sb": "yes",
        "max_iter": 100,
        "tol": 1e-6
    }

    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        u_safe = np.array(
            [float(sol.value(v)), float(sol.value(omega)),],
            dtype=float,
        )
    except RuntimeError:
        if b>=0.0:
            omega_fallback = omega_max
        else:
            omega_fallback = -omega_max
        
        u_safe = np.array([v_min, omega_fallback], dtype=float)
    
    u_safe[0] = np.clip(u_safe[0], v_min, v_max)
    u_safe[1] = np.clip(u_safe[1], -omega_max, omega_max)
    # STUDENT CODE END

    return u_safe


class CBFController(ControllerBackend):
    """Thin wrapper used by ROS2 frontend."""

    def __init__(self, config):
        cfg = dict(config.get("cbf", {}))

        u_min = np.array(
            [float(cfg.get("v_min", -0.2)), float(cfg.get("w_min", -1.2))], dtype=float
        )
        u_max = np.array(
            [float(cfg.get("v_max", 1.0)), float(cfg.get("w_max", 1.2))], dtype=float
        )

        self._u_min = u_min
        self._u_max = u_max
        self._params = cfg

    def get_action(
        self, observation: NDArray, traj: StateActionTrajectory = None
    ) -> NDArray:
        obs = np.asarray(observation, dtype=float).reshape(3)
        action = np.zeros((2,), dtype=float)

        # TODO: Compute safety-filtered action
        # STUDENT CODE START
        if traj is not None and hasattr(traj, "actions") and len(traj.actions) > 0:
            u_nom = np.asarray(traj.actions[0], dtype=float).reshape(2)
        else:
            u_nom = np.array([0.0, 0.0], dtype=float)

        obstacle = self._params.get("obstacle", DEFAULT_OBSTACLE)
        action = solve_cbf_qp(
            x=obs,
            u_nom=u_nom,
            obstacle=obstacle,
            params=self._params,
        )
            
        # STUDENT CODE END

        return np.clip(action, self._u_min, self._u_max), None, None


# ============================================================
# Closed-loop simulation
# ============================================================


def simulate_cbf_filtered_controller(
    x0: NDArray,
    goal: NDArray,
    corridor_params: Dict[str, Any],
    obstacle: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Closed-loop simulation:
      1) compute nominal MPC action
      2) filter it through the CBF-QP
      3) simulate one step
    """
    dt = float(params["dt"])
    sim_steps = int(params["sim_steps"])
    goal_tolerance = float(params["goal_tolerance"])

    x = np.asarray(x0, dtype=float).reshape(3).copy()

    X_hist = [x.copy()]
    U_nom_hist = []
    U_safe_hist = []

    reached_goal = False

    for _ in range(sim_steps):
        u_nom = get_nominal_control(x, goal, corridor_params, params)
        u_safe = solve_cbf_qp(x, u_nom, obstacle, params)

        x = dubins_step_numpy(x, u_safe, dt)

        X_hist.append(x.copy())
        U_nom_hist.append(u_nom.copy())
        U_safe_hist.append(u_safe.copy())

        if np.linalg.norm(x[:2] - np.asarray(goal[:2], dtype=float)) <= goal_tolerance:
            reached_goal = True
            break

    if reached_goal:
        print("Reached the goal.")
    else:
        print("Did not reach the goal within the simulation horizon.")

    if len(U_nom_hist) == 0:
        U_nom_arr = np.zeros((0, 2), dtype=float)
        U_safe_arr = np.zeros((0, 2), dtype=float)
    else:
        U_nom_arr = np.asarray(U_nom_hist, dtype=float)
        U_safe_arr = np.asarray(U_safe_hist, dtype=float)

    return np.asarray(X_hist, dtype=float), U_nom_arr, U_safe_arr


# ============================================================
# Plotting
# ============================================================


def plot_trajectory(
    X: NDArray,
    U_safe: NDArray,
    corridor_params: Dict[str, Any],
    goal: NDArray,
    obstacle: Dict[str, Any],
    params: Dict[str, Any],
):
    import matplotlib.pyplot as plt

    if np.any(np.isnan(X)):
        print("No valid trajectory to plot.")
        return

    X = np.asarray(X, dtype=float)
    U_safe = np.asarray(U_safe, dtype=float)

    px = X[:, 0]
    py = X[:, 1]
    th = X[:, 2]

    xk = corridor_params["x_knots"]
    yl = corridor_params["y_low_knots"]
    yh = corridor_params["y_high_knots"]

    cx, cy = obstacle["center"]
    r_obs = float(obstacle["radius"])
    d_safe = float(obstacle["safety_margin"])

    L = float(params["lookahead_distance"])
    eps = float(params["center_margin_buffer"])
    gamma = float(params["gamma_cbf"])

    r_safe_center = r_obs + d_safe
    r_eff = r_obs + d_safe + L + eps

    pLx = px + L * np.cos(th)
    pLy = py + L * np.sin(th)

    h_vals = []
    cbf_lhs = []

    for k in range(X.shape[0]):
        h, a, b = cbf_terms(X[k], obstacle, L, eps)
        h_vals.append(h)

        if k < U_safe.shape[0]:
            v = U_safe[k, 0]
            omega = U_safe[k, 1]
            cbf_lhs.append(a * v + b * omega + gamma * h)

    h_vals = np.asarray(h_vals, dtype=float)
    cbf_lhs = np.asarray(cbf_lhs, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(xk, yl, "b--", linewidth=2, label="Lower boundary")
    ax.plot(xk, yh, "b--", linewidth=2, label="Upper boundary")
    ax.plot(px, py, "r.-", linewidth=2, markersize=5, label="Robot trajectory")
    ax.plot(pLx, pLy, "m.-", linewidth=1.5, markersize=4, label="Look-ahead trajectory")

    obstacle_circle = plt.Circle(
        (cx, cy), r_obs, color="gray", alpha=0.6, label="Obstacle"
    )
    center_safe_circle = plt.Circle(
        (cx, cy),
        r_safe_center,
        fill=False,
        linestyle="--",
        linewidth=2,
        color="k",
        label="Center safety circle",
    )
    barrier_circle = plt.Circle(
        (cx, cy),
        r_eff,
        fill=False,
        linestyle="-.",
        linewidth=2,
        color="m",
        label="Look-ahead barrier boundary",
    )

    ax.add_patch(obstacle_circle)
    ax.add_patch(center_safe_circle)
    ax.add_patch(barrier_circle)

    ax.plot(X[0, 0], X[0, 1], "go", markersize=10, label="Start")
    ax.plot(goal[0], goal[1], "g*", markersize=14, label="Goal")

    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.set_title("Trajectory and Barrier Geometry")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()

    ax = axes[1]
    ax.plot(np.arange(len(h_vals)), h_vals, "m.-", linewidth=2, markersize=4)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("h(x)")
    ax.set_title("Barrier Value Along Rollout")
    ax.grid(True)

    ax = axes[2]
    ax.plot(np.arange(len(cbf_lhs)), cbf_lhs, "c.-", linewidth=2, markersize=4)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$a(x)v + b(x)\omega + \gamma h(x)$")
    ax.set_title("CBF Constraint Along Rollout")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("cbf_filtered_corridor_with_barrier.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    X, U_nom, U_safe = simulate_cbf_filtered_controller(
        x0=DEFAULT_X0,
        goal=DEFAULT_GOAL,
        corridor_params=DEFAULT_CORRIDOR,
        obstacle=DEFAULT_OBSTACLE,
        params=DEFAULT_PARAMS,
    )
    plot_trajectory(
        X,
        U_safe,
        DEFAULT_CORRIDOR,
        DEFAULT_GOAL,
        DEFAULT_OBSTACLE,
        DEFAULT_PARAMS,
    )
