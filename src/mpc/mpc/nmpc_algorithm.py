from typing import Any, Dict, Tuple

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from controller.controller_base import ControllerBackend

# =======================
# Constants / Defaults: Do not modify!
# =======================

# fmt: off
DEFAULT_PARAMS: Dict[str, Any] = {
    # Horizon long enough to reach px=7.5 in one solve:
    # max distance in one solve = N*dt*v_max = 80*0.1*1.0 = 8.0
    "N": 80,
    "dt": 0.1,  # NOTE: dt corresponds to Δt in the handout
    "v_min": 0.0,
    "v_max": 1.0,
    "omega_max": 1.2,
    "w_pos": 2.0,
    "w_theta": 0.1,
    "w_u": 0.05,
    "w_du": 0.2,
    "w_pos_T": 50.0,  # stronger terminal pull
    "w_theta_T": 0.2,
    # IPOPT options (keep deterministic-ish)
    "solver_opts": {"print_level": 0, "max_iter": 300, "tol": 1e-6, "constr_viol_tol": 1e-4},
}
# fmt: on

# fmt: off
DEFAULT_CORRIDOR: Dict[str, Any] = {
    "x_knots":      [0.0, 2.0, 4.0, 6.0, 8.0],
    "y_low_knots":  [-1.0, -1.4, -1.6, -1.6, -1.4],
    "y_high_knots": [ 1.0,  1.4,  1.6,  1.6,  1.4],
}
# fmt: on

DEFAULT_X0 = np.array([0.5, 0.0, 0.0])
DEFAULT_GOAL = np.array([7.5, 0.0, 0.0])


# =======================
# Helper (Dynamics used for plotting only and corridor): Do not modify!
# =======================


def dubins_step_numpy(x: NDArray, u: NDArray, dt: float) -> NDArray:
    """
    Forward Euler discretization:
      x = [px, py, theta], u = [v, omega]
    Returns x_{k+1} as a NumPy array of shape (3,).
    """
    x = np.asarray(x, dtype=float).reshape(3)
    u = np.asarray(u, dtype=float).reshape(2)
    px, py, th = float(x[0]), float(x[1]), float(x[2])
    v, om = float(u[0]), float(u[1])
    x_new = np.array([px + v * np.cos(th) * dt,
                      py + v * np.sin(th) * dt,
                      th + om * dt], dtype=float)  # fmt: skip
    return x_new


def pwl_clamped(x, x_knots, y_knots):
    xk = list(map(float, x_knots))
    yk = list(map(float, y_knots))
    assert (
        len(xk) == len(yk) and len(xk) >= 2
    ), "knot arrays must match and have length >= 2"

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


# =======================
# Student functions
# =======================
def dubins_step(xk: ca.MX, uk: ca.MX, dt: float) -> ca.MX:
    """
    Forward Euler discretization (symbolic / CasADi):
      x = [px, py, theta], u = [v, omega]
    Returns x_{k+1} as a CasADi vector of shape (3,).
    """
    x_new = None  # NOTE: dummy

    # TODO:
    # STUDENT CODE START
    px_next = xk[0] + uk[0] * ca.cos(xk[2]) * dt
    py_next = xk[1] + uk[0] * ca.sin(xk[2]) * dt
    th_next = xk[2] + uk[1] * dt
    x_new = ca.vertcat(px_next, py_next, th_next)
    # STUDENT CODE END

    return x_new


def solve_mpc(
    x0: NDArray,
    goal: NDArray,
    corridor_params: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[NDArray, NDArray]:
    """
    solve_mpc(x0, goal, corridor_params, params) -> (X, U)

    Outputs:
      X: (N+1, 3) numpy array, rows are k=0..N
      U: (N, 2)   numpy array, rows are k=0..N-1
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
    X = opti.variable(3, N + 1)  # columns: k=0..N
    U = opti.variable(2, N)  # columns: k=0..N-1

    J = ca.MX(0)

    # ============================================================
    # Feasible warm start (inside corridor + px bounds)
    # ============================================================
    v_guess = (gx - float(x0[0])) / max(N * dt, 1e-9)
    v_guess = float(np.clip(v_guess, v_min, v_max))
    u_guess = np.array([v_guess, 0.0], dtype=float)

    X_init = np.zeros((3, N + 1), dtype=float)
    X_init[:, 0] = x0
    for k in range(N):
        X_init[:, k + 1] = dubins_step_numpy(X_init[:, k], u_guess, dt)
        # keep px guess within corridor knot interval
        X_init[0, k + 1] = float(np.clip(X_init[0, k + 1], x_min, x_max))

    U_init = np.zeros((2, N), dtype=float)
    U_init[0, :] = v_guess
    U_init[1, :] = 0.0

    opti.set_initial(X, X_init)
    opti.set_initial(U, U_init)

    # ============================================================
    # STUDENT TODO: constraints + objective
    # ============================================================
    # STUDENT CODE START

    # Initial condition
    opti.subject_to(X[:, 0] == x0_dm)

    # Dynamics + input bounds + stage cost for k=0..N-1
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        # Dynamics
        opti.subject_to(X[:, k + 1] == dubins_step(xk, uk, dt))

        # Input bounds
        opti.subject_to(opti.bounded(v_min, uk[0], v_max))
        opti.subject_to(opti.bounded(-omega_max, uk[1], omega_max))

        # Stage cost
        ep = ca.vertcat(xk[0] - gx, xk[1] - gy)
        eth = xk[2] - gth  # no wrapping (per handout)
        J += w_pos * ca.sumsqr(ep) + w_theta * (eth ** 2) + w_u * ca.sumsqr(uk)

    # Corridor + px bounds for all k=0..N (terminal included)
    for k in range(N + 1):
        px = X[0, k]
        py = X[1, k]
        opti.subject_to(opti.bounded(x_min, px, x_max))
        opti.subject_to(py >= y_low(px, corridor_params))
        opti.subject_to(py <= y_high(px, corridor_params))

    # Terminal cost at k=N
    xN = X[:, N]
    epN = ca.vertcat(xN[0] - gx, xN[1] - gy)
    ethN = xN[2] - gth
    J += w_pos_T * ca.sumsqr(epN) + w_theta_T * (ethN ** 2)

    # Smoothness cost: sum_{k=1}^{N-1} w_du ||u_k - u_{k-1}||^2
    for k in range(1, N):
        du = U[:, k] - U[:, k - 1]
        J += w_du * ca.sumsqr(du)
    # STUDENT CODE END

    opti.minimize(J)

    # ============================================================
    # Robust IPOPT setup
    # ============================================================
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
        print("Warning: MPC solve failed (possibly infeasible). Returning NaNs.")
        X_out = np.full((N + 1, 3), np.nan)
        U_out = np.full((N, 2), np.nan)

    return X_out, U_out


# =======================
# ros2 intergration
# =======================

from mpc.controller_base import ControllerBackend


class NMPCController(ControllerBackend):
    """A thin backend wrapper implements necessary methods used by ROS frontend."""

    def __init__(self, config: Dict[str, Any]):
        self._goal = np.asarray(config.get("goal", DEFAULT_GOAL), dtype=float).reshape(
            3
        )
        self._corridor = dict(config.get("corridor", DEFAULT_CORRIDOR))
        self._action_min = np.array(
            [float(config.get("v_min", 0.0)), -float(config.get("omega_max", 1.2))],
            dtype=float,
        )
        self._action_max = np.array(
            [float(config.get("v_max", 1.0)), float(config.get("omega_max", 1.2))],
            dtype=float,
        )

        mpc_cfg = dict(config.get("mpc", {}))
        solver_opts = dict(DEFAULT_PARAMS["solver_opts"])
        solver_opts["max_iter"] = int(
            mpc_cfg.get("solver_max_iter", solver_opts["max_iter"])
        )
        solver_opts["tol"] = float(mpc_cfg.get("solver_tol", solver_opts["tol"]))
        solver_opts["constr_viol_tol"] = float(
            mpc_cfg.get("solver_constr_viol_tol", solver_opts["constr_viol_tol"])
        )

        weights = dict(config.get("weights", {}))
        self._params = {
            "N": int(mpc_cfg.get("N", DEFAULT_PARAMS["N"])),
            "dt": float(config.get("dt", DEFAULT_PARAMS["dt"])),
            "v_min": float(config.get("v_min", DEFAULT_PARAMS["v_min"])),
            "v_max": float(config.get("v_max", DEFAULT_PARAMS["v_max"])),
            "omega_max": float(config.get("omega_max", DEFAULT_PARAMS["omega_max"])),
            "w_pos": float(weights.get("w_pos", DEFAULT_PARAMS["w_pos"])),
            "w_theta": float(weights.get("w_theta", DEFAULT_PARAMS["w_theta"])),
            "w_u": float(weights.get("w_u", DEFAULT_PARAMS["w_u"])),
            "w_du": float(mpc_cfg.get("w_du", DEFAULT_PARAMS["w_du"])),
            "w_pos_T": float(mpc_cfg.get("w_pos_T", DEFAULT_PARAMS["w_pos_T"])),
            "w_theta_T": float(mpc_cfg.get("w_theta_T", DEFAULT_PARAMS["w_theta_T"])),
            "solver_opts": solver_opts,
        }

    # @override
    def get_action(self, observation: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Return control action [v, omega] for the provided state."""
        x0 = np.asarray(observation, dtype=float).reshape(3)
        try:
            X, U = solve_mpc(
                x0=x0,
                goal=self._goal,
                corridor_params=self._corridor,
                params=self._params,
            )
        except Exception:
            return np.array([0.0, 0.0], dtype=float), None, None

        if U.shape[0] == 0 or np.any(np.isnan(U[0])):
            return np.array([0.0, 0.0], dtype=float), None, None

        return (
            np.clip(
                np.asarray(U[0], dtype=float).reshape(2),
                self._action_min,
                self._action_max,
            ),
            X,
            U,
        )


# =======================
# Demo (run with: python3 nmpc_algorithm.py)
# =======================


def plot_trajectory(X: NDArray, corridor_params: Dict[str, Any], goal: NDArray):
    import matplotlib.pyplot as plt

    if np.any(np.isnan(X)):
        print("No valid solution to plot.")
        return

    px = X[:, 0]
    py = X[:, 1]

    xk = corridor_params["x_knots"]
    yl = corridor_params["y_low_knots"]
    yh = corridor_params["y_high_knots"]
    plt.plot(xk, yl, "b--", label="Lower boundary")
    plt.plot(xk, yh, "b--", label="Upper boundary")

    plt.plot(px, py, "r.-", label="Trajectory")

    plt.plot(X[0, 0], X[0, 1], "go", markersize=10, label="Start")
    plt.plot(goal[0], goal[1], "g*", markersize=12, label="Goal")

    plt.xlabel("px")
    plt.ylabel("py")
    plt.title("Dubins Car NMPC Trajectory in Corridor")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("nmpc.png")
    plt.show()


if __name__ == "__main__":
    X, U = solve_mpc(DEFAULT_X0, DEFAULT_GOAL, DEFAULT_CORRIDOR, DEFAULT_PARAMS)
    plot_trajectory(X, DEFAULT_CORRIDOR, DEFAULT_GOAL)
