from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from controller.dubins3d_2ctrls import DubinsCar3D2Ctrls

DEFAULT_REF_START = np.array([0.0, 0.0, 0.0], dtype=float)


def _generate_to_goal_trajectory(
    dyn: DubinsCar3D2Ctrls,
    start: NDArray,
    goal: NDArray,
    n_steps: int,
    dt: float,
) -> Tuple[NDArray, NDArray]:
    """
    Generate a dynamically-feasible trajectory using a simple go-to-goal law.

    Returns:
      z_ref: (n_steps, 3) state trajectory
      u_ref: (n_steps, 2) control trajectory
    """
    z_ref = np.zeros((n_steps, 3), dtype=float)
    u_ref = np.zeros((n_steps, 2), dtype=float)

    k_rho = 0.9
    k_heading = 1.8
    v_max = float(dyn.u_max[0])
    w_max = float(dyn.u_max[1])
    goal_tol = 0.1

    dyn.reset(start.copy())
    z_ref[0] = start.copy()

    for k in range(n_steps - 1):
        x, y, th = dyn.z_t
        dx = goal[0] - x
        dy = goal[1] - y
        rho = float(np.hypot(dx, dy))

        if rho < goal_tol:
            heading_err = np.arctan2(np.sin(goal[2] - th), np.cos(goal[2] - th))
            v = 0.0
            w = float(np.clip(1.2 * heading_err, -w_max, w_max))
        else:
            desired_heading = np.arctan2(dy, dx)
            heading_err = np.arctan2(
                np.sin(desired_heading - th), np.cos(desired_heading - th)
            )
            v = float(np.clip(k_rho * rho, 0.0, v_max))
            w = float(np.clip(k_heading * heading_err, -w_max, w_max))

        u_ref[k] = np.array([v, w], dtype=float)
        z_ref[k + 1] = dyn.forward_np(dt=dt, ctrl=u_ref[k])

    u_ref[-1] = np.zeros(2, dtype=float)
    return z_ref, u_ref


def generate_reference_trajectory(
    kind: str = "s_curve",
    dt: float = 0.1,
    n_steps: int = 400,
    start_state: NDArray = DEFAULT_REF_START,
    goal_state: NDArray | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate a nominal Dubins reference trajectory.

    Returns:
      tau: (n_steps,) timestamps
      z_ref: (n_steps, 3) state trajectory
      u_ref: (n_steps, 2) control trajectory
    """
    n_steps = int(max(2, n_steps))
    dt = float(dt)
    start = np.asarray(start_state, dtype=float).reshape(3)

    dyn = DubinsCar3D2Ctrls(
        z_0=start.copy(),
        dt=dt,
        u_min=np.array([-0.2, -1.2], dtype=float),
        u_max=np.array([1.0, 1.2], dtype=float),
        d_min=np.zeros(3, dtype=float),
        d_max=np.zeros(3, dtype=float),
        u_mode="min",
        d_mode="max",
    )

    tau = np.arange(n_steps, dtype=float) * dt
    z_ref = np.zeros((n_steps, 3), dtype=float)
    u_ref = np.zeros((n_steps, 2), dtype=float)
    z_ref[0] = start

    if kind == "to_goal":
        if goal_state is None:
            goal = start.copy()
            goal[0] += 3.0
        else:
            goal = np.asarray(goal_state, dtype=float).reshape(3)

        z_ref, u_ref = _generate_to_goal_trajectory(dyn, start, goal, n_steps, dt)
        return tau, z_ref, u_ref

    for k in range(1, n_steps):
        t = tau[k - 1]
        if kind == "straight":
            v = 0.35
            w = 0.0
        else:
            # Smooth heading oscillation with positive forward speed.
            v = 0.35 + 0.05 * np.cos(0.1 * t)
            w = 0.35 * np.sin(0.25 * t)

        u = np.array([v, w], dtype=float)
        u_ref[k - 1] = u
        z_ref[k] = dyn.forward_np(dt=dt, ctrl=u)

    # Fill trailing control for shape consistency.
    u_ref[-1] = u_ref[-2]

    return tau, z_ref, u_ref
