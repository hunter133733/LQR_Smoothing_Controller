import math
import os
import sys

import casadi as ca

# Make sure we can import nmpc_algorithm.py from the parent directory (project root)
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mpc.nmpc_algorithm import dubins_step, solve_mpc


# -----------------------
# Small helpers (no numpy)
# -----------------------
def _shape(A):
    """Return (rows, cols) for list-of-lists, numpy arrays, or CasADi DM."""
    if hasattr(A, "shape"):
        sh = A.shape
        if len(sh) == 1:
            return int(sh[0]), 1
        return int(sh[0]), int(sh[1])
    if hasattr(A, "size1") and hasattr(A, "size2"):
        return int(A.size1()), int(A.size2())
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return rows, cols


def _get(A, i, j):
    """Get element A[i,j] for list-of-lists / numpy arrays / CasADi DM."""
    try:
        return A[i][j]
    except Exception:
        return A[i, j]


def _is_close(a, b, atol):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(atol))


def _assert_allclose_vec(actual, expected, atol, msg_prefix=""):
    # No parameterized values in messages (by request)
    assert len(actual) == len(expected), f"{msg_prefix} output has an incorrect size."
    for a, b in zip(actual, expected):
        assert _is_close(a, b, atol), f"{msg_prefix} output is incorrect."


def _all_finite_matrix(A):
    r, c = _shape(A)
    for i in range(r):
        for j in range(c):
            v = _get(A, i, j)
            if not math.isfinite(float(v)):
                return False
    return True


# -----------------------
# Test constants (grader-owned)
# -----------------------
DEFAULT_PARAMS = {
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
}

DEFAULT_CORRIDOR = {
    "x_knots": [0.0, 2.0, 4.0, 6.0, 8.0],
    "y_low_knots": [-1.0, -1.4, -1.6, -1.6, -1.4],
    "y_high_knots": [1.0, 1.4, 1.6, 1.6, 1.4],
}

DEFAULT_X0 = [0.5, 0.0, 0.0]
DEFAULT_GOAL = [7.5, 0.0, 0.0]


# -----------------------
# Piecewise-linear helper for corridor bounds (grader-owned)
# -----------------------
def pwl_clamped(x: float, x_knots, y_knots) -> float:
    xk = list(map(float, x_knots))
    yk = list(map(float, y_knots))

    if x <= xk[0]:
        return yk[0]
    if x >= xk[-1]:
        return yk[-1]

    for i in range(len(xk) - 1):
        x0, x1 = xk[i], xk[i + 1]
        if x0 <= x < x1:
            y0, y1 = yk[i], yk[i + 1]
            a = (x - x0) / (x1 - x0)
            return y0 + a * (y1 - y0)

    return yk[-1]


def y_low(px: float, corridor) -> float:
    return pwl_clamped(px, corridor["x_knots"], corridor["y_low_knots"])


def y_high(px: float, corridor) -> float:
    return pwl_clamped(px, corridor["x_knots"], corridor["y_high_knots"])


# -----------------------
# Tests
# -----------------------
def test_dubins_step_correctness_simple():
    """Direct unit test for dubins_step."""
    dt = 0.1
    xk = ca.DM([1.0, 2.0, 0.0])
    uk = ca.DM([1.5, -0.2])

    xkp1 = dubins_step(xk, uk, dt)
    xkp1_dm = ca.DM(xkp1)
    xkp1_list = [float(xkp1_dm[i]) for i in range(int(xkp1_dm.numel()))]

    expected = [
        1.0 + 1.5 * 1.0 * dt,
        2.0 + 1.5 * 0.0 * dt,
        0.0 + (-0.2) * dt,
    ]

    _assert_allclose_vec(xkp1_list, expected, atol=1e-9, msg_prefix="dubins_step: ")


def test_solve_mpc_shapes_and_finite():
    X, U = solve_mpc(DEFAULT_X0, DEFAULT_GOAL, DEFAULT_CORRIDOR, DEFAULT_PARAMS)
    N = int(DEFAULT_PARAMS["N"])

    xr, xc = _shape(X)
    ur, uc = _shape(U)

    assert (xr, xc) == (N + 1, 3), "X has the wrong shape."
    assert (ur, uc) == (N, 2), "U has the wrong shape."

    assert _all_finite_matrix(X), "X contains NaN/Inf (solver likely failed)."
    assert _all_finite_matrix(U), "U contains NaN/Inf (solver likely failed)."


def test_mpc_input_bounds():
    X, U = solve_mpc(DEFAULT_X0, DEFAULT_GOAL, DEFAULT_CORRIDOR, DEFAULT_PARAMS)
    N = int(DEFAULT_PARAMS["N"])

    assert _all_finite_matrix(X), "State trajectory contains invalid values."
    assert _all_finite_matrix(U), "Control trajectory contains invalid values."

    v_min = float(DEFAULT_PARAMS["v_min"])
    v_max = float(DEFAULT_PARAMS["v_max"])
    om_max = float(DEFAULT_PARAMS["omega_max"])
    tol_bound = 1e-6

    v_vals = [float(_get(U, k, 0)) for k in range(N)]
    om_vals = [float(_get(U, k, 1)) for k in range(N)]

    assert (
        min(v_vals) >= v_min - tol_bound
    ), "Control speed violates the specified bounds."
    assert (
        max(v_vals) <= v_max + tol_bound
    ), "Control speed violates the specified bounds."
    assert (
        min(om_vals) >= -om_max - tol_bound
    ), "Control turn-rate violates the specified bounds."
    assert (
        max(om_vals) <= om_max + tol_bound
    ), "Control turn-rate violates the specified bounds."


def test_mpc_corridor_and_dynamics():
    X, U = solve_mpc(DEFAULT_X0, DEFAULT_GOAL, DEFAULT_CORRIDOR, DEFAULT_PARAMS)
    N = int(DEFAULT_PARAMS["N"])
    dt = float(DEFAULT_PARAMS["dt"])

    assert _all_finite_matrix(X), "State trajectory contains invalid values."
    assert _all_finite_matrix(U), "Control trajectory contains invalid values."

    x_min = float(DEFAULT_CORRIDOR["x_knots"][0])
    x_max = float(DEFAULT_CORRIDOR["x_knots"][-1])

    tol_state = 2e-3
    tol_corr = 2e-3
    tol_bound = 1e-6

    # Corridor / position bounds
    for k in range(N + 1):
        px = float(_get(X, k, 0))
        py = float(_get(X, k, 1))

        assert (
            (x_min - tol_bound) <= px <= (x_max + tol_bound)
        ), "Position x is outside the corridor knot range."

        yl = y_low(px, DEFAULT_CORRIDOR)
        yh = y_high(px, DEFAULT_CORRIDOR)
        assert py >= yl - tol_corr, "Trajectory violates the corridor lower bound."
        assert py <= yh + tol_corr, "Trajectory violates the corridor upper bound."

    # Dynamics consistency
    for k in range(N):
        px = float(_get(X, k, 0))
        py = float(_get(X, k, 1))
        th = float(_get(X, k, 2))

        v = float(_get(U, k, 0))
        om = float(_get(U, k, 1))

        expected_next = [
            px + v * math.cos(th) * dt,
            py + v * math.sin(th) * dt,
            th + om * dt,
        ]

        actual_next = [float(_get(X, k + 1, j)) for j in range(3)]
        _assert_allclose_vec(
            actual_next, expected_next, atol=tol_state, msg_prefix="dynamics: "
        )


def test_mpc_goal_progress():
    X, U = solve_mpc(DEFAULT_X0, DEFAULT_GOAL, DEFAULT_CORRIDOR, DEFAULT_PARAMS)
    N = int(DEFAULT_PARAMS["N"])

    assert _all_finite_matrix(X), "State trajectory contains invalid values."
    assert _all_finite_matrix(U), "Control trajectory contains invalid values."

    dx = float(_get(X, N, 0)) - float(DEFAULT_GOAL[0])
    dy = float(_get(X, N, 1)) - float(DEFAULT_GOAL[1])
    final_pos_err = math.hypot(dx, dy)

    assert final_pos_err <= 1.0, "Final position is too far from the goal."
    assert (
        float(_get(X, N, 0)) >= 6.8
    ), "Trajectory did not make enough progress toward the goal."
