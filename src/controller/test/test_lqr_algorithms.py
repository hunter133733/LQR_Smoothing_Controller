import math
import os
import sys

import numpy as np
import pytest

THIS_DIR = os.path.dirname(__file__)
CONTROLLER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
NAV_HELPERS_ROOT = os.path.abspath(
    os.path.join(THIS_DIR, "..", "..", "nav_helpers")
)
for _p in (CONTROLLER_ROOT, NAV_HELPERS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller.lqr_algorithm import LQRAlgorithm, LQRController
from controller.lqr_smoothing_algorithm import LQRSmoothingAlgorithm, LQRSmoothingController
from nav_helpers.trajectory import StateActionTrajectory


# ── reference trajectory builders ───────────────────────────────────────────

def _straight_ref(dt=0.1, n_steps=80, v=0.5):
    states = np.zeros((n_steps + 1, 3))
    actions = np.zeros((n_steps, 2))
    actions[:, 0] = v
    for k in range(n_steps):
        x, y, th = states[k]
        states[k + 1] = [x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th]
    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def _arc_ref(dt=0.1, n_steps=80, v=0.5, w=0.30):
    states = np.zeros((n_steps + 1, 3))
    actions = np.zeros((n_steps, 2))
    actions[:, 0] = v
    actions[:, 1] = w
    for k in range(n_steps):
        x, y, th = states[k]
        states[k + 1] = [x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt]
    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def _s_curve_ref(dt=0.1, n_steps=100, v=0.45):
    states = np.zeros((n_steps + 1, 3))
    actions = np.zeros((n_steps, 2))
    actions[:, 0] = v
    for k in range(n_steps):
        t = k * dt
        w = 0.45 * math.sin(0.9 * t)
        actions[k, 1] = w
        x, y, th = states[k]
        states[k + 1] = [x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt]
    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def _aggressive_curve_ref(dt=0.1, n_steps=80, v=0.5, w=0.90):
    """High-constant angular rate — stresses Δω smoothing."""
    states = np.zeros((n_steps + 1, 3))
    actions = np.zeros((n_steps, 2))
    actions[:, 0] = v
    actions[:, 1] = w
    for k in range(n_steps):
        x, y, th = states[k]
        states[k + 1] = [x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt]
    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def _simulate(controller, traj, x0):
    x = np.asarray(x0, dtype=float).reshape(3).copy()
    X, U = [x.copy()], []
    for _ in range(traj.actions.shape[0]):
        u, _, _ = controller.get_action(x, traj)
        u = np.asarray(u, dtype=float).reshape(2)
        x = np.array(
            [
                x[0] + u[0] * math.cos(x[2]) * traj.dt,
                x[1] + u[0] * math.sin(x[2]) * traj.dt,
                x[2] + u[1] * traj.dt,
            ],
            dtype=float,
        )
        X.append(x.copy())
        U.append(u.copy())
    return np.asarray(X), np.asarray(U)


def _base_config():
    return {
        "dt": 0.1,
        "goal": np.array([5.0, 0.0, 0.0], dtype=float),
        "reference": {"kind": "to_goal", "n_steps": 200},
        "lqr": {
            "horizon": 25,
            "x_cost": 5.0, "y_cost": 5.0, "theta_cost": 1.0,
            "v_cost": 0.3, "w_cost": 0.3,
            "v_min": -0.2, "v_max": 1.0, "w_min": -1.2, "w_max": 1.2,
        },
    }


def _smooth_config(r_prime_v=1.0, r_prime_w=1.0):
    cfg = _base_config()
    cfg["lqr_smooth"] = {"r_prime_v": r_prime_v, "r_prime_w": r_prime_w}
    return cfg


# ── LQRAlgorithm: gains & linearization ─────────────────────────────────────

def test_compute_gains_shapes():
    """Ks and Ps have correct shapes for a given horizon."""
    algo = LQRAlgorithm()
    n = 6
    As = np.tile(np.eye(3), (n, 1, 1))
    Bs = np.zeros((n, 3, 2))
    Ks, Ps = algo.compute_gains(As, Bs)
    assert Ks.shape == (n, 2, 3), "Ks has wrong shape"
    assert Ps.shape == (n + 1, 3, 3), "Ps has wrong shape"


def test_compute_gains_terminal_condition():
    """Terminal cost-to-go P[N] must equal the terminal cost matrix L."""
    algo = LQRAlgorithm()
    n = 6
    As = np.tile(np.eye(3), (n, 1, 1))
    Bs = np.zeros((n, 3, 2))
    _, Ps = algo.compute_gains(As, Bs)
    assert np.allclose(Ps[n], algo.L), "P[N] must equal L (Riccati boundary condition)"


def test_compute_gains_finite():
    """Gain and cost-to-go matrices contain no NaN or Inf."""
    algo = LQRAlgorithm()
    traj = _straight_ref()
    As, Bs = algo.linearize_along_traj(traj.states[: algo.n], traj.actions[: algo.n])
    Ks, Ps = algo.compute_gains(As, Bs)
    assert np.all(np.isfinite(Ks)), "Ks contains NaN/Inf"
    assert np.all(np.isfinite(Ps)), "Ps contains NaN/Inf"


# ── LQRAlgorithm: trajectory tracking ───────────────────────────────────────

def test_straight_tracking_error_reduces():
    """Final position error is smaller than initial offset on a straight reference."""
    traj = _straight_ref()
    x0 = np.array([0.0, 0.20, 0.10], dtype=float)
    ctrl = LQRController(_base_config())
    X, _ = _simulate(ctrl, traj, x0)
    ref_X = traj.states[: X.shape[0]]
    init_err = np.linalg.norm(x0[:2] - ref_X[0, :2])
    final_err = np.linalg.norm(X[-1, :2] - ref_X[-1, :2])
    assert final_err < init_err, "Controller did not reduce position error on straight reference"


def test_arc_tracking_bounded_error():
    """Mean position error on arc reference stays below 0.30 m."""
    traj = _arc_ref()
    x0 = np.array([0.0, -0.20, -0.10], dtype=float)
    ctrl = LQRController(_base_config())
    X, _ = _simulate(ctrl, traj, x0)
    ref_X = traj.states[: X.shape[0]]
    mean_err = float(np.mean(np.linalg.norm(X[:, :2] - ref_X[:, :2], axis=1)))
    assert mean_err < 0.30, f"Mean arc tracking error {mean_err:.3f} m exceeds 0.30 m"


def test_large_offset_recovery():
    """Controller reduces position error even starting 1.0 m off the reference."""
    traj = _straight_ref()
    x0 = np.array([0.0, 1.0, 0.0], dtype=float)
    ctrl = LQRController(_base_config())
    X, _ = _simulate(ctrl, traj, x0)
    ref_X = traj.states[: X.shape[0]]
    init_err = np.linalg.norm(x0[:2] - ref_X[0, :2])
    final_err = np.linalg.norm(X[-1, :2] - ref_X[-1, :2])
    assert final_err < init_err, "Controller did not recover from 1.0 m initial offset"


def test_control_within_bounds():
    """All issued commands satisfy the declared v/w limits."""
    traj = _s_curve_ref()
    x0 = np.array([0.0, 0.10, 0.05], dtype=float)
    ctrl = LQRController(_base_config())
    _, U = _simulate(ctrl, traj, x0)
    assert np.all(U[:, 0] >= -0.2 - 1e-9), "v undershot v_min"
    assert np.all(U[:, 0] <= 1.0 + 1e-9), "v exceeded v_max"
    assert np.all(U[:, 1] >= -1.2 - 1e-9), "w undershot w_min"
    assert np.all(U[:, 1] <= 1.2 + 1e-9), "w exceeded w_max"


# ── LQRSmoothingAlgorithm: augmented-state gains ─────────────────────────────

def test_augmented_gains_shapes():
    """Ks has shape (n, 2, 5) and Ps has shape (n+1, 5, 5) for augmented system."""
    algo = LQRSmoothingAlgorithm(r_prime_v=1.0, r_prime_w=1.0)
    n = 6
    As = np.tile(np.eye(3), (n, 1, 1))
    Bs = np.zeros((n, 3, 2))
    Ks, Ps = algo.compute_gains(As, Bs)
    assert Ks.shape == (n, 2, 5), "Augmented Ks must have shape (n, 2, 5)"
    assert Ps.shape == (n + 1, 5, 5), "Augmented Ps must have shape (n+1, 5, 5)"


def test_augmented_terminal_condition():
    """Terminal cost-to-go P[N] must equal L_aug."""
    algo = LQRSmoothingAlgorithm(r_prime_v=1.0, r_prime_w=1.0)
    n = 6
    As = np.tile(np.eye(3), (n, 1, 1))
    Bs = np.zeros((n, 3, 2))
    _, Ps = algo.compute_gains(As, Bs)
    assert np.allclose(Ps[n], algo.L_aug), "P[N] must equal L_aug"


def test_augmented_gains_finite():
    """Augmented gains contain no NaN or Inf."""
    algo = LQRSmoothingAlgorithm(r_prime_v=1.0, r_prime_w=1.0)
    traj = _straight_ref()
    As, Bs = algo.linearize_along_traj(traj.states[: algo.n], traj.actions[: algo.n])
    Ks, Ps = algo.compute_gains(As, Bs)
    assert np.all(np.isfinite(Ks)), "Augmented Ks contains NaN/Inf"
    assert np.all(np.isfinite(Ps)), "Augmented Ps contains NaN/Inf"


def test_zero_r_prime_matches_standard_lqr_gains():
    """With R'=0, augmented gains for the original state block equal standard LQR gains."""
    traj = _straight_ref()
    n = 8
    As = np.tile(np.eye(3), (n, 1, 1))
    Bs = np.zeros((n, 3, 2))

    std = LQRAlgorithm()
    Ks_std, _ = std.compute_gains(As, Bs)

    aug = LQRSmoothingAlgorithm(r_prime_v=0.0, r_prime_w=0.0)
    Ks_aug, _ = aug.compute_gains(As, Bs)

    # Augmented K[:, :, :3] should match standard K (original state block)
    assert np.allclose(Ks_aug[:, :, :3], Ks_std, atol=1e-6), (
        "With R'=0 the augmented gains (state block) must equal standard LQR gains"
    )


def test_update_prev_u_stores_correctly():
    """update_prev_u() must store the provided command in prev_u."""
    algo = LQRSmoothingAlgorithm()
    cmd = np.array([0.4, -0.3], dtype=float)
    algo.update_prev_u(cmd)
    assert np.allclose(algo.prev_u, cmd), "prev_u not updated by update_prev_u()"


def test_solve_does_not_mutate_prev_u():
    """solve() must not change prev_u — only update_prev_u() should do that."""
    traj = _straight_ref()
    algo = LQRSmoothingAlgorithm()
    z_ref = traj.states[: algo.n]
    u_ref = traj.actions[: algo.n]
    initial_prev = np.array([0.3, 0.1], dtype=float)
    algo.prev_u = initial_prev.copy()
    algo.solve(z_0=np.zeros(3), t_0=0.0, z_ref=z_ref, u_ref=u_ref)
    assert np.allclose(algo.prev_u, initial_prev), "solve() must not mutate prev_u"


# ── LQRSmoothingController: end-to-end tests ────────────────────────────────

def test_smoothing_reduces_dw_vs_baseline():
    """Higher r_prime_w gives strictly lower max Δω than baseline on aggressive curve."""
    traj = _aggressive_curve_ref()
    x0 = np.array([0.0, 0.05, 0.0], dtype=float)

    baseline = LQRController(_base_config())
    smooth = LQRSmoothingController(_smooth_config(r_prime_v=0.25, r_prime_w=1.0))

    _, U_base = _simulate(baseline, traj, x0)
    _, U_smooth = _simulate(smooth, traj, x0)

    max_dw_base = float(np.max(np.abs(np.diff(U_base[:, 1]))))
    max_dw_smooth = float(np.max(np.abs(np.diff(U_smooth[:, 1]))))

    assert max_dw_smooth < max_dw_base, (
        f"Smoothing did not reduce max Δω: baseline={max_dw_base:.4f}, "
        f"smooth={max_dw_smooth:.4f}"
    )


def test_higher_r_prime_gives_lower_dw():
    """Increasing r_prime_w monotonically decreases max Δω on aggressive curve."""
    traj = _aggressive_curve_ref()
    x0 = np.array([0.0, 0.0, 0.0], dtype=float)

    max_dws = []
    for r_w in [0.0, 0.1, 1.0, 10.0]:
        ctrl = LQRSmoothingController(_smooth_config(r_prime_v=r_w / 4.0, r_prime_w=r_w))
        _, U = _simulate(ctrl, traj, x0)
        max_dws.append(float(np.max(np.abs(np.diff(U[:, 1])))))

    for i in range(len(max_dws) - 1):
        assert max_dws[i] >= max_dws[i + 1], (
            f"max Δω did not decrease as r_prime_w increased: "
            f"{max_dws[i]:.4f} → {max_dws[i+1]:.4f}"
        )


def test_smoothing_controller_within_bounds():
    """LQRSmoothingController commands satisfy v/w limits."""
    traj = _aggressive_curve_ref()
    x0 = np.array([0.0, 0.0, 0.0], dtype=float)
    ctrl = LQRSmoothingController(_smooth_config(r_prime_v=0.25, r_prime_w=1.0))
    _, U = _simulate(ctrl, traj, x0)
    assert np.all(U[:, 0] >= -0.2 - 1e-9), "v undershot v_min"
    assert np.all(U[:, 0] <= 1.0 + 1e-9), "v exceeded v_max"
    assert np.all(U[:, 1] >= -1.2 - 1e-9), "w undershot w_min"
    assert np.all(U[:, 1] <= 1.2 + 1e-9), "w exceeded w_max"


def test_s_curve_final_error():
    """End position error on s-curve (near-180° heading start) is within 0.5 m."""
    traj = _s_curve_ref()
    x0 = np.array([0.0, 0.15, 3.13], dtype=float)
    ctrl = LQRSmoothingController(_smooth_config(r_prime_v=0.25, r_prime_w=1.0))
    X, _ = _simulate(ctrl, traj, x0)
    ref_X = traj.states[: X.shape[0]]
    final_err = np.linalg.norm(X[-1, :2] - ref_X[-1, :2])
    assert final_err < 0.5, f"Final s-curve error {final_err:.3f} m exceeds 0.5 m"


def test_large_offset_s_curve_recovery():
    """LQRSmoothingController reduces tracking error from 0.6 m initial offset."""
    traj = _s_curve_ref()
    x0 = np.array([0.0, 0.60, 0.0], dtype=float)
    ctrl = LQRSmoothingController(_smooth_config(r_prime_v=0.25, r_prime_w=1.0))
    X, _ = _simulate(ctrl, traj, x0)
    ref_X = traj.states[: X.shape[0]]
    init_err = np.linalg.norm(x0[:2] - ref_X[0, :2])
    final_err = np.linalg.norm(X[-1, :2] - ref_X[-1, :2])
    assert final_err < init_err, (
        f"Controller did not recover from large s-curve offset "
        f"(init={init_err:.3f} m, final={final_err:.3f} m)"
    )
