"""
Experiment 1: Waypoint / Reference Tracking
============================================
Compares Baseline LQR vs Smoothed LQR (λ ∈ {0.1, 1.0, 10.0}) and the
post-filter LQRSmoothingController on two reference trajectories:
  1. straight_turn  — straight line into a 90° left turn, then straight
  2. figure8        — two full circles in opposite directions

λ maps to the SmoothedLQRController's angular-rate penalty (dw_cost = λ,
dv_cost = λ/4).  λ = 0 is plain LQRController (no smoothing).

Outputs written to  results/experiment1/<traj_name>/
  <controller>_steps.csv   — per-step pose, commands, errors, Δu, accel proxy
  summary.csv              — one row per controller, aggregate metrics
  plots/                   — path, error, v/ω commands, Δv/Δω, bar charts

Run:
  python3 src/experiment_waypoint_tracking.py
"""

import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── sys.path ─────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.join(_SRC, "controller"),
    os.path.join(_SRC, "nav_helpers"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller.lqr_algorithm import LQRController
from controller.lqr_smoothing_algorithm_old import LQRSmoothingController
from nav_helpers.trajectory import StateActionTrajectory

# ── constants ─────────────────────────────────────────────────────────────────
DT = 0.1
V_REF = 0.5
OUT_DIR = os.path.join(os.path.dirname(_SRC), "results", "experiment1")


# ── reference trajectories ───────────────────────────────────────────────────

def make_straight_turn_ref(
    dt: float = DT,
    v: float = V_REF,
    n_straight1: int = 50,
    n_turn: int = 20,
    n_straight2: int = 40,
) -> StateActionTrajectory:
    """Straight → 90° left turn → straight.

    The turn uses w = (π/2) / (n_turn * dt) so the robot completes exactly 90°
    over the turn segment while maintaining constant forward speed.
    """
    n_total = n_straight1 + n_turn + n_straight2
    states = np.zeros((n_total + 1, 3), dtype=float)
    actions = np.zeros((n_total, 2), dtype=float)

    w_turn = (math.pi / 2.0) / (n_turn * dt)

    actions[:n_straight1, 0] = v
    actions[:n_straight1, 1] = 0.0

    actions[n_straight1 : n_straight1 + n_turn, 0] = v
    actions[n_straight1 : n_straight1 + n_turn, 1] = w_turn

    actions[n_straight1 + n_turn :, 0] = v
    actions[n_straight1 + n_turn :, 1] = 0.0

    for k in range(n_total):
        x, y, th = states[k]
        states[k + 1] = [
            x + actions[k, 0] * math.cos(th) * dt,
            y + actions[k, 0] * math.sin(th) * dt,
            th + actions[k, 1] * dt,
        ]

    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def make_figure8_ref(
    dt: float = DT,
    v: float = 0.4,
    n_per_loop: int = 80,
) -> StateActionTrajectory:
    """Two full circles in opposite directions forming a figure-8.

    Angular rate w = 2π / (n_per_loop * dt) keeps speed constant while
    completing one full revolution per loop segment.
    """
    n_total = 2 * n_per_loop
    states = np.zeros((n_total + 1, 3), dtype=float)
    actions = np.zeros((n_total, 2), dtype=float)

    w_loop = (2.0 * math.pi) / (n_per_loop * dt)

    actions[:n_per_loop, 0] = v
    actions[:n_per_loop, 1] = w_loop
    actions[n_per_loop:, 0] = v
    actions[n_per_loop:, 1] = -w_loop

    for k in range(n_total):
        x, y, th = states[k]
        states[k + 1] = [
            x + v * math.cos(th) * dt,
            y + v * math.sin(th) * dt,
            th + actions[k, 1] * dt,
        ]

    return StateActionTrajectory(states=states, actions=actions, dt=dt)


# ── controller configurations ─────────────────────────────────────────────────

def _base_cfg() -> dict:
    return {
        "dt": DT,
        "goal": np.array([10.0, 5.0, 0.0], dtype=float),
        "reference": {"kind": "to_goal", "n_steps": 500},
        "lqr": {
            "horizon": 25,
            "x_cost": 5.0,
            "y_cost": 5.0,
            "theta_cost": 1.0,
            "v_cost": 0.3,
            "w_cost": 0.3,
            "v_min": -0.2,
            "v_max": 1.0,
            "w_min": -1.2,
            "w_max": 1.2,
        },
    }


def _smooth_cfg(r_prime_v: float = 0.25, r_prime_w: float = 1.0) -> dict:
    """LQRSmoothingController config using augmented-state Δu penalty.

    λ = r_prime_w (penalty on Δω in the augmented Riccati cost).
    r_prime_v = λ/4 by convention.  λ=0 → standard LQR.
    """
    cfg = _base_cfg()
    cfg["lqr_smooth"] = {
        "r_prime_v": r_prime_v,
        "r_prime_w": r_prime_w,
    }
    return cfg


# name, class, config, plot color
CONTROLLERS = [
    ("baseline_lqr (λ=0)", LQRController,          _base_cfg(),             "#1f77b4"),
    ("smoothed_lam0.1",     LQRSmoothingController, _smooth_cfg(0.025, 0.1), "#ff7f0e"),
    ("smoothed_lam1.0",     LQRSmoothingController, _smooth_cfg(0.25,  1.0), "#2ca02c"),
    ("smoothed_lam10.0",    LQRSmoothingController, _smooth_cfg(2.5,  10.0), "#d62728"),
]


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(
    controller, traj: StateActionTrajectory, x0: np.ndarray
):
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
    return np.asarray(X, dtype=float), np.asarray(U, dtype=float)


# ── metrics ────────────────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def compute_step_rows(
    X: np.ndarray, U: np.ndarray, traj: StateActionTrajectory
) -> list:
    """Build per-step dict rows including Δu and acceleration proxy."""
    n = U.shape[0]
    ref_X = traj.states[: n + 1]
    rows = []
    prev_u = np.zeros(2, dtype=float)

    for i in range(n):
        pos_err = float(np.linalg.norm(X[i, :2] - ref_X[i, :2]))
        yaw_err = _wrap(X[i, 2] - ref_X[i, 2])
        delta_v = float(U[i, 0]) - float(prev_u[0])
        delta_w = float(U[i, 1]) - float(prev_u[1])
        rows.append(
            {
                "step": i,
                "t_sec": round(i * traj.dt, 3),
                "x": round(float(X[i, 0]), 6),
                "y": round(float(X[i, 1]), 6),
                "yaw": round(float(X[i, 2]), 6),
                "ref_x": round(float(ref_X[i, 0]), 6),
                "ref_y": round(float(ref_X[i, 1]), 6),
                "ref_yaw": round(float(ref_X[i, 2]), 6),
                "pos_err": round(pos_err, 6),
                "yaw_err": round(yaw_err, 6),
                "v_cmd": round(float(U[i, 0]), 6),
                "w_cmd": round(float(U[i, 1]), 6),
                "delta_v": round(delta_v, 6),
                "delta_w": round(delta_w, 6),
                # Δu/dt approximates linear/angular acceleration (m/s², rad/s²)
                "lin_accel": round(delta_v / traj.dt, 6),
                "ang_accel": round(delta_w / traj.dt, 6),
            }
        )
        prev_u = U[i].copy()

    return rows


def compute_summary(
    name: str, X: np.ndarray, U: np.ndarray, traj: StateActionTrajectory
) -> dict:
    n = U.shape[0]
    ref_X = traj.states[: n + 1]

    pos_err = np.linalg.norm(X[:n, :2] - ref_X[:n, :2], axis=1)
    rmse_pos = float(np.sqrt(np.mean(pos_err**2)))
    max_pos_err = float(np.max(pos_err))

    dU = np.diff(U, axis=0)
    total_var_v = float(np.sum(np.abs(dU[:, 0])))
    total_var_w = float(np.sum(np.abs(dU[:, 1])))
    max_dv = float(np.max(np.abs(dU[:, 0])))
    max_dw = float(np.max(np.abs(dU[:, 1])))
    rms_dw = float(np.sqrt(np.mean(dU[:, 1] ** 2)))

    return {
        "controller": name,
        "rmse_pos_m": round(rmse_pos, 6),
        "max_pos_err_m": round(max_pos_err, 6),
        "total_var_v": round(total_var_v, 6),
        "total_var_w": round(total_var_w, 6),
        "max_dv": round(max_dv, 6),
        "max_dw": round(max_dw, 6),
        "rms_dw": round(rms_dw, 6),
    }


# ── CSV ────────────────────────────────────────────────────────────────────────

STEP_FIELDS = [
    "step", "t_sec",
    "x", "y", "yaw",
    "ref_x", "ref_y", "ref_yaw",
    "pos_err", "yaw_err",
    "v_cmd", "w_cmd",
    "delta_v", "delta_w",
    "lin_accel", "ang_accel",
]

SUMMARY_FIELDS = [
    "controller",
    "rmse_pos_m", "max_pos_err_m",
    "total_var_v", "total_var_w",
    "max_dv", "max_dw", "rms_dw",
]


def _write_csv(path: str, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── plots ──────────────────────────────────────────────────────────────────────

def _savefig(fig, filename: str, plotdir: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, filename), dpi=150)
    plt.close(fig)


def plot_path(traj, run_data: list, plotdir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(traj.states[:, 0], traj.states[:, 1], "k--", lw=2.5, label="reference")
    for name, X, _, color in run_data:
        ax.plot(X[:, 0], X[:, 1], lw=1.8, label=name, color=color)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Path comparison")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)
    _savefig(fig, "path.png", plotdir)


def plot_timeseries(
    run_rows: list, field: str, ylabel: str, title: str, filename: str, plotdir: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, rows, color in run_rows:
        ts = [r["t_sec"] for r in rows]
        ys = [r[field] for r in rows]
        ax.plot(ts, ys, lw=1.6, label=name, color=color)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=8)
    _savefig(fig, filename, plotdir)


def plot_bar(
    summaries: list, field: str, ylabel: str, title: str, filename: str, plotdir: str
) -> None:
    names = [s["controller"] for s in summaries]
    vals = [s[field] for s in summaries]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, vals, color=colors[: len(names)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=18)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _savefig(fig, filename, plotdir)


def plot_delta_w_comparison(run_rows: list, plotdir: str) -> None:
    """Smoking-gun plot: Δω per timestep for every controller."""
    fig, axes = plt.subplots(
        len(run_rows), 1, figsize=(9, 2.2 * len(run_rows)), sharex=True
    )
    if len(run_rows) == 1:
        axes = [axes]
    for ax, (name, rows, color) in zip(axes, run_rows):
        ts = [r["t_sec"] for r in rows]
        dw = [r["delta_w"] for r in rows]
        ax.plot(ts, dw, lw=1.4, color=color)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("Δω (rad/s)", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(True)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Δω per timestep — smoothness comparison", fontsize=10)
    _savefig(fig, "delta_w_subplots.png", plotdir)


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    traj_name: str, traj: StateActionTrajectory, x0: np.ndarray
) -> None:
    traj_outdir = os.path.join(OUT_DIR, traj_name)
    plotdir = os.path.join(traj_outdir, "plots")
    os.makedirs(plotdir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Trajectory: {traj_name}  ({traj.actions.shape[0]} steps, dt={traj.dt}s)")
    print(f"{'='*62}")
    print(f"  {'controller':26s}  {'RMSE(m)':>9s}  {'TV_ω':>9s}  {'max_Δω':>9s}  {'rms_Δω':>9s}")
    print(f"  {'-'*26}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    run_data = []   # (name, X, U, color)
    run_rows = []   # (name, rows, color)
    summaries = []

    for ctrl_name, ctrl_cls, cfg, color in CONTROLLERS:
        ctrl = ctrl_cls(cfg)
        X, U = simulate(ctrl, traj, x0)
        rows = compute_step_rows(X, U, traj)
        summary = compute_summary(ctrl_name, X, U, traj)

        run_data.append((ctrl_name, X, U, color))
        run_rows.append((ctrl_name, rows, color))
        summaries.append(summary)

        _write_csv(
            os.path.join(traj_outdir, f"{ctrl_name}_steps.csv"),
            STEP_FIELDS,
            rows,
        )

        print(
            f"  {ctrl_name:26s}  {summary['rmse_pos_m']:9.4f}  "
            f"{summary['total_var_w']:9.4f}  {summary['max_dw']:9.4f}  "
            f"{summary['rms_dw']:9.4f}"
        )

    _write_csv(os.path.join(traj_outdir, "summary.csv"), SUMMARY_FIELDS, summaries)

    # ── plots ──
    plot_path(traj, run_data, plotdir)

    plot_timeseries(run_rows, "pos_err",   "position error (m)",   "Position tracking error",            "pos_error.png",  plotdir)
    plot_timeseries(run_rows, "v_cmd",     "v (m/s)",              "Linear velocity commands",           "v_cmd.png",      plotdir)
    plot_timeseries(run_rows, "w_cmd",     "ω (rad/s)",            "Angular velocity commands",          "w_cmd.png",      plotdir)
    plot_timeseries(run_rows, "delta_v",   "Δv (m/s)",             "Δv per step",                        "delta_v.png",    plotdir)
    plot_timeseries(run_rows, "delta_w",   "Δω (rad/s)",           "Δω per step (smoothness)",           "delta_w.png",    plotdir)
    plot_timeseries(run_rows, "ang_accel", "α (rad/s²)",           "Angular acceleration proxy (Δω/dt)", "ang_accel.png",  plotdir)

    plot_delta_w_comparison(run_rows, plotdir)

    plot_bar(summaries, "rmse_pos_m",   "RMSE (m)",       "Position tracking RMSE",          "bar_rmse.png",    plotdir)
    plot_bar(summaries, "total_var_w",  "Σ|Δω| (rad/s)", "Total variation — angular rate",  "bar_tv_w.png",    plotdir)
    plot_bar(summaries, "total_var_v",  "Σ|Δv| (m/s)",   "Total variation — linear speed",  "bar_tv_v.png",    plotdir)
    plot_bar(summaries, "max_dw",       "max Δω (rad/s)", "Max single-step Δω",              "bar_max_dw.png",  plotdir)
    plot_bar(summaries, "rms_dw",       "RMS Δω (rad/s)", "RMS Δω (average jerk)",           "bar_rms_dw.png",  plotdir)

    print(f"\n  Output: {traj_outdir}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # start aligned with reference (offset can be added here to stress-test recovery)
    x0 = np.array([0.0, 0.0, 0.0], dtype=float)

    run_experiment("straight_turn", make_straight_turn_ref(), x0)
    run_experiment("figure8",       make_figure8_ref(),       x0)

    print(f"\nAll results → {OUT_DIR}")


if __name__ == "__main__":
    main()
