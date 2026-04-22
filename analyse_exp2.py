"""
Experiment 2 — Step Response Post-Processing
=============================================
Reads the per-run CSVs already written by controller_node.py and produces
all plots and a summary CSV for the step-response experiment.

The node already logs:
    t, x, y, yaw, v_cmd, w_cmd   →   results/metrics/current_run.csv

Rename each run's CSV before running the next:
    cp results/metrics/current_run.csv results/metrics/exp2_baseline.csv
    cp results/metrics/current_run.csv results/metrics/exp2_light.csv
    cp results/metrics/current_run.csv results/metrics/exp2_medium.csv
    cp results/metrics/current_run.csv results/metrics/exp2_heavy.csv

Then run:
    python3 analyse_exp2.py

Outputs → results/experiment2/
    summary.csv
    plots/
        path.png
        w_cmd.png          — angular velocity over time (shows transient shape)
        delta_w.png        — Δω per step  (smoking-gun smoothness signal)
        ang_accel.png      — Δω/dt  (angular acceleration proxy / jerk)
        v_cmd.png          — linear velocity over time
        heading.png        — yaw over time  (shows rise time difference)
        bar_rise_time.png  — how long each controller takes to reach steady heading
        bar_max_dw.png     — peak Δω per step
        bar_rms_dw.png     — RMS Δω
        bar_total_var_w.png
        bar_rmse_pos.png
        pareto.png         — RMSE vs total variation of ω
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── configuration ─────────────────────────────────────────────────────────────

# Goal used in all exp2 YAMLs — needed to compute position error
GOAL = np.array([0.0, -3.0, -math.pi / 2.0], dtype=float)

# Where the node writes CSVs  (relative to this script's location)
METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "metrics")

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "experiment2")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

# Each entry: (display label, csv filename, plot colour)
RUNS = [
    ("Baseline LQR",     "exp2_baseline.csv", "#1f77b4"),
    ("Smoothed (light)",  "exp2_light.csv",   "#ff7f0e"),
    ("Smoothed (medium)", "exp2_medium.csv",  "#2ca02c"),
    ("Smoothed (heavy)",  "exp2_heavy.csv",   "#d62728"),
]

# Heading is considered "settled" when it is within this many radians of the
# goal heading for SETTLE_WINDOW consecutive samples.
HEADING_SETTLE_RAD    = 0.15   # rad  (~8.6°)
HEADING_SETTLE_WINDOW = 5      # consecutive samples


# ── helpers ───────────────────────────────────────────────────────────────────

def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def load_csv(path: str) -> dict:
    """Load node CSV → dict of numpy arrays keyed by column name."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    keys = list(rows[0].keys())
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def augment(data: dict, goal: np.ndarray, dt: float = 0.1) -> dict:
    """
    Add derived columns to the loaded data dict:
        delta_w, delta_v, ang_accel, lin_accel, pos_err, yaw_err
    dt is inferred from the timestamps if possible.
    """
    t   = data["t"]
    if len(t) > 1:
        dt = float(np.median(np.diff(t)))

    v = data["v_cmd"]
    w = data["w_cmd"]

    dv = np.diff(v, prepend=0.0)
    dw = np.diff(w, prepend=0.0)

    data["delta_v"]   = dv
    data["delta_w"]   = dw
    data["lin_accel"] = dv / dt
    data["ang_accel"] = dw / dt

    x   = data["x"]
    y   = data["y"]
    yaw = data["yaw"]

    data["pos_err"] = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    data["yaw_err"] = np.array([wrap(th - goal[2]) for th in yaw], dtype=float)

    return data


def heading_rise_time(t: np.ndarray, yaw_err: np.ndarray,
                      tol: float = HEADING_SETTLE_RAD,
                      window: int = HEADING_SETTLE_WINDOW) -> float:
    """
    Return the time (s) at which yaw_err first stays within ±tol for
    `window` consecutive samples.  Returns nan if never settled.
    """
    settled = np.abs(yaw_err) < tol
    for i in range(len(settled) - window + 1):
        if np.all(settled[i:i + window]):
            return float(t[i])
    return float("nan")


def compute_summary(label: str, data: dict) -> dict:
    dw = data["delta_w"]
    dv = data["delta_v"]
    pe = data["pos_err"]
    rt = heading_rise_time(data["t"], data["yaw_err"])

    return {
        "controller":    label,
        "rise_time_s":   round(rt, 3) if not math.isnan(rt) else "DNF",
        "rmse_pos_m":    round(float(np.sqrt(np.mean(pe**2))), 6),
        "max_pos_err_m": round(float(np.max(pe)), 6),
        "total_var_w":   round(float(np.sum(np.abs(dw))), 6),
        "total_var_v":   round(float(np.sum(np.abs(dv))), 6),
        "max_dw":        round(float(np.max(np.abs(dw))), 6),
        "rms_dw":        round(float(np.sqrt(np.mean(dw**2))), 6),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=150)
    plt.close(fig)


def plot_xy_path(runs_data):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*GOAL[:2], marker="*", s=200, color="black", zorder=5, label="goal")
    for label, data, color in runs_data:
        ax.plot(data["x"], data["y"], lw=1.8, label=label, color=color)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Exp 2 — Step Response: XY Path")
    ax.axis("equal"); ax.grid(True); ax.legend(fontsize=9)
    _savefig(fig, "path.png")


def plot_timeseries(runs_data, field, ylabel, title, filename, ref_line=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, data, color in runs_data:
        ax.plot(data["t"], data[field], lw=1.6, label=label, color=color)
    if ref_line is not None:
        ax.axhline(ref_line, color="black", lw=1.0, ls="--", label="goal")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True); ax.legend(fontsize=9)
    _savefig(fig, filename)


def plot_delta_w_subplots(runs_data):
    """One subplot per controller — the smoking-gun smoothness comparison."""
    n = len(runs_data)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, data, color) in zip(axes, runs_data):
        ax.plot(data["t"], data["delta_w"], lw=1.4, color=color)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("Δω (rad/s)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(True)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Δω per timestep — step response smoothness", fontsize=10)
    _savefig(fig, "delta_w_subplots.png")


def plot_bar(summaries, field, ylabel, title, filename, colors):
    labels = [s["controller"] for s in summaries]
    # rise_time may be "DNF" string — handle gracefully
    vals = []
    for s in summaries:
        v = s[field]
        vals.append(float(v) if v != "DNF" else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.tick_params(axis="x", rotation=12)
    for bar, val in zip(bars, vals):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    0.01,
                    "DNF", ha="center", va="bottom", fontsize=8, color="red")
    _savefig(fig, filename)


def plot_pareto(summaries, colors):
    fig, ax = plt.subplots(figsize=(6, 5))
    for s, color in zip(summaries, colors):
        tv = s["total_var_w"]
        rm = s["rmse_pos_m"]
        ax.scatter(tv, rm, color=color, s=100, zorder=3)
        ax.annotate(s["controller"], (tv, rm),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Σ|Δω|  [lower = smoother]")
    ax.set_ylabel("Position RMSE (m)  [lower = more accurate]")
    ax.set_title("Exp 2 — Smoothness / Accuracy Pareto")
    ax.grid(True)
    _savefig(fig, "pareto.png")


# ── CSV output ────────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "controller", "rise_time_s", "rmse_pos_m", "max_pos_err_m",
    "total_var_w", "total_var_v", "max_dw", "rms_dw",
]


def write_summary(summaries):
    path = os.path.join(OUT_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summaries)
    print(f"[summary] saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    runs_data  = []   # (label, data_dict, color)
    summaries  = []
    colors     = []
    missing    = []

    for label, fname, color in RUNS:
        path = os.path.join(METRICS_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} not found — run that experiment first.")
            missing.append(label)
            continue

        data = load_csv(path)
        data = augment(data, GOAL)
        runs_data.append((label, data, color))
        summaries.append(compute_summary(label, data))
        colors.append(color)

    if not runs_data:
        print("No experiment CSVs found. Nothing to plot.")
        print(f"Expected files in: {METRICS_DIR}")
        return

    if missing:
        print(f"Warning: missing runs — {missing}. Plots will only include available data.")

    # ── plots ──
    plot_xy_path(runs_data)

    plot_timeseries(runs_data, "w_cmd",    "ω (rad/s)",   "Angular velocity — step response",          "w_cmd.png",    ref_line=None)
    plot_timeseries(runs_data, "v_cmd",    "v (m/s)",     "Linear velocity — step response",           "v_cmd.png",    ref_line=None)
    plot_timeseries(runs_data, "delta_w",  "Δω (rad/s)",  "Δω per step  [smoothness signal]",          "delta_w.png",  ref_line=0.0)
    plot_timeseries(runs_data, "ang_accel","α (rad/s²)",  "Angular acceleration proxy  (Δω / dt)",     "ang_accel.png",ref_line=0.0)
    plot_timeseries(runs_data, "yaw",      "yaw (rad)",   "Heading over time",                         "heading.png",  ref_line=GOAL[2])
    plot_timeseries(runs_data, "pos_err",  "error (m)",   "Distance to goal over time",                "pos_err.png",  ref_line=0.0)

    plot_delta_w_subplots(runs_data)

    plot_bar(summaries, "rise_time_s",   "rise time (s)",     "Heading rise time to ±0.15 rad",   "bar_rise_time.png",    colors)
    plot_bar(summaries, "max_dw",        "max |Δω| (rad/s)",  "Peak angular rate step",           "bar_max_dw.png",       colors)
    plot_bar(summaries, "rms_dw",        "RMS Δω (rad/s)",    "RMS Δω  (average jerk)",           "bar_rms_dw.png",       colors)
    plot_bar(summaries, "total_var_w",   "Σ|Δω| (rad/s)",     "Total variation — angular rate",   "bar_total_var_w.png",  colors)
    plot_bar(summaries, "rmse_pos_m",    "RMSE (m)",          "Position tracking RMSE",           "bar_rmse_pos.png",     colors)

    plot_pareto(summaries, colors)

    write_summary(summaries)

    # ── console summary table ──
    print(f"\n{'='*72}")
    print(f"  Experiment 2 — Step Response Summary")
    print(f"{'='*72}")
    print(f"  {'controller':22s}  {'rise(s)':>7s}  {'RMSE(m)':>8s}  "
          f"{'TV_ω':>8s}  {'max_Δω':>8s}  {'rms_Δω':>8s}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for s in summaries:
        rt = f"{s['rise_time_s']:7.3f}" if s["rise_time_s"] != "DNF" else "    DNF"
        print(f"  {s['controller']:22s}  {rt}  "
              f"{s['rmse_pos_m']:8.4f}  {s['total_var_w']:8.4f}  "
              f"{s['max_dw']:8.4f}  {s['rms_dw']:8.4f}")
    print(f"\n  Plots → {PLOT_DIR}")


if __name__ == "__main__":
    main()
