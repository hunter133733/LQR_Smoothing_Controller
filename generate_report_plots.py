"""
generate_report_plots.py

1. Make sure all CSV files are in results/metrics 
2. Run:  python3 generate_report_plots.py
3. Outputs go to:  /report_plots

"""

import csv as _csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "results/metrics")
OUT_DIR = os.path.join(SCRIPT_DIR, "report_plots")

# experiment 1 
EXP1_FILES = {
    "Baseline (0,0)": "exp1_baseline_true.csv",
    "Light (0.5,1)":  "exp1_light.csv",
    "Medium (1,2)":   "exp1_medium.csv",
    "Medium2 (3,5)":  "exp1_medium2.csv",
    "Heavy (10,20)":  "exp1_heavy.csv",
}
EXP1_COLORS = {
    "Baseline (0,0)": "#1f77b4",
    "Light (0.5,1)":  "#9467bd",
    "Medium (1,2)":   "#2ca02c",
    "Medium2 (3,5)":  "#ff7f0e",
    "Heavy (10,20)":  "#d62728",
}
GOAL_EXP1 = np.array([-2.0, -4.0])

# experiment 2
FIG8_FILES = {
    "Baseline (0,0)":   "fig8_baseline_true.csv",
    "Light (0.5,1)":    "fig8_light.csv",
    "Medium (1,2)":     "fig8_medium.csv",
    "Medium2 (3,5)":    "fig8_medium2.csv",
    "Heavy (10,20)":    "fig8_heavy.csv",
    "XHeavy (30,60)":   "fig8__30_60_.csv",
    "XXHeavy (50,100)": "fig8__50_100_.csv",
}
FIG8_COLORS = {
    "Baseline (0,0)":   "#1f77b4",
    "Light (0.5,1)":    "#9467bd",
    "Medium (1,2)":     "#2ca02c",
    "Medium2 (3,5)":    "#ff7f0e",
    "Heavy (10,20)":    "#d62728",
    "XHeavy (30,60)":   "#8c564b",
    "XXHeavy (50,100)": "#e377c2",
}
FIG8_LAMBDAS = [0, 1, 2, 5, 20, 60, 100]   # du_w cost values for x-axis


def load(filename: str, goal: np.ndarray) -> dict:
    path = os.path.join(CSV_DIR, filename)
    rows = []
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})

    t   = np.array([r["t"]     for r in rows]); t -= t[0]
    x   = np.array([r["x"]     for r in rows])
    y   = np.array([r["y"]     for r in rows])
    w   = np.array([r["w_cmd"] for r in rows])
    v   = np.array([r["v_cmd"] for r in rows])
    dw  = np.diff(w, prepend=w[0])
    pe  = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    pl  = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

    return {
        "t": t, "x": x, "y": y, "w": w, "v": v, "dw": dw, "pe": pe,
        "summary": {
            "tv_w":     round(float(np.sum(np.abs(dw))), 4),
            "max_dw":   round(float(np.max(np.abs(dw))), 4),
            "rms_dw":   round(float(np.sqrt(np.mean(dw**2))), 5),
            "path_len": round(pl, 3),
            "rmse":     round(float(np.sqrt(np.mean(pe**2))), 4),
        }
    }


def savefig(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_xy_paths(data, colors, goal_pt, goal_label, title, fname,
                  show_walls=False):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(*goal_pt, marker="*", s=350, c="black", zorder=6,
               label=goal_label)
    if show_walls:
        wall = plt.Polygon([(-5,-5),(-5,5),(5,5),(5,-5)],
                           fill=False, edgecolor='black', lw=2.5)
        ax.add_patch(wall)
    for label, d in data.items():
        ax.plot(d["x"], d["y"], lw=2.0, label=label,
                color=colors[label], alpha=0.85)
        ax.plot(d["x"][-1], d["y"][-1], "s",
                color=colors[label], ms=8, zorder=5)
    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis("equal"); ax.grid(True, alpha=0.4); ax.legend(fontsize=9)
    savefig(fig, fname)


def plot_xy_baseline_vs_heavy(data, colors, goal_pt, goal_label, title, fname):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*goal_pt, marker="*", s=350, c="black", zorder=6,
               label=goal_label)
    for label in ["Baseline (0,0)", "Heavy (10,20)"]:
        if label not in data: continue
        d = data[label]
        ax.plot(d["x"], d["y"], lw=2.2, label=label,
                color=colors[label], alpha=0.9)
        ax.plot(d["x"][0],  d["y"][0],  "o",
                color=colors[label], ms=10, zorder=5)
        ax.plot(d["x"][-1], d["y"][-1], "s",
                color=colors[label], ms=10, zorder=5)
    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis("equal"); ax.grid(True, alpha=0.4); ax.legend(fontsize=11)
    savefig(fig, fname)


def plot_peak_dw_bar(data, colors, title, fname):
    labels      = list(data.keys())
    bar_colors_ = [colors[l] for l in labels]
    vals     = [data[l]["summary"]["max_dw"] for l in labels]
    base_val = vals[0]

    fig, ax = plt.subplots(figsize=(max(8, 1.4*len(labels)), 5))
    bars = ax.bar(labels, vals, color=bar_colors_, width=0.6,
                  edgecolor='white', linewidth=1.5)
    ax.set_ylabel("max |Δω| (rad/s)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.tick_params(axis="x", rotation=18, labelsize=9)
    ax.grid(True, axis="y", alpha=0.35); ax.set_axisbelow(True)
    for bar, val, label in zip(bars, vals, labels):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight='bold')
        if label != labels[0]:
            pct = (base_val - val) / base_val * 100
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"−{pct:.0f}%", ha="center", va="center",
                    fontsize=9.5, fontweight='bold', color='white')
    savefig(fig, fname)


def plot_pareto(data, colors, title, fname, ideal_band=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, d in data.items():
        s = d["summary"]
        ax.scatter(s["tv_w"], s["path_len"],
                   color=colors[label], s=180, zorder=3)
        ax.annotate(label, (s["tv_w"], s["path_len"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    ax.set_xlabel("TV_ω  Σ|Δω|  [lower = smoother]", fontsize=12)
    ax.set_ylabel("Path length (m)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4)
    if ideal_band is not None:
        lo, hi = ideal_band
        ax.axhspan(lo, hi, alpha=0.07, color='green')
    savefig(fig, fname)


def plot_velocity_baseline_vs_heavy(data, colors, signal, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label in ["Baseline (0,0)", "Heavy (10,20)"]:
        if label not in data: continue
        d = data[label]
        ax.plot(d["t"], d[signal], lw=2.0, label=label, color=colors[label])
    if signal == "w":
        ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4); ax.legend(fontsize=10)
    savefig(fig, fname)


def plot_diminishing_returns(data, colors, lambdas, title, fname):
    labels   = list(data.keys())
    max_vals = [data[l]["summary"]["max_dw"] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, max_vals, 'k--', lw=1.0, alpha=0.4, zorder=1)
    for lam, mv, label in zip(lambdas, max_vals, labels):
        ax.scatter(lam, mv, color=colors[label], s=160, zorder=3, label=label)
        ax.annotate(f"{mv:.4f}", (lam, mv),
                    textcoords="offset points", xytext=(0, 9),
                    ha='center', fontsize=8.5,
                    color=colors[label], fontweight='bold')
    ax.axvspan(55, max(lambdas)*1.1, alpha=0.06, color='gray')
    ax.text(57, max(max_vals)*0.98, "diminishing\nreturns zone",
            fontsize=8.5, color='gray', va='top')
    ax.set_xlabel("du_w cost  (λ)", fontsize=12)
    ax.set_ylabel("max |Δω| (rad/s)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.35); ax.legend(fontsize=9, loc='upper right')
    savefig(fig, fname)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n── Experiment 1: Point-to-Point ──")
    exp1 = {}
    for label, fname in EXP1_FILES.items():
        path = os.path.join(CSV_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: missing — {path}")
            continue
        exp1[label] = load(fname, GOAL_EXP1)
        print(f"  loaded: {label}")

    if exp1:
        plot_xy_paths(exp1, EXP1_COLORS, GOAL_EXP1, "goal (-2,-4)",
                      "Exp 1 — XY Path: All 5 Runs  (□=end)",
                      "exp1_xy_path_all5.png", show_walls=False)
        plot_xy_baseline_vs_heavy(exp1, EXP1_COLORS, GOAL_EXP1, "goal (-2,-4)",
                      "Exp 1 — XY Path: Baseline vs Heavy (10,20)\n(○=start  □=end)",
                      "exp1_xy_path_baseline_vs_heavy.png")
        plot_peak_dw_bar(exp1, EXP1_COLORS,
                      "Exp 1 — Peak Angular Jerk (max |Δω|) vs Smoothing Weight λ",
                      "exp1_peak_dw_all5.png")
        plot_pareto(exp1, EXP1_COLORS,
                      "Exp 1 — Smoothness vs Path Length (Pareto Frontier)",
                      "exp1_smoothness_vs_pathlength_all5.png")
        plot_velocity_baseline_vs_heavy(exp1, EXP1_COLORS, "w", "ω (rad/s)",
                      "Exp 1 — Angular Velocity: Baseline vs Heavy (10,20)",
                      "exp1_angular_velocity_baseline_vs_heavy.png")
        plot_velocity_baseline_vs_heavy(exp1, EXP1_COLORS, "v", "v (m/s)",
                      "Exp 1 — Linear Velocity: Baseline vs Heavy (10,20)",
                      "exp1_linear_velocity_baseline_vs_heavy.png")

    print("\n── Experiment 2: Figure-8 ──")
    fig8 = {}
    for label, fname in FIG8_FILES.items():
        path = os.path.join(CSV_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: missing — {path}")
            continue
        fig8[label] = load(fname, np.array([0.0, 0.0]))
        print(f"  loaded: {label}")

    if fig8:
        plot_xy_paths(fig8, FIG8_COLORS, np.array([0.0, 0.0]),
                      "crossing (0,0)",
                      "Exp 2 — XY Path: All 7 Runs  (□=end)",
                      "fig8_xy_path_all7.png", show_walls=True)
        plot_xy_baseline_vs_heavy(fig8, FIG8_COLORS, np.array([0.0, 0.0]),
                      "crossing (0,0)",
                      "Exp 2 — XY Path: Baseline vs Heavy (10,20)\n(○=start  □=end)",
                      "fig8_xy_path_baseline_vs_heavy.png")
        plot_peak_dw_bar(fig8, FIG8_COLORS,
                      "Exp 2 — Peak Angular Jerk (max |Δω|) vs Smoothing Weight λ",
                      "fig8_peak_dw_all7.png")
        plot_pareto(fig8, FIG8_COLORS,
                      "Exp 2 — Smoothness vs Path Length (Pareto — all 7 runs)",
                      "fig8_smoothness_vs_pathlength_all7.png",
                      ideal_band=(29.0, 29.5))
        plot_velocity_baseline_vs_heavy(fig8, FIG8_COLORS, "w", "ω (rad/s)",
                      "Exp 2 — Angular Velocity: Baseline vs Heavy (10,20)",
                      "fig8_angular_velocity_baseline_vs_heavy.png")
        plot_velocity_baseline_vs_heavy(fig8, FIG8_COLORS, "v", "v (m/s)",
                      "Exp 2 — Linear Velocity: Baseline vs Heavy (10,20)",
                      "fig8_linear_velocity_baseline_vs_heavy.png")

        if len(fig8) == len(FIG8_FILES):
            plot_diminishing_returns(fig8, FIG8_COLORS, FIG8_LAMBDAS,
                      "Figure-8 — Diminishing Returns: Peak Δω vs λ",
                      "fig8_diminishing_returns_peak_dw.png")
        else:
            print("  skipped diminishing returns plot — need all 7 runs")

    print(f"\nAll plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
