import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

from controller.lqr_algorithm import LQRController
from controller.smooth_lqr_algorithm import SmoothedLQRController
from nav_helpers.trajectory import StateActionTrajectory


@dataclass
class RunResult:
    name: str
    states: np.ndarray
    actions: np.ndarray
    pos_mse: float
    theta_mse: float
    max_pos_err: float
    control_energy: float
    delta_u_energy: float
    delta_v_energy: float
    delta_w_energy: float


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def make_straight_reference(dt=0.1, n_steps=120, v=0.5):
    states = np.zeros((n_steps + 1, 3), dtype=float)
    actions = np.zeros((n_steps, 2), dtype=float)
    actions[:, 0] = v
    actions[:, 1] = 0.0

    for k in range(n_steps):
        x, y, th = states[k]
        states[k + 1] = [
            x + v * math.cos(th) * dt,
            y + v * math.sin(th) * dt,
            th,
        ]

    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def make_arc_reference(dt=0.1, n_steps=120, v=0.5, w=0.25):
    states = np.zeros((n_steps + 1, 3), dtype=float)
    actions = np.zeros((n_steps, 2), dtype=float)
    actions[:, 0] = v
    actions[:, 1] = w

    for k in range(n_steps):
        x, y, th = states[k]
        states[k + 1] = [
            x + v * math.cos(th) * dt,
            y + v * math.sin(th) * dt,
            th + w * dt,
        ]

    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def make_s_curve_reference(dt=0.1, n_steps=150, v=0.45):
    states = np.zeros((n_steps + 1, 3), dtype=float)
    actions = np.zeros((n_steps, 2), dtype=float)
    actions[:, 0] = v

    for k in range(n_steps):
        t = k * dt
        w = 0.45 * math.sin(0.9 * t)
        actions[k, 1] = w

        x, y, th = states[k]
        states[k + 1] = [
            x + v * math.cos(th) * dt,
            y + v * math.sin(th) * dt,
            th + w * dt,
        ]

    return StateActionTrajectory(states=states, actions=actions, dt=dt)


def simulate_controller(controller, traj: StateActionTrajectory, x0: np.ndarray):
    x = np.asarray(x0, dtype=float).reshape(3).copy()

    X = [x.copy()]
    U = []

    for _ in range(traj.actions.shape[0]):
        u, _, _ = controller.get_action(x, traj)
        u = np.asarray(u, dtype=float).reshape(2)

        # Dubins forward Euler step
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


def compute_metrics(name: str, X: np.ndarray, U: np.ndarray, traj: StateActionTrajectory):
    ref_X = traj.states[: X.shape[0]]
    ref_U = traj.actions[: U.shape[0]]

    pos_err = X[:, :2] - ref_X[:, :2]
    pos_err_norm = np.linalg.norm(pos_err, axis=1)

    theta_err = np.array(
        [wrap_angle(X[k, 2] - ref_X[k, 2]) for k in range(X.shape[0])],
        dtype=float,
    )

    if U.shape[0] >= 2:
        dU = U[1:] - U[:-1]
    else:
        dU = np.zeros((0, 2), dtype=float)

    return RunResult(
        name=name,
        states=X,
        actions=U,
        pos_mse=float(np.mean(np.sum(pos_err**2, axis=1))),
        theta_mse=float(np.mean(theta_err**2)),
        max_pos_err=float(np.max(pos_err_norm)),
        control_energy=float(np.sum(np.sum(U**2, axis=1))),
        delta_u_energy=float(np.sum(np.sum(dU**2, axis=1))),
        delta_v_energy=float(np.sum(dU[:, 0] ** 2)) if dU.shape[0] else 0.0,
        delta_w_energy=float(np.sum(dU[:, 1] ** 2)) if dU.shape[0] else 0.0,
    )


def plot_run_comparison(traj, results, title, outdir):
    os.makedirs(outdir, exist_ok=True)

    # path plot
    plt.figure(figsize=(8, 5))
    plt.plot(traj.states[:, 0], traj.states[:, 1], "k--", linewidth=2, label="reference")
    for r in results:
        plt.plot(r.states[:, 0], r.states[:, 1], linewidth=2, label=r.name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title}: path comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_path.png"), dpi=180)
    plt.close()

    # v plot
    plt.figure(figsize=(8, 4))
    for r in results:
        plt.plot(r.actions[:, 0], linewidth=2, label=r.name)
    plt.xlabel("time step")
    plt.ylabel("v")
    plt.title(f"{title}: linear speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_v.png"), dpi=180)
    plt.close()

    # omega plot
    plt.figure(figsize=(8, 4))
    for r in results:
        plt.plot(r.actions[:, 1], linewidth=2, label=r.name)
    plt.xlabel("time step")
    plt.ylabel("omega")
    plt.title(f"{title}: angular rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_omega.png"), dpi=180)
    plt.close()

    # delta omega plot
    plt.figure(figsize=(8, 4))
    for r in results:
        if r.actions.shape[0] >= 2:
            dw = np.diff(r.actions[:, 1])
            plt.plot(dw, linewidth=2, label=r.name)
    plt.xlabel("time step")
    plt.ylabel("Δomega")
    plt.title(f"{title}: control smoothness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_delta_omega.png"), dpi=180)
    plt.close()


def print_metrics_table(results):
    print("\n" + "=" * 110)
    print(
        f"{'name':24s} {'pos_mse':>12s} {'theta_mse':>12s} {'max_pos_err':>14s} "
        f"{'u_energy':>12s} {'du_energy':>12s} {'dv_energy':>12s} {'dw_energy':>12s}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r.name:24s} "
            f"{r.pos_mse:12.6f} {r.theta_mse:12.6f} {r.max_pos_err:14.6f} "
            f"{r.control_energy:12.6f} {r.delta_u_energy:12.6f} "
            f"{r.delta_v_energy:12.6f} {r.delta_w_energy:12.6f}"
        )
    print("=" * 110 + "\n")

def save_case_metrics(results, title, outdir):
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, f"{title}_metrics.csv")
    json_path = os.path.join(outdir, f"{title}_metrics.json")

    fieldnames = [
        "name",
        "pos_mse",
        "theta_mse",
        "max_pos_err",
        "control_energy",
        "delta_u_energy",
        "delta_v_energy",
        "delta_w_energy",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "name": r.name,
                    "pos_mse": r.pos_mse,
                    "theta_mse": r.theta_mse,
                    "max_pos_err": r.max_pos_err,
                    "control_energy": r.control_energy,
                    "delta_u_energy": r.delta_u_energy,
                    "delta_v_energy": r.delta_v_energy,
                    "delta_w_energy": r.delta_w_energy,
                }
            )

    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "name": r.name,
                    "pos_mse": r.pos_mse,
                    "theta_mse": r.theta_mse,
                    "max_pos_err": r.max_pos_err,
                    "control_energy": r.control_energy,
                    "delta_u_energy": r.delta_u_energy,
                    "delta_v_energy": r.delta_v_energy,
                    "delta_w_energy": r.delta_w_energy,
                }
                for r in results
            ],
            f,
            indent=2,
        )


def save_summary_metrics(all_results, outdir):
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "all_cases_metrics.csv")
    json_path = os.path.join(outdir, "all_cases_metrics.json")

    fieldnames = [
        "case",
        "name",
        "pos_mse",
        "theta_mse",
        "max_pos_err",
        "control_energy",
        "delta_u_energy",
        "delta_v_energy",
        "delta_w_energy",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, results in all_results.items():
            for r in results:
                writer.writerow(
                    {
                        "case": case_name,
                        "name": r.name,
                        "pos_mse": r.pos_mse,
                        "theta_mse": r.theta_mse,
                        "max_pos_err": r.max_pos_err,
                        "control_energy": r.control_energy,
                        "delta_u_energy": r.delta_u_energy,
                        "delta_v_energy": r.delta_v_energy,
                        "delta_w_energy": r.delta_w_energy,
                    }
                )

    json_data = {}
    for case_name, results in all_results.items():
        json_data[case_name] = [
            {
                "name": r.name,
                "pos_mse": r.pos_mse,
                "theta_mse": r.theta_mse,
                "max_pos_err": r.max_pos_err,
                "control_energy": r.control_energy,
                "delta_u_energy": r.delta_u_energy,
                "delta_v_energy": r.delta_v_energy,
                "delta_w_energy": r.delta_w_energy,
            }
            for r in results
        ]

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def run_angle_wrap_test():
    print("\n[Angle-wrap sanity test]")
    raw = 3.13 - (-3.13)
    wrapped = wrap_angle(raw)
    print(f"raw heading difference     = {raw:.6f}")
    print(f"wrapped heading difference = {wrapped:.6f}")
    print("Expected: wrapped value should be small, not close to 2*pi.\n")


def make_config(dv_cost=0.0, dw_cost=0.0):
    return {
        "dt": 0.1,
        "goal": np.array([5.0, 0.0, 0.0], dtype=float),
        "reference": {"kind": "to_goal", "n_steps": 500},
        "lqr": {
            "horizon": 25,
            "x_cost": 5.0,
            "y_cost": 5.0,
            "theta_cost": 1.0,
            "v_cost": 0.3,
            "w_cost": 0.3,
            "dv_cost": dv_cost,
            "dw_cost": dw_cost,
            "v_min": -0.2,
            "v_max": 1.0,
            "w_min": -1.2,
            "w_max": 1.2,
        },
    }


def run_one_case(title, traj, x0, outdir):
    baseline = LQRController(make_config(dv_cost=0.0, dw_cost=0.0))
    smooth_zero = SmoothedLQRController(make_config(dv_cost=0.0, dw_cost=0.0))
    smooth_on = SmoothedLQRController(make_config(dv_cost=0.05, dw_cost=0.20))

    Xb, Ub = simulate_controller(baseline, traj, x0)
    Xz, Uz = simulate_controller(smooth_zero, traj, x0)
    Xs, Us = simulate_controller(smooth_on, traj, x0)

    results = [
        compute_metrics("baseline_lqr", Xb, Ub, traj),
        compute_metrics("smoothed_zero_penalty", Xz, Uz, traj),
        compute_metrics("smoothed_nonzero_penalty", Xs, Us, traj),
    ]

    print(f"\n=== {title} ===")
    print_metrics_table(results)
    plot_run_comparison(traj, results, title, outdir)
    save_case_metrics(results, title, outdir)

    return results



def main():
    outdir = "lqr_smoothing_test_results"
    os.makedirs(outdir, exist_ok=True)

    run_angle_wrap_test()

    straight = make_straight_reference()
    arc = make_arc_reference()
    s_curve = make_s_curve_reference()

    x0_1 = np.array([0.0, 0.20, 0.10], dtype=float)
    x0_2 = np.array([0.0, -0.25, -0.15], dtype=float)
    x0_3 = np.array([0.0, 0.15, 3.13], dtype=float)

    all_results = {}

    all_results["straight_case"] = run_one_case("straight_case", straight, x0_1, outdir)
    all_results["arc_case"] = run_one_case("arc_case", arc, x0_2, outdir)
    all_results["s_curve_case"] = run_one_case("s_curve_case", s_curve, x0_3, outdir)

    save_summary_metrics(all_results, outdir)

    print(f"Saved plots and metrics to: {outdir}")


if __name__ == "__main__":
    main()