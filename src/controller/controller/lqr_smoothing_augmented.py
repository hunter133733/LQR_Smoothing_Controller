from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from controller.controller_base import ControllerBackend
from controller.dubins3d_2ctrls import DubinsCar3D2Ctrls
from controller.reference_trajectory import generate_reference_trajectory
from nav_helpers.trajectory import StateActionTrajectory


# Default cost weights
# Tuning knobs. YAML files overwrite these and change them for testing cases
DEFAULT_COST_COEFS = {
    "x": 5.0,
    "y": 5.0,
    "theta": 1.0,
    "v": 0.3, # Abs linear velocity penalty
    "w": 0.3, # Abs angular velocity penalty
}

# Augmented state: x_aug = [x, y, theta, v_prev, w_prev]
# New control variable: du = u_k - u_{k-1}
# This Lets us penalize command-rate changes inside the optimization. 
class LQRSmoothingAlgorithm:
    
    
    def __init__(
        self,
        cost_coefs: Dict | None = None, # Per state cost weights (x, y, theta, v, w)
        dt: float = 0.1, # Timestep (seconds) used for linearisation
        n: int = 25, # Horizon length H. Needed for LQR Riccati recursion
        u_min: NDArray | None = None, # Limits on v, w. 
        u_max: NDArray | None = None, # Limits on v, w
        du_costs: Dict | None = None, # Penalty weights on control rate.
    ):
        self.n = int(max(1, n))
        self.dt = float(dt)
        self.cost_coefs = dict(DEFAULT_COST_COEFS if cost_coefs is None else cost_coefs)

        # Build LQR cost matrices
        self.Qx = np.diag(
            [self.cost_coefs["x"], self.cost_coefs["y"], self.cost_coefs["theta"]]
        ).astype(float)

        # Cost-to-go at the end of the horizon
        self.Lx = self.Qx.copy()

        # Penalizes the absolute velocity in the augmented-state
        self.Ru = np.diag([self.cost_coefs["v"], self.cost_coefs["w"]]).astype(float)

        # Penalizes the change in control
        if du_costs is None:
            du_costs = {"dv": 1.0, "dw": 1.0}
        self.Rdu = np.diag([float(du_costs["dv"]), float(du_costs["dw"])]).astype(float)

        # Control limits:
        u_min_arr = (
            np.array([-0.2, -1.2], dtype=float)
            if u_min is None
            else np.asarray(u_min, dtype=float).reshape(2)
        )
        u_max_arr = (
            np.array([1.0, 1.2], dtype=float)
            if u_max is None
            else np.asarray(u_max, dtype=float).reshape(2)
        )


        #Dynamics model:
        self.dynsys = DubinsCar3D2Ctrls(
            z_0=np.zeros(3, dtype=float),
            dt=self.dt,
            u_min=u_min_arr,
            u_max=u_max_arr,
            d_min=np.zeros(3, dtype=float),
            d_max=np.zeros(3, dtype=float),
            u_mode="min",
            d_mode="max",
        )

        self.action_min = u_min_arr
        self.action_max = u_max_arr


        # Augmented cost matrices Q and L (slide 19 in berkley lecture for visualization)
        self.Q_aug = np.block(
            [
                [self.Qx, np.zeros((3, 2), dtype=float)],
                [np.zeros((2, 3), dtype=float), self.Ru],
            ]
        )
        self.L_aug = np.block(
            [
                [self.Lx, np.zeros((3, 2), dtype=float)],
                [np.zeros((2, 3), dtype=float), self.Ru],
            ]
        )

        self.z_sol: NDArray | None = None
        self.u_sol: NDArray | None = None
        self.tau_sol: NDArray | None = None

        # Memory of the last applied control
        self.prev_u = np.zeros(2, dtype=float)

    def __str__(self):
        ret = "\n## LQRSmoothingAlgorithm\n"
        ret += f"- dt: {self.dt}\n"
        ret += f"- n: {self.n}\n"
        ret += f"- dynamics: {self.dynsys.__class__.__name__}\n"
        ret += f"- costs: {self.cost_coefs}\n"
        ret += f"- du_costs: diag({np.diag(self.Rdu)})\n"
        return ret

    # Solve interface (public)
    def solve(
        self,
        z_0: NDArray,
        t_0: float,
        z_ref: NDArray,
        u_ref: NDArray,
        u_prev_0: NDArray,
        u_prev_ref_0: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        

        # Input validation and type coercion
        z_ref = np.asarray(z_ref, dtype=float)
        u_ref = np.asarray(u_ref, dtype=float)
        z_0 = np.asarray(z_0, dtype=float).reshape(3)
        u_prev_0 = np.asarray(u_prev_0, dtype=float).reshape(2)
        u_prev_ref_0 = np.asarray(u_prev_ref_0, dtype=float).reshape(2)
        t_0 = float(t_0)


        # if statements used for testing
        if z_ref.ndim != 2 or z_ref.shape[1] != 3:
            raise ValueError("z_ref wrong shape")
        if u_ref.ndim != 2 or u_ref.shape[1] != 2:
            raise ValueError("u_ref wrong shape")
        if z_ref.shape[0] < 2 or u_ref.shape[0] < 1:
            raise ValueError("reference trajectories are too short")

        # Clamping horizon to available reference length, ensuring model doesn't
        # try to predict further into future than it can justify
        n_track = int(min(self.n, z_ref.shape[0], u_ref.shape[0]))
        z_track = z_ref[:n_track, :]
        u_track = u_ref[:n_track, :]

        # Step 1: Linearize along reference
        As, Bs = self.linearize_along_traj(z_track, u_track)

        # Step 2: Build augmented dynamics
        A_aug, B_aug = self.build_augmented_dynamics(As, Bs)

        # Step 3: Run Riccati recursion:
        # Compute gains runs backward pass
        Ks, _ = self.compute_gains(A_aug, B_aug)


        # Precompute reference du. Allows controller to correct deviations around ref rate
        u_prev_ref = np.zeros((n_track, 2), dtype=float)
        du_ref = np.zeros((n_track, 2), dtype=float)

        u_prev_ref[0, :] = u_prev_ref_0
        du_ref[0, :] = u_track[0, :] - u_prev_ref[0, :]

        for i in range(1, n_track):
            u_prev_ref[i, :] = u_track[i - 1, :]
            du_ref[i, :] = u_track[i, :] - u_prev_ref[i, :]


        # Step 4: Forward sim
        self.dynsys.reset(z_0)
        z_pred = np.zeros((n_track, 3), dtype=float)
        u_pred = np.zeros((n_track, 2), dtype=float)

        z_now = z_0.copy()
        u_prev_now = u_prev_0.copy()

        for i in range(n_track):

            # Augmented state and ref
            x_aug_now = np.concatenate([z_now, u_prev_now])
            x_aug_ref = np.concatenate([z_track[i, :], u_prev_ref[i, :]])

            # Error in augmented state
            delta_aug = x_aug_now - x_aug_ref

            # Wrap heading error (to help avoid discontinuities)
            delta_aug[2] = np.arctan2(np.sin(delta_aug[2]), np.cos(delta_aug[2]))

            # Closed-loop law
            du_t = du_ref[i, :] - Ks[i] @ delta_aug
            u_t = u_prev_now + du_t
            u_t = np.clip(u_t, self.action_min, self.action_max)

            # Sim a step of non linear dynamics
            self.dynsys.forward_np(dt=self.dt, ctrl=u_t)
            z_now = self.dynsys.z_hist[-1, :]
            u_prev_now = u_t.copy()

            z_pred[i, :] = z_now
            u_pred[i, :] = u_t

        # Cache and return
        self.z_sol = z_pred
        self.u_sol = u_pred
        self.tau_sol = t_0 + np.linspace(self.dt, self.dt * n_track, n_track)
        return self.z_sol, self.u_sol, self.tau_sol


    # Backward Riccati recursion
    def compute_gains(self, As: NDArray, Bs: NDArray) -> Tuple[NDArray, NDArray]:

        As = np.asarray(As, dtype=float)
        Bs = np.asarray(Bs, dtype=float)
        if As.shape[0] != Bs.shape[0]:
            raise ValueError("As and Bs must have same horizon length")

        n = As.shape[0]

        # Gains k_t and cost-to-go P_t preallocated
        Ks = np.zeros((n, 2, 5), dtype=float)
        Ps = np.zeros((n + 1, 5, 5), dtype=float)

        # Terminal condition (cost-to-go = L, at step n)
        Ps[n] = self.L_aug

        # Backward pass
        for i in range(n - 1, -1, -1):
            At = As[i]
            Bt = Bs[i]
            Pnext = Ps[i + 1]

            S = self.Rdu + Bt.T @ Pnext @ Bt
            Ks[i] = np.linalg.inv(S) @ (Bt.T @ Pnext @ At)
            Ps[i] = self.Q_aug + (At.T @ Pnext @ At) - (At.T @ Pnext @ Bt @ Ks[i])
        return Ks, Ps



    # Gets (At, Bt) from nonlinear model.
    # Linearizes the original Dubins dynamics at each point along the reference trajectory
    def linearize_along_traj(
        self, z_traj: NDArray, u_traj: NDArray
    ) -> Tuple[NDArray, NDArray]:

        z_traj = np.asarray(z_traj, dtype=float)
        u_traj = np.asarray(u_traj, dtype=float)

        if z_traj.shape[0] != u_traj.shape[0]:
            raise ValueError("z_traj and u_traj must have same number of rows")

        n = int(min(self.n, z_traj.shape[0]))
        As = np.zeros((n, 3, 3), dtype=float)
        Bs = np.zeros((n, 3, 2), dtype=float)

        for t in range(n):
            As[t], Bs[t] = self.dynsys.linearize(
                z_t=z_traj[t, :], u_t=u_traj[t, :], discrete=True, dt=self.dt
            )
        return As, Bs


    #Builds the augmented dyanmics with du as the new control.
    @staticmethod
    def build_augmented_dynamics(
        As: NDArray,
        Bs: NDArray,
    ) -> Tuple[NDArray, NDArray]:
    
        n = As.shape[0]
        A_aug = np.zeros((n, 5, 5), dtype=float)
        B_aug = np.zeros((n, 5, 2), dtype=float)

        for i in range(n):
            # State transition (A_t)
            A_aug[i, :3, :3] = As[i]

            # How u_prev affects next z (B_t)
            A_aug[i, :3, 3:] = Bs[i]

            # identity
            A_aug[i, 3:, 3:] = np.eye(2, dtype=float)

            # Enter z dynamics (B_t)
            B_aug[i, :3, :] = Bs[i]

            # identity
            B_aug[i, 3:, :] = np.eye(2, dtype=float)

        return A_aug, B_aug


class LQRSmoothingController(ControllerBackend):
    """Thin wrapper used by ROS2 frontend."""

    def __init__(self, config):
        cfg = dict(config.get("lqr", {}))
        smooth_cfg = dict(config.get("lqr_smooth", {}))
        dt = float(config.get("dt", 0.1))
        horizon = int(cfg.get("horizon", 25))
        cost_coefs = {
            "x": float(cfg.get("x_cost", DEFAULT_COST_COEFS["x"])),
            "y": float(cfg.get("y_cost", DEFAULT_COST_COEFS["y"])),
            "theta": float(cfg.get("theta_cost", DEFAULT_COST_COEFS["theta"])),
            "v": float(cfg.get("v_cost", DEFAULT_COST_COEFS["v"])),
            "w": float(cfg.get("w_cost", DEFAULT_COST_COEFS["w"])),
        }
        u_min = np.array(
            [float(cfg.get("v_min", -0.2)), float(cfg.get("w_min", -1.2))], dtype=float
        )
        u_max = np.array(
            [float(cfg.get("v_max", 1.0)), float(cfg.get("w_max", 1.2))], dtype=float
        )

        self._algo = LQRSmoothingAlgorithm(
            cost_coefs=cost_coefs,
            dt=dt,
            n=horizon,
            u_min=u_min,
            u_max=u_max,
            du_costs={
                "dv": float(smooth_cfg.get("du_v_cost", 1.0)),
                "dw": float(smooth_cfg.get("du_w_cost", 1.0)),
            },
        )
        ref_cfg = dict(config.get("reference", {}))
        self._ref_kind = str(ref_cfg.get("kind", "to_goal"))
        self._ref_n_steps = int(ref_cfg.get("n_steps", 500))
        self._goal = np.asarray(
            config.get("goal", np.array([3.5, 2.5, 0.0])), dtype=float
        ).reshape(3)
        self._tau_ref: NDArray | None = None
        self._z_ref: NDArray | None = None
        self._u_ref: NDArray | None = None
        self._step = 0
        self._u_min = u_min
        self._u_max = u_max

    @staticmethod
    def closest_reference_index(
        z_ref: NDArray,
        obs: NDArray,
        prev_idx: int,
        search_back: int = 5,
        search_ahead: int = 60,
    ) -> int:
        start = max(0, prev_idx - search_back)
        end = min(z_ref.shape[0], prev_idx + search_ahead)

        pts = z_ref[start:end, :2]
        diff = pts - obs[:2]
        dist2 = np.sum(diff * diff, axis=1)

        return start + int(np.argmin(dist2))

    def get_action(self, observation, traj=None):
        obs = np.asarray(observation, dtype=float).reshape(3)

        if traj is not None:
            self._z_ref = traj.states
            self._u_ref = traj.actions

        if self._z_ref is None or self._u_ref is None:
            self._tau_ref, self._z_ref, self._u_ref = generate_reference_trajectory(
                kind=self._ref_kind,
                dt=self._algo.dt,
                n_steps=self._ref_n_steps,
                start_state=obs,
                goal_state=self._goal,
            )

        # ── Bug 1 fix: check goal FIRST and hard-stop before doing anything else
        goal_err = np.linalg.norm(obs[:2] - self._goal[:2])
        yaw_err  = np.arctan2(
            np.sin(obs[2] - self._goal[2]),
            np.cos(obs[2] - self._goal[2]),
        )
        if goal_err < 0.20 and abs(yaw_err) < 0.20:
            self._algo.prev_u = np.zeros(2, dtype=float)
            zero = np.zeros(2, dtype=float)
            # Return dummy z_sol/u_sol so callers don't break
            dummy = np.zeros((self._algo.n, 3), dtype=float)
            dummy_u = np.zeros((self._algo.n, 2), dtype=float)
            return zero, dummy, dummy_u

        ref_idx = self.closest_reference_index(
            self._z_ref, obs, self._step
        )

        # ── Bug 1 fix: clamp ref_idx so it never runs off the end of the trajectory
        max_idx = self._z_ref.shape[0] - 2   # leave at least a 2-step window
        ref_idx = min(ref_idx, max_idx)

        z_ref_win, u_ref_win = self.sample_reference_window(
            self._z_ref, self._u_ref, ref_idx, self._algo.n
        )

        # ── Bug 2 fix: zero out the padded tail's reference actions
        n_remaining = self._u_ref.shape[0] - ref_idx
        if n_remaining < self._algo.n:
            u_ref_win[n_remaining:] = 0.0   # don't chase a stale non-zero u_ref

        # ── Bug 3 fix: u_prev_ref_0 is zeros before the trajectory starts
        if ref_idx > 0:
            u_prev_ref_0 = np.asarray(
                self._u_ref[ref_idx - 1], dtype=float
            ).reshape(2)
        else:
            u_prev_ref_0 = np.zeros(2, dtype=float)   # robot was stationary

        z_sol, u_sol, _ = self._algo.solve(
            z_0=obs,
            t_0=ref_idx * self._algo.dt,
            z_ref=z_ref_win,
            u_ref=u_ref_win,
            u_prev_0=self._algo.prev_u,
            u_prev_ref_0=u_prev_ref_0,
        )

        action = u_sol[0, :]
        self._algo.prev_u = action.copy()
        self._step = ref_idx + 1

        return np.clip(action, self._u_min, self._u_max), z_sol, u_sol

    @staticmethod
    def sample_reference_window(
        z_ref: NDArray,
        u_ref: NDArray,
        start_idx: int,
        horizon: int,
    ) -> Tuple[NDArray, NDArray]:
        """Slice a fixed-length horizon and pad with the final sample if needed."""
        start = int(max(0, start_idx))
        horizon = int(max(1, horizon))

        z_out = np.zeros((horizon, 3), dtype=float)
        u_out = np.zeros((horizon, 2), dtype=float)

        z_last = np.asarray(z_ref[-1], dtype=float).reshape(3)
        u_last = np.asarray(u_ref[-1], dtype=float).reshape(2)
        n_z = z_ref.shape[0]
        n_u = u_ref.shape[0]

        for i in range(horizon):
            idx = start + i
            z_out[i] = (
                np.asarray(z_ref[idx], dtype=float).reshape(3) if idx < n_z else z_last
            )
            u_out[i] = (
                np.asarray(u_ref[idx], dtype=float).reshape(2) if idx < n_u else u_last
            )

        return z_out, u_out
