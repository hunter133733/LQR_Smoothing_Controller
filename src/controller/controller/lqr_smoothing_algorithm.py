from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from controller.controller_base import ControllerBackend
from controller.dubins3d_2ctrls import DubinsCar3D2Ctrls
from controller.reference_trajectory import generate_reference_trajectory
from nav_helpers.trajectory import StateActionTrajectory

DEFAULT_COST_COEFS = {
    "x": 5.0,
    "y": 5.0,
    "theta": 1.0,
    "v": 0.3,
    "w": 0.3,
}


class LQRSmoothingAlgorithm:
    """
    Finite-horizon time-varying LQR for Dubins trajectory tracking with
    augmented-state smoothing.

    Smoothing approach (consistent with LQR Ext3 slide):
    -------------------------------------------------------
    Instead of a post-hoc filter, we augment the state with the previous
    control input so that the LQR cost directly penalises Δu:

        Augmented state:  x'_t = [ x_t (3,) | u_{t-1} (2,) ]   → dim 5
        Augmented input:  u'_t = Δu_t                           → dim 2

        Augmented dynamics:
            A' = [ A   B ]      B' = [ B ]
                 [ 0   I ]           [ I ]

        Augmented cost matrices:
            Q' = diag(Q, R)      (penalise state error AND u_{t-1} magnitude)
            R' = diag(r'_v, r'_w)  (penalise Δu — the smoothing knob)

    Setting R' = 0 recovers standard LQR exactly.

    The terminal cost L and stage cost Q still operate on the original 3-dim
    state; the augmented terminal cost L' = diag(L, R).
    """

    def __init__(
        self,
        cost_coefs: Dict | None = None,
        dt: float = 0.1,
        n: int = 25,
        u_min: NDArray | None = None,
        u_max: NDArray | None = None,
        # --- smoothing knob (R' diagonal) ---
        r_prime_v: float = 1.0,   # penalty on Δv  (0 → standard LQR)
        r_prime_w: float = 1.0,   # penalty on Δω  (0 → standard LQR)
    ):
        self.n = int(max(1, n))
        self.dt = float(dt)
        self.cost_coefs = dict(DEFAULT_COST_COEFS if cost_coefs is None else cost_coefs)

        # --- original-state cost matrices (3×3) ---
        self.L = np.diag(
            [self.cost_coefs["x"], self.cost_coefs["y"], self.cost_coefs["theta"]]
        ).astype(float)
        self.Q = np.diag(
            [self.cost_coefs["x"], self.cost_coefs["y"], self.cost_coefs["theta"]]
        ).astype(float)
        self.R = np.diag([self.cost_coefs["v"], self.cost_coefs["w"]]).astype(float)

        # --- smoothing penalty R' (2×2) on Δu ---
        self.R_prime = np.diag(
            [float(r_prime_v), float(r_prime_w)]
        ).astype(float)

        # --- augmented cost matrices (5×5 and 2×2) ---
        # Q'  = diag(Q, R)  penalises [state error, u_{t-1}]
        # L'  = diag(L, R)  terminal cost on augmented state
        # R'  already defined above — penalises Δu
        self.Q_aug = np.block([
            [self.Q, np.zeros((3, 2))],
            [np.zeros((2, 3)), self.R],
        ])  # 5×5
        self.L_aug = np.block([
            [self.L, np.zeros((3, 2))],
            [np.zeros((2, 3)), self.R],
        ])  # 5×5

        # --- action bounds ---
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
        self.action_min = u_min_arr
        self.action_max = u_max_arr

        # Δu bounds derived from action bounds and a reasonable per-step rate
        # (kept generous so the Riccati solution is the primary limit)
        self.du_min = u_min_arr  # Δu can be at most as large as u range
        self.du_max = u_max_arr

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

        # Solution trajectory
        self.z_sol: NDArray | None = None
        self.u_sol: NDArray | None = None
        self.tau_sol: NDArray | None = None

        # Memory of last applied control (needed for augmented state)
        self.prev_u = np.zeros(2, dtype=float)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __str__(self):
        ret = "\n## LQRSmoothingAlgorithm (augmented-state)\n"
        ret += f"- dt: {self.dt}\n"
        ret += f"- n: {self.n}\n"
        ret += f"- dynamics: {self.dynsys.__class__.__name__}\n"
        ret += f"- costs: {self.cost_coefs}\n"
        ret += f"- R_prime (Δu penalty): {np.diag(self.R_prime).tolist()}\n"
        return ret

    # ------------------------------------------------------------------
    # Augmented-state helpers
    # ------------------------------------------------------------------
    def _build_augmented_matrices(
        self, A: NDArray, B: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Build augmented A' (5×5) and B' (5×2) from original A (3×3), B (3×2).

        A' = [ A   B ]    B' = [ B ]
             [ 0   I ]         [ I ]
        """
        nz, nu = B.shape  # 3, 2
        A_aug = np.block([
            [A,                  B              ],
            [np.zeros((nu, nz)), np.eye(nu)     ],
        ])  # 5×5
        B_aug = np.block([
            [B         ],
            [np.eye(nu)],
        ])  # 5×2
        return A_aug, B_aug

    def _augmented_state(self, z: NDArray, u_prev: NDArray) -> NDArray:
        """Concatenate [z (3,), u_prev (2,)] → x' (5,)."""
        return np.concatenate([z.reshape(3), u_prev.reshape(2)])

    def _augmented_ref(
        self, z_track: NDArray, u_track: NDArray
    ) -> NDArray:
        """
        Build augmented reference states x'_ref of shape (T, 5).
        x'_ref[t] = [z_track[t], u_track[t-1]]
        At t=0 we use zeros for u_{-1} (robot starts from rest on reference).
        """
        T = z_track.shape[0]
        x_ref_aug = np.zeros((T, 5), dtype=float)
        x_ref_aug[:, :3] = z_track
        # u_{t-1}: shift u_track one step, pad first row with zeros
        x_ref_aug[1:, 3:] = u_track[:-1, :]
        return x_ref_aug

    # ------------------------------------------------------------------
    # Riccati backward pass — operates on augmented system
    # ------------------------------------------------------------------
    def compute_gains(
        self, As: NDArray, Bs: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute finite-horizon time-varying LQR gains on the AUGMENTED system.

        Inputs:
            As:  shape (T, 3, 3) — original A matrices along trajectory
            Bs:  shape (T, 3, 2) — original B matrices along trajectory

        Outputs:
            Ks:  shape (T, 2, 5) — augmented gains; Δu* = -K x'_error
            Ps:  shape (T+1, 5, 5) — augmented cost-to-go matrices
        """
        As = np.asarray(As, dtype=float)
        Bs = np.asarray(Bs, dtype=float)
        if As.shape[0] != Bs.shape[0]:
            raise ValueError("As and Bs must have same horizon length")

        n = As.shape[0]
        Ks = np.zeros((n, 2, 5), dtype=float)
        Ps = np.zeros((n + 1, 5, 5), dtype=float)

        # Terminal cost on augmented state
        Ps[n] = self.L_aug  # 5×5

        for i in range(n - 1, -1, -1):
            A_aug, B_aug = self._build_augmented_matrices(As[i], Bs[i])
            Pnext = Ps[i + 1]

            # Discrete-time Riccati
            #   S  = R' + B'ᵀ P_{i+1} B'
            #   K  = S⁻¹ B'ᵀ P_{i+1} A'
            #   P  = Q' + A'ᵀ P_{i+1} A' − A'ᵀ P_{i+1} B' K
            S = self.R_prime + B_aug.T @ Pnext @ B_aug          # 2×2
            Ks[i] = np.linalg.solve(S, B_aug.T @ Pnext @ A_aug) # 2×5
            Ps[i] = (
                self.Q_aug
                + A_aug.T @ Pnext @ A_aug
                - A_aug.T @ Pnext @ B_aug @ Ks[i]
            )  # 5×5

        return Ks, Ps

    # ------------------------------------------------------------------
    # Linearise along trajectory (unchanged — original state space)
    # ------------------------------------------------------------------
    def linearize_along_traj(
        self, z_traj: NDArray, u_traj: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Linearize system along a trajectory segment."""
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

    # ------------------------------------------------------------------
    # Solve: forward rollout using augmented gains
    # ------------------------------------------------------------------
    def solve(
        self,
        z_0: NDArray,
        t_0: float,
        z_ref: NDArray,
        u_ref: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Returns sequence of controls and states from the augmented-state LQR.

        The planning rollout uses a *local copy* of prev_u so that the
        internal simulation does not corrupt the controller's memory.
        prev_u is updated only in get_action() after the real command is sent.
        """
        z_ref = np.asarray(z_ref, dtype=float)
        u_ref = np.asarray(u_ref, dtype=float)
        z_0 = np.asarray(z_0, dtype=float).reshape(3)
        t_0 = float(t_0)

        if z_ref.ndim != 2 or z_ref.shape[1] != 3:
            raise ValueError("z_ref must have shape (T, 3)")
        if u_ref.ndim != 2 or u_ref.shape[1] != 2:
            raise ValueError("u_ref must have shape (T, 2)")
        if z_ref.shape[0] < 2 or u_ref.shape[0] < 1:
            raise ValueError("reference trajectories are too short")

        n_track = int(min(self.n, z_ref.shape[0], u_ref.shape[0]))
        z_track = z_ref[:n_track, :]
        u_track = u_ref[:n_track, :]

        # Linearise and compute augmented gains
        As, Bs = self.linearize_along_traj(z_track, u_track)
        Ks, _ = self.compute_gains(As, Bs)  # Ks: (T, 2, 5)

        # Build augmented reference states
        x_ref_aug = self._augmented_ref(z_track, u_track)  # (T, 5)

        # Forward rollout — use a local u_prev copy so self.prev_u is untouched
        self.dynsys.reset(z_0)
        u_prev_local = self.prev_u.copy()

        for i in range(n_track):
            z_t = self.dynsys.z_hist[-1, :]  # current state (3,)

            # Augmented state error: x'_error = x'_t - x'_ref_t
            x_aug_t = self._augmented_state(z_t, u_prev_local)
            x_aug_ref = x_ref_aug[i]
            delta_x_aug = x_aug_t - x_aug_ref
            # Wrap heading component of error
            delta_x_aug[2] = np.arctan2(
                np.sin(delta_x_aug[2]), np.cos(delta_x_aug[2])
            )

            # Optimal Δu from augmented gain
            delta_u = -Ks[i] @ delta_x_aug  # (2,)

            # Recover absolute control: u_t = u_ref_t + Δu* + Δu_correction
            # The reference Δu is u_ref[t] - u_ref[t-1]
            u_ref_prev = u_track[i - 1] if i > 0 else np.zeros(2)
            delta_u_ref = u_track[i] - u_ref_prev
            u_t = u_prev_local + delta_u_ref + delta_u

            u_t = np.clip(u_t, self.action_min, self.action_max)
            u_prev_local = u_t.copy()

            self.dynsys.forward_np(dt=self.dt, ctrl=u_t)

        self.z_sol = self.dynsys.z_hist[1:, :]
        self.u_sol = self.dynsys.u_hist
        self.tau_sol = t_0 + np.linspace(self.dt, self.dt * n_track, n_track)

        return self.z_sol, self.u_sol, self.tau_sol

    def update_prev_u(self, u_applied: NDArray) -> None:
        """Call this after the real command is sent to update controller memory."""
        self.prev_u = np.asarray(u_applied, dtype=float).reshape(2).copy()


# ---------------------------------------------------------------------------
# Controller wrapper (ROS2 frontend)
# ---------------------------------------------------------------------------

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
            [float(cfg.get("v_min", -0.2)), float(cfg.get("w_min", -1.2))],
            dtype=float,
        )
        u_max = np.array(
            [float(cfg.get("v_max", 1.0)), float(cfg.get("w_max", 1.2))],
            dtype=float,
        )

        # r_prime_{v,w} are the smoothing knobs — set to 0 for standard LQR
        self._algo = LQRSmoothingAlgorithm(
            cost_coefs=cost_coefs,
            dt=dt,
            n=horizon,
            u_min=u_min,
            u_max=u_max,
            r_prime_v=float(smooth_cfg.get("r_prime_v", 1.0)),
            r_prime_w=float(smooth_cfg.get("r_prime_w", 1.0)),
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def get_action(
        self, observation: NDArray, traj: StateActionTrajectory = None
    ) -> NDArray:
        obs = np.asarray(observation, dtype=float).reshape(3)
        action = np.zeros((2,), dtype=float)

        if traj is not None:
            self._z_ref = traj.states
            self._u_ref = traj.actions

        if self._z_ref is None or self._u_ref is None:  # fallback for debugging
            self._tau_ref, self._z_ref, self._u_ref = generate_reference_trajectory(
                kind=self._ref_kind,
                dt=self._algo.dt,
                n_steps=self._ref_n_steps,
                start_state=obs,
                goal_state=self._goal,
            )

        ref_idx = self.closest_reference_index(
            self._z_ref,
            obs,
            self._step,
        )

        zRefWin, uRefWin = self.sample_reference_window(
            self._z_ref,
            self._u_ref,
            ref_idx,
            self._algo.n,
        )

        z_sol, u_sol, tau_sol = self._algo.solve(
            z_0=obs,
            t_0=ref_idx * self._algo.dt,
            z_ref=zRefWin,
            u_ref=uRefWin,
        )

        # The first planned command is the action to execute.
        # Smoothing is baked into the Riccati solution — no post-hoc filter needed.
        action = np.clip(u_sol[0, :], self._u_min, self._u_max)

        # Update controller memory with the command actually sent to the robot
        self._algo.update_prev_u(action)

        # Stop at goal
        goal_err = np.linalg.norm(obs[:2] - self._goal[:2])
        yaw_err = np.arctan2(
            np.sin(obs[2] - self._goal[2]),
            np.cos(obs[2] - self._goal[2]),
        )
        if goal_err < 0.10 and abs(yaw_err) < 0.10:
            action = np.zeros(2, dtype=float)

        self._step = ref_idx + 1

        return np.clip(action, self._u_min, self._u_max), z_sol, u_sol

    # ------------------------------------------------------------------
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