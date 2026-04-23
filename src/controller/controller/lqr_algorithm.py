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


class LQRAlgorithm:
    """Finite-horizon time-varying LQR for Dubins trajectory tracking."""

    def __init__(
        self,
        cost_coefs: Dict | None = None,
        dt: float = 0.1,  # time discretization
        n: int = 25,  # time horizon in number of steps to look ahead
        u_min: NDArray | None = None,
        u_max: NDArray | None = None,
    ):
        self.n = int(max(1, n))
        self.dt = float(dt)
        self.cost_coefs = dict(DEFAULT_COST_COEFS if cost_coefs is None else cost_coefs)

        self.L = np.diag(
            [self.cost_coefs["x"], self.cost_coefs["y"], self.cost_coefs["theta"]]
        ).astype(float)
        self.Q = np.diag(
            [self.cost_coefs["x"], self.cost_coefs["y"], self.cost_coefs["theta"]]
        ).astype(float)
        self.R = np.diag([self.cost_coefs["v"], self.cost_coefs["w"]]).astype(float)

        # speed and turn rate
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

        # Solution trajectory
        self.z_sol: NDArray | None = None
        self.u_sol: NDArray | None = None
        self.tau_sol: NDArray | None = None

    def __str__(self):
        ret = "\n## LQRAlgorithm\n"
        ret += f"- dt: {self.dt}\n"
        ret += f"- n: {self.n}\n"
        ret += f"- dynamics: {self.dynsys.__class__.__name__}\n"
        ret += f"- costs: {self.cost_coefs}\n"
        return ret

    def solve(
        self,
        z_0: NDArray,
        t_0: float,
        z_ref: NDArray,
        u_ref: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Returns sequence of controls and states obtained from time-varying LQR
        for following a state and control trajectory
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

        # Track the first up to n steps of the reference trajectory
        n_track = int(min(self.n, z_ref.shape[0], u_ref.shape[0]))
        z_track = z_ref[:n_track, :]
        u_track = u_ref[:n_track, :]

        # Solve LQR
        As, Bs = self.linearize_along_traj(z_track, u_track)
        Ks, _ = self.compute_gains(As, Bs)

        # Compute controls
        self.dynsys.reset(z_0)
        for i in range(n_track):
            u_t = np.zeros_like(u_track[0, :])

            # TODO: Compute state deviation, and then the saturated tracking control
            # STUDENT CODE START
            
            #G et robots current state
            zT = self.dynsys.z_hist[-1, :]

            # State error
            deltaZ = zT - z_track[i, :]

            # Wrap heading error
            deltaZ[2] = np.arctan2(np.sin(deltaZ[2]), np.cos(deltaZ[2]))

            # Tracking control law
            u_t = u_track[i, :] - Ks[i] @ deltaZ

            # Enforce physical bounds b/c LQR is unconstrained
            u_t = np.clip(u_t, self.action_min, self.action_max)
            
            # STUDENT CODE END

            self.dynsys.forward_np(dt=self.dt, ctrl=u_t)

        self.z_sol = self.dynsys.z_hist[1:, :]
        self.u_sol = self.dynsys.u_hist
        self.tau_sol = t_0 + np.linspace(self.dt, self.dt * n_track, n_track)

        return self.z_sol, self.u_sol, self.tau_sol

    def compute_gains(self, As: NDArray, Bs: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Compute finite-horizon time-varying LQR gains.

        Inputs:
            As:   As[i] is the A matrix at time step i
            Bs:   Bs[i] is the B matrix at time step i

        Outputs:
            Ks: Feedback control matrices from times 0 to n-1
                Optimal control is given by u[i,:] = -Ks[i]*z_t
            Ps: Cost-to-go matrices from times 0 to n
        """
        As = np.asarray(As, dtype=float)
        Bs = np.asarray(Bs, dtype=float)
        if As.shape[0] != Bs.shape[0]:
            raise ValueError("As and Bs must have same horizon length")

        # Initialize list of K and P matrices
        n = As.shape[0]
        Ks = np.zeros((n, 2, 3), dtype=float)
        Ps = np.zeros((n + 1, 3, 3), dtype=float)

        # TODO: Compute sequence of K and P matrices
        # Hint: start from P_{i+1}, compute K_i from the discrete-time
        # Riccati formula, then update P_i.
        # STUDENT CODE START

        # Terminal cost
        Ps[n] = self.L

        #Backward pass
        for i in range(n-1, -1, -1):
            At = As[i]
            Bt = Bs[i]
            Pnext = Ps[i+1]

            # Input cost
            S = self.R + Bt.T @ Pnext @ Bt

            # Optimal gain matrix
            Ks[i] = np.linalg.inv(S) @ (Bt.T @ Pnext @ At)

            # Riccati Update
            Ps[i] = (
                self.Q + (At.T @ Pnext @ At) - (At.T @ Pnext @ Bt @ Ks[i])
            )


        # STUDENT CODE END

        return Ks, Ps

    def linearize_along_traj(
        self, z_traj: NDArray, u_traj: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Linearize system along a trajectory segment."""
        z_traj = np.asarray(z_traj, dtype=float)
        u_traj = np.asarray(u_traj, dtype=float)
        if z_traj.shape[0] != u_traj.shape[0]:
            raise ValueError("z_traj and u_traj must have same number of rows")

        # Linearizations
        n = int(min(self.n, z_traj.shape[0]))
        As = np.zeros((n, 3, 3), dtype=float)
        Bs = np.zeros((n, 3, 2), dtype=float)
        for t in range(n):
            As[t], Bs[t] = self.dynsys.linearize(
                z_t=z_traj[t, :], u_t=u_traj[t, :], discrete=True, dt=self.dt
            )
        return As, Bs


class LQRController(ControllerBackend):
    """Thin wrapper used by ROS2 frontend."""

    def __init__(self, config):
        cfg = dict(config.get("lqr", {}))
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

        self._algo = LQRAlgorithm(
            cost_coefs=cost_coefs,
            dt=dt,
            n=horizon,
            u_min=u_min,
            u_max=u_max,
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



    # Added function to search local window around prev_idx for the nearest reference point.
    # Preventing looping paths and makes testing easier. 
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
                kind=self._ref_kind, dt=self._algo.dt,
                n_steps=self._ref_n_steps, start_state=obs, goal_state=self._goal,
            )


        # Stop and check to see if close to the goal, If we are close to the goal, return zero velocity
        goal_err = np.linalg.norm(obs[:2] - self._goal[:2])
        yaw_err  = np.arctan2(np.sin(obs[2]-self._goal[2]), np.cos(obs[2]-self._goal[2]))
        if goal_err < 0.15 and abs(yaw_err) < 0.15:
            dummy = np.zeros((self._algo.n, 3), dtype=float)
            dummy_u = np.zeros((self._algo.n, 2), dtype=float)
            return np.zeros(2, dtype=float), dummy, dummy_u
        
        # Find closest reference point to robot, using windowed search
        ref_idx = self.closest_reference_index(self._z_ref, obs, self._step)
        ref_idx = min(ref_idx, self._z_ref.shape[0] - 2)
        
        # Extract horizon length of the reference
        zRefWin, uRefWin = self.sample_reference_window(
            self._z_ref, self._u_ref, ref_idx, self._algo.n
        )

        # If trajectory ends, ask robot to stop.
        n_remaining = self._u_ref.shape[0] - ref_idx
        if n_remaining < self._algo.n:
            uRefWin[n_remaining:] = 0.0

        # Solve LQR
        z_sol, u_sol, tau_sol = self._algo.solve(
            z_0=obs, t_0=ref_idx * self._algo.dt,
            z_ref=zRefWin, u_ref=uRefWin
        )

        # Receding horizon rule
        action = u_sol[0, :]

        # advance progress counter
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
