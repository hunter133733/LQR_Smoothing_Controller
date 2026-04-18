from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

###############################################################################
#          Defaults: aligned with the previous Dubins NMPC assignment         #
###############################################################################


DEFAULT_CORRIDOR: Dict[str, Any] = {
    "x_knots": [0.0, 2.0, 4.0, 6.0, 8.0],
    "y_low_knots": [-1.0, -1.4, -1.6, -1.6, -1.4],
    "y_high_knots": [1.0, 1.4, 1.6, 1.6, 1.4],
}

DEFAULT_X0 = np.array([0.5, 0.0, 0.0], dtype=float)
DEFAULT_GOAL = np.array([7.5, 0.0, 0.0], dtype=float)

DEFAULT_COST_WEIGHTS: Dict[str, float] = {
    "w_pos": 2.0,
    "w_theta": 0.1,
    "w_u": 0.05,
    "w_corr": 5e3,  # strong penalty for leaving the corridor / knot interval
}

###############################################################################
#                   Corridor helpers (NumPy): do not modify                   #
###############################################################################


def pwl_clamped_np(x: float, x_knots, y_knots) -> float:
    xk = list(map(float, x_knots))
    yk = list(map(float, y_knots))
    assert len(xk) == len(yk) and len(xk) >= 2

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


def y_low_np(px: float, corridor_params: Dict[str, Any]) -> float:
    return pwl_clamped_np(
        px, corridor_params["x_knots"], corridor_params["y_low_knots"]
    )


def y_high_np(px: float, corridor_params: Dict[str, Any]) -> float:
    return pwl_clamped_np(
        px, corridor_params["x_knots"], corridor_params["y_high_knots"]
    )


def corridor_violation_sq(
    px: float, py: float, corridor_params: Dict[str, Any]
) -> float:
    """
    Squared violation of:
      x_min <= px <= x_max
      y_low(px) <= py <= y_high(px)
    """
    x_min = float(corridor_params["x_knots"][0])
    x_max = float(corridor_params["x_knots"][-1])

    yl = float(y_low_np(px, corridor_params))
    yh = float(y_high_np(px, corridor_params))

    vx_lo = max(0.0, x_min - px)
    vx_hi = max(0.0, px - x_max)
    vy_lo = max(0.0, yl - py)
    vy_hi = max(0.0, py - yh)

    return vx_lo * vx_lo + vx_hi * vx_hi + vy_lo * vy_lo + vy_hi * vy_hi


###############################################################################
#            Dubins dynamics + running cost (NumPy): do not modify            #
###############################################################################

from mpc.nmpc_algorithm import dubins_step_numpy


def dubins_running_cost(
    x_next: NDArray,
    u: NDArray,
    goal: NDArray,
    corridor_params: Dict[str, Any],
    weights: Dict[str, float],
) -> float:
    """
    Running cost evaluated at the NEXT state (x_{t+1}):

      ell(x_{t+1}, u_t) =
        w_pos * ||p_{t+1} - p_g||^2
        + w_theta * (theta_{t+1} - theta_g)^2
        + w_u * ||u_t||^2
        + w_corr * Phi(p_{t+1})

    where Phi is squared violation of corridor and px knot interval.
    """
    x_next = np.asarray(x_next, dtype=float).reshape(3)
    u = np.asarray(u, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(3)

    w_pos = float(weights.get("w_pos", 0.0))
    w_theta = float(weights.get("w_theta", 0.0))
    w_u = float(weights.get("w_u", 0.0))
    w_corr = float(weights.get("w_corr", 0.0))

    px, py, th = float(x_next[0]), float(x_next[1]), float(x_next[2])
    gx, gy, gth = float(goal[0]), float(goal[1]), float(goal[2])

    pos_err_sq = (px - gx) ** 2 + (py - gy) ** 2
    th_err_sq = (th - gth) ** 2
    u_sq = float(u[0] * u[0] + u[1] * u[1])

    viol_sq = corridor_violation_sq(px, py, corridor_params)

    return w_pos * pos_err_sq + w_theta * th_err_sq + w_u * u_sq + w_corr * viol_sq


def make_dubins_corridor_stepper(
    dt: float,
    goal: NDArray,
    corridor_params: Dict[str, Any],
    weights: Dict[str, float],
) -> Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """
    Returns a dynamics function compatible with MPPI.rollout:

      step(x, u) -> (x_next, cost)

    where cost is dubins_running_cost(x_next, u, ...).
    """
    goal = np.asarray(goal, dtype=float).reshape(3)

    def step(x: NDArray, u: NDArray) -> Tuple[NDArray, NDArray]:
        x_next = dubins_step_numpy(x, u, dt)
        c = dubins_running_cost(x_next, u, goal, corridor_params, weights)
        return x_next, np.array(c, dtype=float)

    return step


###############################################################################
#                     (STUDENT CODE) MPPI implementation                      #
###############################################################################


@dataclass(frozen=True)
class MPPIParams:
    """General MPPI configuration."""

    n_traj: int = 512
    horizon: int = 30
    act_dim: int = 2

    # NOTE: action_min/action_max can be scalars or length-act_dim arrays.
    action_min: Tuple[float, float] = (0.0, -1.2)  # [v_min, -omega_max]
    action_max: Tuple[float, float] = (1.0, 1.2)  # [v_max,  omega_max]

    # fmt: off
    noise_sigma: Annotated[float | Tuple[float, ...], "can be a scalar (applied to all action dims) or a length-act_dim array."] = (0.6, 0.4)
    temperature: Annotated[float, r"corresponds to $\lambda$ in the handout."] = 1.0
    epsilon: float = 1e-10
    # fmt: on

    w_du: Annotated[
        float, "smoothness penalty weight (added inside rollout for t>=1)."
    ] = 0.2


class MPPI:
    """Model Predictive Path Integral (MPPI) controller."""

    def __init__(
        self,
        params: MPPIParams,
        rng: np.random.Generator,
        dynamics_func: Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]],
    ):
        self.n_traj = int(params.n_traj)
        self.horizon = int(params.horizon)
        self.act_dim = int(params.act_dim)

        self.action_min = np.asarray(params.action_min, dtype=float).reshape(
            self.act_dim
        )
        self.action_max = np.asarray(params.action_max, dtype=float).reshape(
            self.act_dim
        )
        assert np.all(self.action_min < self.action_max)

        self.noise_sigma = np.asarray(params.noise_sigma, dtype=float)
        if self.noise_sigma.size == 1:
            self.noise_sigma = np.full(
                (self.act_dim,), float(self.noise_sigma.reshape(-1)[0])
            )
        self.noise_sigma = self.noise_sigma.reshape(
            self.act_dim,
        )

        self.temperature = float(params.temperature)
        self.epsilon = float(params.epsilon)
        self.w_du = float(params.w_du)

        assert self.n_traj >= 2
        assert self.horizon >= 1
        assert self.act_dim >= 1
        assert self.temperature > 0.0

        self.actions = np.zeros((self.horizon, self.act_dim), dtype=float)
        self.rng = rng
        self.dynamics_func = dynamics_func

    def reset(self) -> None:
        self.actions[...] = 0.0

    def sample_noise(self) -> NDArray:
        """
        Returns:
          noise: (K, H, m) Gaussian noise with std = self.noise_sigma (per-dimension).
        """
        noise = self.rng.normal(size=(self.n_traj, self.horizon, self.act_dim))
        return noise * self.noise_sigma.reshape(1, 1, self.act_dim)

    def rollout(self, observation: NDArray, actions: NDArray) -> NDArray:
        r"""
        Roll out K trajectories with supplied action sequences and return per-step costs.

        Args:
          observation: initial state x_0, shape (3,)
          actions: (K, H, m) array where actions[k, t] is u^{(k)}_t

        Returns:
          costs: (K, H) where costs[k, t] is the running cost at time t.
                 Include the smoothness term w_du*||u_t-u_{t-1}||^2 for t>=1.
        """
        costs = np.full((1,), fill_value=np.nan)  # NOTE: dummy

        # TODO: Implement MPPI rollout
        # =========================
        # STUDENT CODE START
        observation = np.asarray(observation, dtype=float).reshape(
            3,
        )
        actions = np.asarray(actions, dtype=float)
        assert actions.ndim == 3, "actions must be (K, H, m)"
        K, H, m = actions.shape
        assert H == self.horizon and m == self.act_dim

        costs = np.zeros((K, H), dtype=float)

        for k in range(K):
            x = observation.copy()
            u_prev = None
            for t in range(H):
                u = actions[k, t]
                x, c = self.dynamics_func(x, u)
                c_val = float(np.asarray(c).reshape(-1)[0])
                if t >= 1 and u_prev is not None:
                    du = u - u_prev
                    c_val += self.w_du * float(np.dot(du, du))
                costs[k, t] = c_val
                u_prev = u
        # STUDENT CODE END

        return costs

    def get_action(self, observation: NDArray) -> NDArray:
        """
        One MPPI update step.

        Returns:
          best_action: (m,)
        """
        best_action = np.full((2,), fill_value=np.nan)  # NOTE: dummy

        # TODO: Implement MPPI per the handout.
        # =========================
        # STUDENT CODE START
        observation = np.asarray(observation, dtype=float)

        # 1) sample noise and build perturbed actions (K, H, m)
        noise = self.sample_noise()
        perturbed = self.actions.reshape(1, self.horizon, self.act_dim) + noise
        perturbed = np.clip(
            perturbed,
            self.action_min.reshape(1, 1, self.act_dim),
            self.action_max.reshape(1, 1, self.act_dim),
        )

        # 2) rollout -> per-step costs, then total costs
        costs = self.rollout(observation, perturbed)
        total_costs = np.sum(costs, axis=1)  # (K,)

        # 3) importance weights with baseline subtraction
        min_cost = float(np.min(total_costs))
        norm_costs = total_costs - min_cost
        weights = np.exp((-1.0 / self.temperature) * norm_costs)  # (K,)
        w_sum = float(np.sum(weights)) + self.epsilon

        # 4) weighted average update of the whole sequence
        weighted_actions = np.sum(
            weights.reshape(self.n_traj, 1, 1) * perturbed, axis=0
        )  # (H, m)
        new_actions = weighted_actions / w_sum
        new_actions = np.clip(
            new_actions,
            self.action_min.reshape(1, self.act_dim),
            self.action_max.reshape(1, self.act_dim),
        )

        # 5) receding-horizon: return first action, then shift
        best_action = new_actions[0].copy()
        self.actions = np.roll(new_actions, shift=-1, axis=0)
        self.actions[-1] = self.actions[-2]  # duplicate last
        # STUDENT CODE END

        return best_action


###############################################################################
#                              ros2 intergration                              #
###############################################################################

from mpc.controller_base import ControllerBackend


class MPPIController(ControllerBackend):
    """A thin backend wrapper implements necessary methods used by ROS frontend."""

    def __init__(self, config: Dict[str, Any]):
        dt = float(config.get("dt", 0.1))
        seed = int(config.get("seed", 7))
        goal = np.asarray(config.get("goal", DEFAULT_GOAL), dtype=float).reshape(3)
        corridor = dict(config.get("corridor", DEFAULT_CORRIDOR))
        weights = dict(config.get("weights", DEFAULT_COST_WEIGHTS))

        v_min = float(config.get("v_min", 0.0))
        v_max = float(config.get("v_max", 1.0))
        omega_max = float(config.get("omega_max", 1.2))

        mppi_cfg = dict(config.get("mppi", {}))
        params = MPPIParams(
            n_traj=int(mppi_cfg.get("n_traj", 128)),
            horizon=int(mppi_cfg.get("horizon", 32)),
            act_dim=2,
            action_min=(v_min, -omega_max),
            action_max=(v_max, omega_max),
            noise_sigma=(
                float(mppi_cfg.get("noise_sigma_v", 0.7)),
                float(mppi_cfg.get("noise_sigma_omega", 0.6)),
            ),
            temperature=float(mppi_cfg.get("temperature", 1.0)),
            w_du=float(mppi_cfg.get("w_du", 0.25)),
        )

        stepper = make_dubins_corridor_stepper(
            dt=dt,
            goal=goal,
            corridor_params=corridor,
            weights=weights,
        )
        self._action_min = np.array([v_min, -omega_max], dtype=float)
        self._action_max = np.array([v_max, omega_max], dtype=float)
        self._controller = MPPI(
            params=params,
            rng=np.random.default_rng(seed=seed),
            dynamics_func=stepper,
        )

    # @override
    def get_action(self, observation: NDArray) -> NDArray:
        """Return control action [v, omega] for the provided state."""
        u = self._controller.get_action(np.asarray(observation, dtype=float).reshape(3))
        return (
            np.clip(
                np.asarray(u, dtype=float).reshape(2),
                self._action_min,
                self._action_max,
            ),
            None,
        )


###############################################################################
#                  Demo (run with: python mppi_algorithm.py)                  #
###############################################################################


def simulate_mppi(
    controller: "MPPI",
    x0: NDArray,
    dt: float,
    n_steps: int,
) -> Tuple[NDArray, NDArray]:
    """
    Closed-loop simulation with receding-horizon MPPI.

    Returns:
      X: (n_steps+1, 3) states
      U: (n_steps, 2) controls
    """
    x = np.asarray(x0, dtype=float).reshape(
        3,
    )
    X = np.zeros((n_steps + 1, 3), dtype=float)
    U = np.zeros((n_steps, 2), dtype=float)
    X[0] = x
    for t in range(n_steps):
        u = controller.get_action(x)
        U[t] = u
        x = dubins_step_numpy(x, u, dt)
        X[t + 1] = x
    return X, U


def plot_trajectory(
    X: NDArray,
    corridor_params: Dict[str, Any],
    goal: NDArray,
    title: str = "Dubins Car MPPI Trajectory in Corridor",
):
    """This plots the final trajectry"""

    import matplotlib.pyplot as plt

    if np.any(np.isnan(X)):  # pragma: no cover
        print("No valid trajectory to plot.")
        return

    px = X[:, 0]
    py = X[:, 1]

    xk = corridor_params["x_knots"]
    yl = corridor_params["y_low_knots"]
    yh = corridor_params["y_high_knots"]
    plt.plot(xk, yl, "b--", label="Lower boundary")
    plt.plot(xk, yh, "b--", label="Upper boundary")

    plt.plot(px, py, "r.-", label="Trajectory")

    plt.plot(X[0, 0], X[0, 1], "go", markersize=10, label="Start")
    plt.plot(goal[0], goal[1], "g*", markersize=12, label="Goal")

    plt.xlabel("px")
    plt.ylabel("py")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("mppi.png")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    # Default demo settings: tuned for a quick visible trajectory
    dt = 0.1
    weights = dict(DEFAULT_COST_WEIGHTS)
    stepper = make_dubins_corridor_stepper(
        dt=dt, goal=DEFAULT_GOAL, corridor_params=DEFAULT_CORRIDOR, weights=weights
    )

    rng = np.random.default_rng(7)
    params = MPPIParams(
        n_traj=1024,
        horizon=64,
        act_dim=2,
        action_min=(0.0, -1.2),
        action_max=(1.0, 1.2),
        noise_sigma=(0.7, 0.6),
        temperature=1.0,
        w_du=0.25,
    )
    ctrl = MPPI(params=params, rng=rng, dynamics_func=stepper)

    X, U = simulate_mppi(
        ctrl, DEFAULT_X0, dt=dt, n_steps=100
    )  # NOTE: reduce n_steps for quick test
    plot_trajectory(X, DEFAULT_CORRIDOR, DEFAULT_GOAL)
