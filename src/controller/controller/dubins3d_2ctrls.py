from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import heterocl as hcl
except ImportError:  # pragma: no cover
    _has_hcl = False
else:  # pragma: no cover
    _has_hcl = True


class DynSys(ABC):
    """Dynamical systems base class.

    Attributes:
    - z_t: current state, shape (self.state_dims,)
    - state_dims: number of dimensions in state
    - ctrl_dims: number of dimensions in control
    - z_hist:
        state history, shape (N, self.state_dims) where N is the number of
        historical time steps. Initially, N = 1.
    - u_hist:
        control history, shape (M, self.ctrl_dims) where M is the number of
        historical time steps. Initially, M = 0.
    - dstb_dims:
        disturbance history, shape (M, self.ctrl_dims) where M is the number of
        historical time steps. Initially, M = 0.
    - info: Dictionary of information that defines that system, hashable
    """

    def __init__(
        self,
        z_0: NDArray | None = None,
        dt: float | None = None,
        u_min: NDArray | None = None,
        u_max: NDArray | None = None,
        d_min: NDArray | None = None,
        d_max: NDArray | None = None,
        u_mode: str = "min",
        d_mode: str = "max",
    ):
        if z_0 is None:
            z_0 = np.zeros(self.state_dims, dtype=float)
        self.reset(z_0=z_0)
        self.dt = float(dt if dt is not None else 0.1)
        self.u_min = np.asarray(
            u_min if u_min is not None else np.full(self.ctrl_dims, -1.0), dtype=float
        )
        self.u_max = np.asarray(
            u_max if u_max is not None else np.full(self.ctrl_dims, 1.0), dtype=float
        )
        self.d_min = np.asarray(
            d_min if d_min is not None else np.zeros(self.dstb_dims), dtype=float
        )
        self.d_max = np.asarray(
            d_max if d_max is not None else np.zeros(self.dstb_dims), dtype=float
        )

        if (u_mode, d_mode) not in {("min", "max"), ("max", "min")}:
            raise ValueError("Combination of `u_mode` and `d_mode` is not supported")
        self.u_mode = u_mode
        self.d_mode = d_mode

    @property
    @abstractmethod
    def state_dims(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def ctrl_dims(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dstb_dims(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def periodic_dims(self):
        raise NotImplementedError

    def reset(self, z_0: NDArray):
        z_0 = np.asarray(z_0, dtype=float).reshape(self.state_dims)
        self.z_t = z_0.copy()
        self.z_hist = z_0.reshape(1, self.state_dims)
        self.u_hist = np.empty((0, self.ctrl_dims), dtype=float)

    @abstractmethod
    def opt_ctrl(self, t: float, state, spat_deriv):
        raise NotImplementedError

    @abstractmethod
    def opt_dstb(self, t: float, state, spat_deriv):
        raise NotImplementedError

    @abstractmethod
    def dynamics(self, t: float, state, u, d):
        raise NotImplementedError

    @abstractmethod
    def opt_ctrl_np(self, state: NDArray, spat_deriv: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def opt_dstb_np(self, state: NDArray, spat_deriv: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def dynamics_np(
        self, t: float, state: NDArray, ctrl: NDArray, dstb: NDArray
    ) -> NDArray:
        raise NotImplementedError

    def forward_np(self, dt: float | None = None, *args, **kwargs) -> NDArray:
        if dt is None:
            dt = self.dt

        state = kwargs.setdefault("state", self.z_t)
        kwargs.setdefault("t", 0.0)
        ctrl = kwargs.setdefault("ctrl", np.zeros(self.ctrl_dims, dtype=float))
        dstb = kwargs.setdefault("dstb", np.zeros(self.dstb_dims, dtype=float))

        next_state = np.asarray(state, dtype=float).reshape(self.state_dims) + float(
            dt
        ) * self.dynamics_np(*args, **kwargs)
        for dim in self.periodic_dims:
            angle = float(next_state[dim])
            next_state[dim] = np.arctan2(np.sin(angle), np.cos(angle))

        self.update_state(
            next_state, np.asarray(ctrl, dtype=float), np.asarray(dstb, dtype=float)
        )
        return next_state

    def update_state(self, new_state: NDArray, ctrl: NDArray, dstb: NDArray):
        _ = dstb
        self.z_t = np.asarray(new_state, dtype=float).reshape(self.state_dims)
        self.z_hist = np.vstack((self.z_hist, self.z_t))
        self.u_hist = np.vstack(
            (self.u_hist, np.asarray(ctrl, dtype=float).reshape(self.ctrl_dims))
        )


class DubinsCar3D2Ctrls(DynSys):
    """Dubins Car with speed v and turn rate w as control inputs

    Dynamics:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = w

    Control:
        u[0] = v
        u[1] = w
    """

    state_dims = 3
    ctrl_dims = 2
    dstb_dims = 3
    periodic_dims = [2]
    ignore_dims = [2]

    def opt_ctrl(self, t, state, spat_deriv):
        _ = t
        if not _has_hcl:
            raise ImportError("heterocl is required but is not installed")

        opt_speed = hcl.scalar(self.u_max[0], "opt_speed")
        opt_w = hcl.scalar(self.u_max[1], "opt_w")
        in4 = hcl.scalar(0, "in4")

        # Declare hcl scalars for the coefficient
        deriv0 = hcl.scalar(0, "deriv0")
        deriv1 = hcl.scalar(0, "deriv1")
        theta = hcl.scalar(0, "theta")
        deriv0[0] = spat_deriv[0]
        deriv1[0] = spat_deriv[1]
        theta[0] = state[2]
        coefficient = deriv0[0] * hcl.cos(theta[0]) + deriv1[0] * hcl.sin(theta[0])

        with hcl.if_(self.u_mode == "min"):
            with hcl.if_(coefficient > 0):
                opt_speed[0] = self.u_min[0]
            with hcl.if_(spat_deriv[2] > 0):
                opt_w[0] = self.u_min[1]
        with hcl.if_(self.u_mode == "max"):
            with hcl.if_(coefficient < 0):
                opt_speed[0] = self.u_min[0]
            with hcl.if_(spat_deriv[2] < 0):
                opt_w[0] = self.u_min[1]
        return (opt_speed[0], opt_w[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        _ = (t, state, spat_deriv)
        if not _has_hcl:
            raise ImportError("heterocl is required but is not installed")
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, u, d):
        _ = (t, d)
        if not _has_hcl:
            raise ImportError("heterocl is required but is not installed")
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        x_dot[0] = u[0] * hcl.cos(state[2])
        y_dot[0] = u[0] * hcl.sin(state[2])
        theta_dot[0] = u[1]
        return (x_dot[0], y_dot[0], theta_dot[0])

    def opt_ctrl_np(self, state: NDArray, spat_deriv: NDArray) -> NDArray:
        opt_speed = self.u_max[0]
        opt_w = self.u_max[1]
        coefficient = spat_deriv[0] * np.cos(state[2]) + spat_deriv[1] * np.sin(
            state[2]
        )
        if self.u_mode == "min":
            if coefficient > 0:
                opt_speed = self.u_min[0]
            if spat_deriv[2] > 0:
                opt_w = self.u_min[1]
        else:
            if coefficient < 0:
                opt_speed = self.u_min[0]
            if spat_deriv[2] < 0:
                opt_w = self.u_min[1]
        return np.array([opt_speed, opt_w], dtype=float)

    def opt_dstb_np(self, state: NDArray, spat_deriv: NDArray) -> NDArray:
        _ = (state, spat_deriv)
        return np.zeros(self.dstb_dims, dtype=float)

    def dynamics_np(
        self, t: float, state: NDArray, ctrl: NDArray, dstb: NDArray
    ) -> NDArray:
        _ = (t, dstb)
        x, y, theta = np.asarray(state, dtype=float).reshape(3)
        v, w = np.asarray(ctrl, dtype=float).reshape(2)
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = w
        return np.array([x_dot, y_dot, theta_dot], dtype=float)

    def linearize(
        self,
        z_t: NDArray | None = None,
        u_t: NDArray | None = None,
        discrete: bool = True,
        dt: float | None = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Linearize Dubins dynamics around a nominal state and control.

        If `discrete` is True, return the forward-Euler discrete-time matrices.
        """
        if z_t is None:
            z_t = self.z_t
        if u_t is None:
            u_t = np.zeros(self.ctrl_dims, dtype=float)
        if dt is None:
            dt = self.dt

        z_t = np.asarray(z_t, dtype=float).reshape(3)
        u_t = np.asarray(u_t, dtype=float).reshape(2)
        theta_t = float(z_t[2])
        v_t = float(u_t[0])

        A_cont = np.array(
            [
                [0.0, 0.0, -v_t * np.sin(theta_t)],
                [0.0, 0.0, v_t * np.cos(theta_t)],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        B_cont = np.array(
            [
                [np.cos(theta_t), 0.0],
                [np.sin(theta_t), 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        if not discrete:
            return A_cont, B_cont

        A = np.eye(self.state_dims, dtype=float) + float(dt) * A_cont
        B = float(dt) * B_cont
        return A, B
