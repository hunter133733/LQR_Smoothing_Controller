import numpy as np

from mpc.mppi_algorithm import (
    DEFAULT_CORRIDOR,
    DEFAULT_COST_WEIGHTS,
    DEFAULT_GOAL,
    DEFAULT_X0,
    MPPI,
    MPPIParams,
    dubins_step_numpy,
    make_dubins_corridor_stepper,
    y_high_np,
    y_low_np,
)


def test_rollout_shapes_and_values_simple_dubins():
    """
    Deterministic rollout check using Dubins dynamics with a very simple cost.
    We set weights so that cost is just ||p||^2 at the next state.

    Tests:
      - rollout returns correct shape (K, H)
      - rollout correctly propagates Dubins dynamics
      - running cost is evaluated at x_{t+1}
      - no smoothness penalty when w_du = 0

    This isolates correctness of dynamics + per-step cost accumulation.
    """
    corridor = {
        "x_knots": [-10.0, 10.0],
        "y_low_knots": [-10.0, -10.0],
        "y_high_knots": [10.0, 10.0],
    }
    goal = np.array([0.0, 0.0, 0.0], dtype=float)
    weights = {"w_pos": 1.0, "w_theta": 0.0, "w_u": 0.0, "w_corr": 0.0}

    dt = 1.0
    stepper = make_dubins_corridor_stepper(
        dt=dt, goal=goal, corridor_params=corridor, weights=weights
    )

    rng = np.random.default_rng(0)
    params = MPPIParams(
        n_traj=2,
        horizon=3,
        act_dim=2,
        action_min=(0.0, -10.0),
        action_max=(10.0, 10.0),
        noise_sigma=(0.0, 0.0),
        temperature=1.0,
        w_du=0.0,
    )
    ctrl = MPPI(params=params, rng=rng, dynamics_func=stepper)

    actions = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=float,
    )
    obs0 = np.array([0.0, 0.0, 0.0], dtype=float)

    costs = ctrl.rollout(obs0, actions)
    assert isinstance(costs, np.ndarray)
    assert costs.shape == (2, 3)

    expected = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=float)
    np.testing.assert_allclose(costs, expected, rtol=0, atol=1e-12)


def test_rollout_includes_du_smoothness_penalty():
    """
    Unit-test that rollout adds w_du * ||u_t - u_{t-1}||^2 for t>=1.
    We use a dynamics function that returns zero running cost, so all cost is from w_du.

    Tests:
      - smoothness penalty w_du * ||u_t - u_{t-1}||^2 is applied for t >= 1
      - no smoothness penalty at t = 0
      - smoothness term is added to running cost correctly

    Uses zero running cost so total cost is purely from smoothness.
    """

    def zero_cost_step(state: np.ndarray, action: np.ndarray):
        return np.asarray(state).copy(), np.array(0.0, dtype=float)

    rng = np.random.default_rng(0)
    params = MPPIParams(
        n_traj=2,  # must satisfy MPPI's n_traj >= 2 assertion
        horizon=3,
        act_dim=2,
        action_min=(-10.0, -10.0),
        action_max=(10.0, 10.0),
        noise_sigma=(0.0, 0.0),
        temperature=1.0,
        w_du=2.0,
    )
    ctrl = MPPI(params=params, rng=rng, dynamics_func=zero_cost_step)

    # Two identical trajectories; test only uses the first row of expected values.
    actions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]],
        ],
        dtype=float,
    )
    obs0 = np.zeros((3,), dtype=float)

    costs = ctrl.rollout(obs0, actions)
    assert costs.shape == (2, 3)

    expected_row = np.array([0.0, 2.0, 8.0], dtype=float)
    np.testing.assert_allclose(costs[0], expected_row, rtol=0, atol=1e-12)
    np.testing.assert_allclose(costs[1], expected_row, rtol=0, atol=1e-12)


def test_get_action_respects_bounds_and_shifts_dubins():
    """
    Tests:
      - get_action returns shape (2,)
      - returned action respects control bounds
      - updated nominal sequence respects bounds
      - receding-horizon shift duplicates the last action
      - MPPI produces forward-progress action toward goal

    This validates the MPPI update + shift logic.
    """
    rng = np.random.default_rng(123)

    weights = dict(DEFAULT_COST_WEIGHTS)
    stepper = make_dubins_corridor_stepper(
        dt=0.1, goal=DEFAULT_GOAL, corridor_params=DEFAULT_CORRIDOR, weights=weights
    )

    params = MPPIParams(
        n_traj=1500,
        horizon=35,
        act_dim=2,
        action_min=(0.0, -1.2),
        action_max=(1.0, 1.2),
        noise_sigma=(0.6, 0.5),
        temperature=0.9,
        w_du=0.2,
    )
    ctrl = MPPI(params=params, rng=rng, dynamics_func=stepper)

    obs = DEFAULT_X0.copy()
    u0 = ctrl.get_action(obs)

    assert u0.shape == (2,)
    assert np.all(np.isfinite(u0))

    v, om = float(u0[0]), float(u0[1])
    assert 0.0 <= v <= 1.0 + 1e-9
    assert -1.2 - 1e-9 <= om <= 1.2 + 1e-9

    assert v > 0.15, f"Expected forward progress action, got v={v}"

    np.testing.assert_allclose(
        ctrl.actions[-1, :], ctrl.actions[-2, :], rtol=0, atol=1e-12
    )

    assert np.max(ctrl.actions[:, 0]) <= 1.0 + 1e-9
    assert np.min(ctrl.actions[:, 0]) >= 0.0 - 1e-9
    assert np.max(ctrl.actions[:, 1]) <= 1.2 + 1e-9
    assert np.min(ctrl.actions[:, 1]) >= -1.2 - 1e-9


def test_closed_loop_progress_and_stays_in_corridor():
    """
    Tests:
      - closed-loop MPPI drives Dubins car forward (px increases)
      - trajectory remains inside corridor (soft-constraint penalty effective)
      - repeated receding-horizon updates are stable

    This verifies end-to-end behavior of MPPI controller.
    """
    rng = np.random.default_rng(7)
    weights = dict(DEFAULT_COST_WEIGHTS)
    stepper = make_dubins_corridor_stepper(
        dt=0.1, goal=DEFAULT_GOAL, corridor_params=DEFAULT_CORRIDOR, weights=weights
    )

    params = MPPIParams(
        n_traj=1800,
        horizon=40,
        act_dim=2,
        action_min=(0.0, -1.2),
        action_max=(1.0, 1.2),
        noise_sigma=(0.7, 0.6),
        temperature=1.0,
        w_du=0.25,
    )
    ctrl = MPPI(params=params, rng=rng, dynamics_func=stepper)

    x = DEFAULT_X0.copy()
    dt = 0.1

    for _ in range(18):
        u = ctrl.get_action(x)
        x = dubins_step_numpy(x, u, dt)

        px, py = float(x[0]), float(x[1])
        yl = y_low_np(px, DEFAULT_CORRIDOR)
        yh = y_high_np(px, DEFAULT_CORRIDOR)

        assert py >= yl - 0.15
        assert py <= yh + 0.15

    assert float(x[0]) > 2.0, f"Expected x-progress; got px={x[0]}"
