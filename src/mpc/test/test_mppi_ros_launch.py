"""
Test MPPI planner integration.
Run with: colcon test --packages-select mpc_planner --event-handlers console_direct+
"""

import time
import unittest

import launch
import launch_testing
import pytest
from conftest import BaseMPCPlannerTest, make_odom, spin
from launch_ros.actions import Node as LNode


@pytest.mark.launch_test
def generate_test_description():
    return (
        launch.LaunchDescription(
            [
                LNode(
                    package="mpc",
                    executable="mpc",
                    parameters=[
                        {
                            "backend_class": "mpc.mppi_algorithm:MPPIController",
                            "pose_topic": "/robot_pose",
                            "cmd_vel_topic": "/cmd_vel",
                            "control_rate": 10.0,
                            "dt": 0.1,
                            "seed": 7,
                            "goal_x": 7.5,
                            "goal_y": 0.0,
                            "goal_theta": 0.0,
                            "v_min": 0.0,
                            "v_max": 1.0,
                            "omega_max": 1.2,
                            "x_knots": [0.0, 2.0, 4.0, 6.0, 8.0],
                            "y_low_knots": [-1.0, -1.4, -1.6, -1.6, -1.4],
                            "y_high_knots": [1.0, 1.4, 1.6, 1.6, 1.4],
                            "w_pos": 2.0,
                            "w_theta": 0.1,
                            "w_u": 0.05,
                            "w_corr": 5000.0,
                            "mppi.n_traj": 128,
                            "mppi.horizon": 32,
                            "mppi.noise_sigma_v": 0.7,
                            "mppi.noise_sigma_omega": 0.6,
                            "mppi.temperature": 1.0,
                            "mppi.w_du": 0.25,
                        }
                    ],
                    output="screen",
                ),
                launch.actions.TimerAction(
                    period=0.5, actions=[launch_testing.actions.ReadyToTest()]
                ),
            ]
        ),
        {},
    )


class TestMPCPlanner(BaseMPCPlannerTest, unittest.TestCase):
    node_name = "test_mppi_planner"

    def test_publishes_bounded_cmd_vel_after_odom(self):
        assert False, "temporarily disable this test for testing release"
        self._wait_for_odom_subscriber()

        self.ctrl_hist.clear()
        start = time.time()
        px = 0.5
        while time.time() - start < 3.0:
            self.odom_pub.publish(make_odom(px=px, py=0.0, yaw=0.0))
            px += 0.005
            spin(self.node, 0.05)

        spin(self.node, 1.0)
        self.assertGreaterEqual(len(self.ctrl_hist), 8)
        self._assert_cmd_vel_bounded()
