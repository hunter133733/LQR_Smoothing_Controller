"""
Test NMPC planner integration.
Run with: colcon test --packages-select mpc_planner --event-handlers console_direct+
"""

import time
import unittest

import launch
import launch_testing
import pytest
from conftest import BaseMPCPlannerTest, make_odom, spin
from launch_ros.actions import Node as LNode


# TODO: Use /robot_pose topic. This means we need to launch more nodes/launch files so that localization actually works
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
                            "backend_class": "mpc.nmpc_algorithm:NMPCController",
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
                            "mpc.N": 40,  # NOTE: reduce this if planner too slow for the test
                            "mpc.w_du": 0.2,
                            "mpc.w_pos_T": 50.0,
                            "mpc.w_theta_T": 0.2,
                            "mpc.solver_max_iter": 100,  # NOTE: reduce this if planner too slow for the test
                            "mpc.solver_tol": 1.0e-6,
                            "mpc.solver_constr_viol_tol": 1.0e-4,
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
    node_name = "test_nmpc_planner"

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
