"""
Common test utilities for MPC planner integration tests.
"""

import math
import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def spin(node, seconds: float):
    """Spin a ROS2 node for a specified duration."""
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)


def make_odom(px: float = 0.5, py: float = 0.0, yaw: float = 0.0) -> Odometry:
    """Create an Odometry message with specified pose."""
    msg = Odometry()
    msg.header.frame_id = "odom"
    msg.child_frame_id = "base_link"
    msg.pose.pose.position.x = float(px)
    msg.pose.pose.position.y = float(py)
    half = 0.5 * float(yaw)
    msg.pose.pose.orientation.z = math.sin(half)
    msg.pose.pose.orientation.w = math.cos(half)
    return msg


class BaseMPCPlannerTest:
    """Base test class for MPC planners (MPPI, NMPC, etc.)"""

    node_name = "test_mpc_planner"  # Override in subclass

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node(self.node_name)
        self.ctrl_hist = []
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.cmd_vel_sub = self.node.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_cb, 10
        )

    def tearDown(self):
        self.node.destroy_node()

    def _cmd_vel_cb(self, msg: Twist):
        self.ctrl_hist.append((float(msg.linear.x), float(msg.angular.z)))

    def _wait_for_odom_subscriber(self, timeout_iterations=80):
        """Wait for odom subscriber to connect"""
        for _ in range(timeout_iterations):
            if self.odom_pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        raise RuntimeError("No /odom subscriber")

    def _assert_cmd_vel_bounded(
        self, min_v=-1e-3, max_v=1.05, max_w=1.25, min_active_v=0.05
    ):
        """Assert control commands are finite and within bounds"""
        vs = np.array([v for v, _ in self.ctrl_hist], dtype=float)
        ws = np.array([w for _, w in self.ctrl_hist], dtype=float)

        assert np.all(np.isfinite(vs)), "Linear velocities contain NaN/Inf"
        assert np.all(np.isfinite(ws)), "Angular velocities contain NaN/Inf"
        assert np.min(vs) >= min_v, f"Min velocity {np.min(vs)} < {min_v}"
        assert np.max(vs) <= max_v, f"Max velocity {np.max(vs)} > {max_v}"
        assert (
            np.max(np.abs(ws)) <= max_w
        ), f"Max angular velocity {np.max(np.abs(ws))} > {max_w}"
        assert np.max(vs) > min_active_v, f"Max velocity {np.max(vs)} too low"


@pytest.fixture
def mpc_test_base():
    """Fixture providing base test functionality"""
    return BaseMPCPlannerTest
