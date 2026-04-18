"""ROS2 planner frontend"""
import importlib
import math
from typing import Any, Dict, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from numpy.typing import NDArray
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from mpc.controller_base import ControllerBackend
from nav_helpers.trajectory import (
    StateActionTrajectory,
    euler_from_quaternion,
    quaternion_from_euler,
)
from nav_helpers_msgs.msg import StateActionTrajectory as TrajMsg


def load_backend_class(backend_class_path: str):
    if ":" not in backend_class_path:
        raise ValueError(
            f"backend_class must be 'module.path:ClassName'. Got '{backend_class_path}'"
        )

    module_name, class_name = backend_class_path.split(":", 1)
    module = importlib.import_module(module_name.strip())
    backend_cls = getattr(module, class_name.strip(), None)
    if backend_cls is None:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )
    return backend_cls


class MPCPlanner(Node):
    def __init__(self) -> None:
        super().__init__("mpc_planner")

        # fmt: off
        self.declare_parameter("backend_class", "mpc.nmpc_algorithm:NMPCController")

        self.declare_parameter("dt", 0.1)
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("seed", 7)
        self.declare_parameter("goal_x", 4.0)
        self.declare_parameter("goal_y", 1.0)
        self.declare_parameter("goal_theta", 0.0)
        self.declare_parameter("v_min", 0.0)
        self.declare_parameter("v_max", 0.5)
        self.declare_parameter("omega_max", 0.8)
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("x_knots", [0.0, 2.0, 4.0, 6.0, 8.0])
        self.declare_parameter("y_low_knots", [-1.0, -1.4, -1.6, -1.6, -1.4])
        self.declare_parameter("y_high_knots", [1.0, 1.4, 1.6, 1.6, 1.4])
        self.declare_parameter("w_pos", 2.0)
        self.declare_parameter("w_theta", 0.1)
        self.declare_parameter("w_u", 0.05)
        self.declare_parameter("w_corr", 5000.0)

        ## MPPI backend params
        self.declare_parameter("mppi.n_traj", 128)
        self.declare_parameter("mppi.horizon", 32)
        self.declare_parameter("mppi.noise_sigma_v", 0.7)
        self.declare_parameter("mppi.noise_sigma_omega", 0.6)
        self.declare_parameter("mppi.temperature", 1.0)
        self.declare_parameter("mppi.w_du", 0.25)

        ## NMPC backend params
        self.declare_parameter("mpc.N", 80)
        self.declare_parameter("mpc.w_du", 0.2)
        self.declare_parameter("mpc.w_pos_T", 50.0)
        self.declare_parameter("mpc.w_theta_T", 0.2)
        self.declare_parameter("mpc.solver_max_iter", 300)
        self.declare_parameter("mpc.solver_tol", 1.0e-6)
        self.declare_parameter("mpc.solver_constr_viol_tol", 1.0e-4)
        # fmt: on

        self._backend = self._build_backend()

        pose_topic = str(self.get_parameter("pose_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        rate_hz = max(1e-3, float(self.get_parameter("control_rate").value))

        backend_class = str(self.get_parameter("backend_class").value)

        # TODO: Add a subscriber, publisher, and timer
        # =========================
        # STUDENT CODE START
        self._latest_state = None
        self._sub_pose = self.create_subscription(
            PoseStamped, pose_topic, self._on_pose, 10
        )
        self._pub_cmd = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        # STUDENT CODE END

        self.traj_marker_pub = self.create_publisher(MarkerArray, "traj_markers", 10)
        self.traj_path_pub = self.create_publisher(Path, "traj_path", 10)
        self.traj_pub = self.create_publisher(TrajMsg, "traj", 10)

        self.get_logger().info(
            f"MPCPlanner up. backend_class={backend_class}, rate={rate_hz:.1f}Hz"
        )

    def _build_backend(self) -> ControllerBackend:
        backend_class_path = str(self.get_parameter("backend_class").value)
        backend_cls = load_backend_class(backend_class_path)
        config = self._read_backend_config()
        backend: ControllerBackend = backend_cls(config)

        if not isinstance(backend, ControllerBackend):
            raise TypeError(
                f"Backend class '{backend_class_path}' must implement ControllerBackend."
            )
        return backend

    def _read_backend_config(self) -> Dict[str, Any]:
        """Regulate config from different backends."""
        return {
            "dt": float(self.get_parameter("dt").value),
            "seed": int(self.get_parameter("seed").value),
            "goal": np.array(
                [
                    float(self.get_parameter("goal_x").value),
                    float(self.get_parameter("goal_y").value),
                    float(self.get_parameter("goal_theta").value),
                ],
                dtype=float,
            ),
            "v_min": float(self.get_parameter("v_min").value),
            "v_max": float(self.get_parameter("v_max").value),
            "omega_max": float(self.get_parameter("omega_max").value),
            "corridor": {
                "x_knots": list(self.get_parameter("x_knots").value),
                "y_low_knots": list(self.get_parameter("y_low_knots").value),
                "y_high_knots": list(self.get_parameter("y_high_knots").value),
            },
            "weights": {
                "w_pos": float(self.get_parameter("w_pos").value),
                "w_theta": float(self.get_parameter("w_theta").value),
                "w_u": float(self.get_parameter("w_u").value),
                "w_corr": float(self.get_parameter("w_corr").value),
            },
            "mppi": {
                "n_traj": int(self.get_parameter("mppi.n_traj").value),
                "horizon": int(self.get_parameter("mppi.horizon").value),
                "noise_sigma_v": float(self.get_parameter("mppi.noise_sigma_v").value),
                "noise_sigma_omega": float(
                    self.get_parameter("mppi.noise_sigma_omega").value
                ),
                "temperature": float(self.get_parameter("mppi.temperature").value),
                "w_du": float(self.get_parameter("mppi.w_du").value),
            },
            "mpc": {
                "N": int(self.get_parameter("mpc.N").value),
                "w_du": float(self.get_parameter("mpc.w_du").value),
                "w_pos_T": float(self.get_parameter("mpc.w_pos_T").value),
                "w_theta_T": float(self.get_parameter("mpc.w_theta_T").value),
                "solver_max_iter": int(self.get_parameter("mpc.solver_max_iter").value),
                "solver_tol": float(self.get_parameter("mpc.solver_tol").value),
                "solver_constr_viol_tol": float(
                    self.get_parameter("mpc.solver_constr_viol_tol").value
                ),
            },
        }

    # TODO: Implement callback functions for subscriber and timer
    # =========================
    # STUDENT CODE START
    def _on_pose(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion(
            float(q.x), float(q.y), float(q.z), float(q.w)
        )
        self._latest_state = np.array(
            [float(msg.pose.position.x), float(msg.pose.position.y), float(yaw)],
            dtype=float,
        )

    def _on_timer(self) -> None:
        if self._latest_state is None:
            return
        try:
            u, X, U = self._backend.get_action(self._latest_state)
            u = np.asarray(u, dtype=float).reshape(2)

        except Exception as exc:
            self.get_logger().error(f"Backend get_action failed: {exc}")
            u = np.array([0.0, 0.0], dtype=float)
            X = None
            U = None

        msg = Twist()
        msg.linear.x = float(u[0])
        msg.angular.z = float(u[1])
        self._pub_cmd.publish(msg)

        if X is not None:
            self.publish_traj_as_markers(X)
            self.publish_traj_as_path(X)

            if U is not None:
                self.publish_traj(X, U, dt=self._backend._params["dt"])

    # STUDENT CODE END

    def publish_traj_as_path(self, arr):
        path = Path()

        # Header for the whole path
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for (x, y, yaw) in arr:
            pose = PoseStamped()

            pose.header.frame_id = "map"
            pose.header.stamp = path.header.stamp  # keep consistent timing

            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            q = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.w = q[0]
            pose.pose.orientation.x = q[1]
            pose.pose.orientation.y = q[2]
            pose.pose.orientation.z = q[3]

            path.poses.append(pose)

        self.traj_path_pub.publish(path)

    def publish_traj_as_markers(self, arr):
        markers = MarkerArray()

        for i, (x, y, yaw) in enumerate(arr):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()

            m.ns = "poses"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0

            q = quaternion_from_euler(0, 0, yaw)
            m.pose.orientation.w = q[0]
            m.pose.orientation.x = q[1]
            m.pose.orientation.y = q[2]
            m.pose.orientation.z = q[3]

            m.scale.x = 0.05  # arrow length
            m.scale.y = 0.02
            m.scale.z = 0.02

            m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)

            markers.markers.append(m)

        self.traj_marker_pub.publish(markers)

    def publish_traj(self, X: NDArray, U: NDArray, dt: float, frame_id: str = "map"):
        """
        Publishes the state action trajectory (needed for LQR to use as reference)
        """
        state_action_traj = StateActionTrajectory(
            states=X,
            actions=U,
            dt=dt,
            frame_id=frame_id,
        )
        self.traj_pub.publish(state_action_traj.to_msg(self.get_clock()))


def main():
    rclpy.init()
    node = MPCPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
