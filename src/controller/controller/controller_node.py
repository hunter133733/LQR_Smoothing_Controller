"""ROS2 planner frontend for LQR backend controllers."""

import importlib
import math
from typing import Any, Dict, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from controller.controller_base import ControllerBackend
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


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_node")

        # fmt: off
        self.declare_parameter("backend_class", "controller.lqr_algorithm:LQRController")
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("nom_traj_topic", "/traj")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("lqr.horizon", 25)
        self.declare_parameter("lqr.x_cost", 5.0)
        self.declare_parameter("lqr.y_cost", 5.0)
        self.declare_parameter("lqr.theta_cost", 1.0)
        self.declare_parameter("lqr.v_cost", 0.3)
        self.declare_parameter("lqr.w_cost", 0.3)
        self.declare_parameter("lqr.v_min", -0.2)
        self.declare_parameter("lqr.v_max", 1.0)
        self.declare_parameter("lqr.w_min", -1.2)
        self.declare_parameter("lqr.w_max", 1.2)
        self.declare_parameter("reference.kind", "s_curve")
        self.declare_parameter("reference.n_steps", 500)
        self.declare_parameter("goal_x", 3.5)
        self.declare_parameter("goal_y", 2.5)
        self.declare_parameter("goal_theta", 0.0)

        self.declare_parameter("cbf.gamma_cbf", 2.0)
        self.declare_parameter("cbf.lookahead_distance", 0.8)
        self.declare_parameter("cbf.center_margin_buffer", 0.05)
        self.declare_parameter("cbf.v_min", 0.0)
        self.declare_parameter("cbf.v_max", 1.0)
        self.declare_parameter("cbf.omega_max", 1.2)
        self.declare_parameter("cbf.w_cbf_v", 1.0)
        self.declare_parameter("cbf.w_cbf_omega", 0.05)
        self.declare_parameter("cbf.obstacle.center", [4.3, 0.15])
        self.declare_parameter("cbf.obstacle.radius", 0.5)
        self.declare_parameter("cbf.obstacle.safety_margin", 0.2)

        self.declare_parameter("hj.epsilon", 0.1)
        self.declare_parameter("hj.obstacle.center", [4.3, 0.15])
        self.declare_parameter("hj.obstacle.radius", 0.5)
        self.declare_parameter("hj.obstacle.safety_margin", 0.2)

        # fmt: on

        self._backend = self._build_backend()
        pose_topic = str(self.get_parameter("pose_topic").value)
        nom_traj_topic = str(self.get_parameter("nom_traj_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        rate_hz = max(1e-3, float(self.get_parameter("control_rate").value))
        self._latest_state = None
        self._latest_traj = None

        self.obs_pub = self.create_publisher(MarkerArray, "obstacle", 10)
        self.traj_path_pub = self.create_publisher(Path, "lqr_traj_path", 10)
        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)

        # TODO: Subs for robot pose, nominal trajectory, pub for /cmd_vel
        # Hint: follow the same ROS2 pattern as MPC Planner
        # STUDENT CODE START
        self._pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self._on_pose,
            10,
        )

        self._traj_sub = self.create_subscription(
            TrajMsg,
            nom_traj_topic,
            self._on_nom_traj,
            10,
        )

        self._cmd_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10,
        )
        # STUDENT CODE END

        self.get_logger().info(f"LQRPlanner up. rate={rate_hz:.1f}Hz")

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
        return {
            "dt": float(self.get_parameter("dt").value),
            "lqr": {
                "horizon": int(self.get_parameter("lqr.horizon").value),
                "x_cost": float(self.get_parameter("lqr.x_cost").value),
                "y_cost": float(self.get_parameter("lqr.y_cost").value),
                "theta_cost": float(self.get_parameter("lqr.theta_cost").value),
                "v_cost": float(self.get_parameter("lqr.v_cost").value),
                "w_cost": float(self.get_parameter("lqr.w_cost").value),
                "v_min": float(self.get_parameter("lqr.v_min").value),
                "v_max": float(self.get_parameter("lqr.v_max").value),
                "w_min": float(self.get_parameter("lqr.w_min").value),
                "w_max": float(self.get_parameter("lqr.w_max").value),
            },
            "cbf": {
                "gamma_cbf": float(self.get_parameter("cbf.gamma_cbf").value),
                "lookahead_distance": float(
                    self.get_parameter("cbf.lookahead_distance").value
                ),
                "center_margin_buffer": float(
                    self.get_parameter("cbf.center_margin_buffer").value
                ),
                "v_min": float(self.get_parameter("cbf.v_min").value),
                "v_max": float(self.get_parameter("cbf.v_max").value),
                "omega_max": float(self.get_parameter("cbf.omega_max").value),
                "w_cbf_v": float(self.get_parameter("cbf.w_cbf_v").value),
                "w_cbf_omega": float(self.get_parameter("cbf.w_cbf_omega").value),
                "obstacle": {
                    "center": np.array(self.get_parameter("cbf.obstacle.center").value),
                    "radius": float(self.get_parameter("cbf.obstacle.radius").value),
                    "safety_margin": float(
                        self.get_parameter("cbf.obstacle.safety_margin").value
                    ),
                },
            },
            "hj": {
                "epsilon": float(self.get_parameter("hj.epsilon").value),
                # Note: if any of the obstacle parameters change, you need to modify accordingly and re-run the
                # HJ pre-computation script to generate new value function tables
                "obstacle": {
                    "center": np.array(self.get_parameter("hj.obstacle.center").value),
                    "radius": float(self.get_parameter("hj.obstacle.radius").value),
                    "safety_margin": float(
                        self.get_parameter("hj.obstacle.safety_margin").value
                    ),
                },
            },
            "reference": {
                "kind": str(self.get_parameter("reference.kind").value),
                "n_steps": int(self.get_parameter("reference.n_steps").value),
            },
            "goal": np.array(
                [
                    float(self.get_parameter("goal_x").value),
                    float(self.get_parameter("goal_y").value),
                    float(self.get_parameter("goal_theta").value),
                ],
                dtype=float,
            ),
        }

    def _on_pose(self, msg: PoseStamped) -> None:
        # TODO: Save latest state in self._latest_state in numpy format
        # STUDENT CODE START
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)

        self._latest_state = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                yaw,
            ],
            dtype=float,
        )


        # STUDENT CODE END
        return

    def _on_nom_traj(self, msg: TrajMsg) -> None:
        # TODO: Save latest state in self._latest_traj as StateActionTrajectory
        # STUDENT CODE START
        self._latest_traj = StateActionTrajectory.from_msg(msg)
        # STUDENT CODE END
        return

    def _on_timer(self) -> None:
        if self._latest_state is None or self._latest_traj is None:
            return

        u = np.array([0.0, 0.0], dtype=float)
        Z = None
        U = None
        # TODO: obtain control (and state & control sequence if LQR for visualization),
        #       and then publish the control
        # Hint: query the backend for [v, omega]. If something fails, publish
        # a safe zero command instead of crashing the node.
        # STUDENT CODE START
        try:
            u, Z, U = self._backend.get_action(self._latest_state, self._latest_traj)
        except Exception as e:
            self.get_logger().warn(f"Controller error: {e}")
            u = np.array([0.0, 0.0], dtype=float)
            Z = None
            U = None
        
        cmd = Twist()
        cmd.linear.x = float(u[0])
        cmd.angular.z = float(u[1])
        self._cmd_pub.publish(cmd)
        # STUDENT CODE END

        if Z is not None:
            self.publish_traj_as_path(Z)

        self.publish_obstacle()

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

    def make_filled_circle(self, x, y, r_inner, frame_id="map"):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = "inner_circle"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = 2 * r_inner
        m.scale.y = 2 * r_inner
        m.scale.z = 0.01  # thin disk

        m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)

        return m

    def make_circle_outline(self, x, y, r_outer, frame_id="map", n=100):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = "outer_circle"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.scale.x = 0.03  # line width

        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

        for i in range(n + 1):  # close loop
            theta = 2.0 * np.pi * i / n

            p = Point()
            p.x = x + r_outer * np.cos(theta)
            p.y = y + r_outer * np.sin(theta)
            p.z = 0.01

            m.points.append(p)

        return m

    def publish_obstacle(self):
        backend_class_path = str(self.get_parameter("backend_class").value)
        if "cbf" in backend_class_path:
            x, y = np.array(self.get_parameter("cbf.obstacle.center").value)
            r_inner = float(self.get_parameter("cbf.obstacle.radius").value)
            r_outer = r_inner + float(
                self.get_parameter("cbf.obstacle.safety_margin").value
            )

        elif "hj" in backend_class_path:
            x, y = np.array(self.get_parameter("hj.obstacle.center").value)
            r_inner = float(self.get_parameter("hj.obstacle.radius").value)
            r_outer = r_inner + float(
                self.get_parameter("hj.obstacle.safety_margin").value
            )

        else:
            return

        ma = MarkerArray()

        ma.markers.append(self.make_filled_circle(x, y, r_inner))
        ma.markers.append(self.make_circle_outline(x, y, r_outer))

        self.obs_pub.publish(ma)


def main() -> None:
    rclpy.init()
    node = ControllerNode()
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
