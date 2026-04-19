"""ROS2 planner frontend for LQR backend controllers."""

import csv
import os
import time

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
        self.declare_parameter("lqr.dv_cost", 0.05) # for analyzing smoothing 
        self.declare_parameter("lqr.dw_cost", 0.10) # for analyzing smoothing 
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

        # CSV logging / metrics
        self._log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(self._log_dir, exist_ok=True)

        backend_name = str(self.get_parameter("backend_class").value).split(":")[-1]
        stamp = time.strftime("%Y%m%d_%H%M%S")

        self._log_path = os.path.join(self._log_dir, f"{backend_name}_{stamp}.csv")
        self._summary_path = os.path.join(
            self._log_dir, f"{backend_name}_{stamp}_summary.csv"
        )

        self._log_file = open(self._log_path, "w", newline="")
        self._csv_writer = csv.writer(self._log_file)
        self._csv_writer.writerow(
            [
                "step",
                "t_sec",
                "x",
                "y",
                "yaw",
                "ref_x",
                "ref_y",
                "ref_yaw",
                "err_x",
                "err_y",
                "err_yaw",
                "tracking_error_sq",
                "tracking_deviation",
                "v_cmd",
                "w_cmd",
                "control_effort",
                "solve_time_ms",
                "stage_cost",
                "cumulative_cost",
            ]
        )

        self._t0 = time.time()
        self._step_idx = 0
        self._num_logged_steps = 0

        self._sum_sq_tracking_error = 0.0   # for MSE tracking error
        self._max_tracking_deviation = 0.0  # max Euclidean position deviation
        self._sum_sq_control = 0.0          # control effort
        self._cumulative_cost = 0.0
        self._total_solve_time = 0.0        # runtime per trajectory
        self._prev_u_cmd = None

        # Cost weights used for stage-cost logging
        self._x_cost = float(self.get_parameter("lqr.x_cost").value)
        self._y_cost = float(self.get_parameter("lqr.y_cost").value)
        self._theta_cost = float(self.get_parameter("lqr.theta_cost").value)
        self._v_cost = float(self.get_parameter("lqr.v_cost").value)
        self._w_cost = float(self.get_parameter("lqr.w_cost").value)
        self._dv_cost = float(self.get_parameter("lqr.dv_cost").value)
        self._dw_cost = float(self.get_parameter("lqr.dw_cost").value)


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
        _, _, yaw = euler_from_quaternion(
            float(q.x), float(q.y), float(q.z), float(q.w)
        )

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
        # Changed so LQR can use its built - in reference generator
        if self._latest_state is None:
            return

        u = np.array([0.0, 0.0], dtype=float)
        Z = None
        U = None

        solve_start = time.perf_counter() # record time used for metrics logging
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

        solve_time_s = time.perf_counter() - solve_start
        
        cmd = Twist()
        cmd.linear.x = float(u[0])
        cmd.angular.z = float(u[1])
        self._cmd_pub.publish(cmd)

        self._log_step(u, solve_time_s)


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
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _get_reference_state_for_logging(self):
        """
        Reference state used to compute tracking error.
        If the backend has a step counter, align with that step.
        Otherwise fall back to the first reference state.
        """
        if self._latest_traj is None:
            return None

        states = np.asarray(self._latest_traj.states, dtype=float)
        if states.size == 0:
            return None

        idx = 0
        if hasattr(self._backend, "_step"):
            try:
                idx = int(getattr(self._backend, "_step"))
            except Exception:
                idx = 0

        idx = max(0, min(idx, states.shape[0] - 1))
        return states[idx].reshape(3)

    def _log_step(self, u: NDArray, solve_time_s: float) -> None:
        if self._latest_state is None:
            return

        ref_state = self._get_reference_state_for_logging()
        if ref_state is None:
            ref_state = np.asarray(self._latest_state, dtype=float).reshape(3)

        state = np.asarray(self._latest_state, dtype=float).reshape(3)
        u = np.asarray(u, dtype=float).reshape(2)

        err = state - ref_state
        err[2] = self._wrap_angle(err[2])

        tracking_error_sq = float(err[0] ** 2 + err[1] ** 2 + err[2] ** 2)
        tracking_deviation = float(np.hypot(err[0], err[1]))
        control_effort = float(u[0] ** 2 + u[1] ** 2)

        if self._prev_u_cmd is None:
            du = np.zeros(2, dtype=float)
        else:
            du = u - self._prev_u_cmd

        stage_cost = float(
            self._x_cost * err[0] ** 2
            + self._y_cost * err[1] ** 2
            + self._theta_cost * err[2] ** 2
            + self._v_cost * u[0] ** 2
            + self._w_cost * u[1] ** 2
            + self._dv_cost * du[0] ** 2
            + self._dw_cost * du[1] ** 2
        )

        self._cumulative_cost += stage_cost
        self._sum_sq_tracking_error += tracking_error_sq
        self._max_tracking_deviation = max(
            self._max_tracking_deviation, tracking_deviation
        )
        self._sum_sq_control += control_effort
        self._total_solve_time += solve_time_s
        self._num_logged_steps += 1

        t_now = time.time() - self._t0

        self._csv_writer.writerow(
            [
                self._step_idx,
                float(t_now),
                float(state[0]),
                float(state[1]),
                float(state[2]),
                float(ref_state[0]),
                float(ref_state[1]),
                float(ref_state[2]),
                float(err[0]),
                float(err[1]),
                float(err[2]),
                tracking_error_sq,
                tracking_deviation,
                float(u[0]),
                float(u[1]),
                control_effort,
                float(solve_time_s * 1000.0),
                stage_cost,
                float(self._cumulative_cost),
            ]
        )
        self._log_file.flush()

        self._prev_u_cmd = u.copy()
        self._step_idx += 1

    def _write_summary(self) -> None:
        if self._num_logged_steps == 0:
            return

        mse_tracking_error = self._sum_sq_tracking_error / self._num_logged_steps

        with open(self._summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mean_squared_tracking_error", mse_tracking_error])
            writer.writerow(["maximum_tracking_deviation", self._max_tracking_deviation])
            writer.writerow(["control_effort", self._sum_sq_control])
            writer.writerow(["runtime_per_trajectory_sec", self._total_solve_time])
            writer.writerow(
                [
                    "avg_solver_runtime_ms",
                    (self._total_solve_time / self._num_logged_steps) * 1000.0,
                ]
            )
            writer.writerow(["num_logged_steps", self._num_logged_steps])
            writer.writerow(["final_cumulative_cost", self._cumulative_cost])

    def destroy_node(self):
        try:
            self._write_summary()
        finally:
            try:
                if hasattr(self, "_log_file") and self._log_file and not self._log_file.closed:
                    self._log_file.close()
            except Exception:
                pass
            super().destroy_node()


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
