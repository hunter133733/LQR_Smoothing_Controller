"""
figure8_publisher.py
====================
Publishes a figure-8 reference trajectory on /traj.

Trajectory sequence:
  1. Straight approach from spawn (-4, 3.5) to crossing point (0, 0)
  2. Left lobe  (+w, CCW) — into +y territory
  3. Right lobe (-w, CW)  — into -y territory
  4. Returns to (0, 0) — goal check fires and sim stops

Coordinate frame:
  +x = forward from robot start
  +y = left
  -y = right
  Robot spawns at (-4, 3.5) in map frame, heading east (theta=0)

Wall clearances with centre=(0,0) radius=1.8m:
  X: [-1.8, 1.8]  clearance ±3.2m
  Y: [-3.6, 3.6]  clearance ±1.4m
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from nav_helpers_msgs.msg import StateActionTrajectory as TrajMsg
from nav_helpers.trajectory import StateActionTrajectory

# Trajectory builder
def make_figure8(
    cx: float = 0.0,
    cy: float = 0.0,
    radius: float = 1.8,
    v: float = 0.3,
    dt: float = 0.1,
    n_laps: int = 1,
    start_x: float = -4.0,
    start_y: float = 3.5,
) -> StateActionTrajectory:

    # 1: Straight approach from spawn to crossing point
    dx   = cx - start_x
    dy   = cy - start_y
    dist = math.hypot(dx, dy)
    heading_approach = math.atan2(dy, dx)
    approach_steps   = max(1, int(round(dist / (v * dt))))

    app_states  = []
    app_actions = []

    for k in range(approach_steps):
        frac = k / approach_steps
        app_states.append([
            start_x + frac * dx,
            start_y + frac * dy,
            heading_approach,
        ])
        app_actions.append([v, 0.0])

    # Final approach state = crossing point, heading east ready for figure-8
    app_states.append([cx, cy, 0.0])

    # 2: Figure-8 loops
    w_mag            = v / radius
    steps_per_circle = int(round((2.0 * math.pi / w_mag) / dt))
    steps_per_fig8   = 2 * steps_per_circle
    n_loop_actions   = n_laps * steps_per_fig8

    loop_states  = []
    loop_actions = []
    x, y, th = cx, cy, 0.0

    for k in range(n_loop_actions):
        pos = k % steps_per_fig8
        w_k = +w_mag if pos < steps_per_circle else -w_mag
        loop_actions.append([v, w_k])
        x_new  = x  + v * math.cos(th) * dt
        y_new  = y  + v * math.sin(th) * dt
        th_new = th + w_k * dt
        loop_states.append([x_new, y_new, th_new])
        x, y, th = x_new, y_new, th_new

    # 3: Combine approach and loop into one trajectory
    all_states  = np.array(app_states  + loop_states,  dtype=float)
    all_actions = np.array(app_actions + loop_actions, dtype=float)

    # For testing
    assert all_states.shape[0] == all_actions.shape[0] + 1, (
        f"Shape mismatch: states {all_states.shape[0]} != "
        f"actions {all_actions.shape[0]} + 1"
    )

    return StateActionTrajectory(states=all_states, actions=all_actions, dt=dt)


# ROS2 publisher node
class Figure8Publisher(Node):
    def __init__(self):
        super().__init__("figure8_publisher")

        # Parameters matching the /figure8_publisher section of the YAML
        self.declare_parameter("nom_traj_topic", "/traj")
        self.declare_parameter("dt",              0.1)
        self.declare_parameter("fig8.center_x",   0.0)
        self.declare_parameter("fig8.center_y",   0.0)
        self.declare_parameter("fig8.radius",      1.8)
        self.declare_parameter("fig8.speed",       0.3)
        self.declare_parameter("fig8.n_laps",      1)
        # Spawn position in map frame
        self.declare_parameter("fig8.start_x",   -4.0)
        self.declare_parameter("fig8.start_y",    3.5)

        # Read parameter values
        traj_topic = str(self.get_parameter("nom_traj_topic").value)
        dt         = float(self.get_parameter("dt").value)
        cx         = float(self.get_parameter("fig8.center_x").value)
        cy         = float(self.get_parameter("fig8.center_y").value)
        radius     = float(self.get_parameter("fig8.radius").value)
        speed      = float(self.get_parameter("fig8.speed").value)
        n_laps     = int(self.get_parameter("fig8.n_laps").value)
        start_x    = float(self.get_parameter("fig8.start_x").value)
        start_y    = float(self.get_parameter("fig8.start_y").value)

        # Build trajectory
        self._traj = make_figure8(
            cx=cx, cy=cy, radius=radius,
            v=speed, dt=dt, n_laps=n_laps,
            start_x=start_x, start_y=start_y,
        )

        # Logs
        n_states = self._traj.states.shape[0]
        duration = (n_states - 1) * dt
        self.get_logger().info(
            f"Figure-8 ready: "
            f"approach ({start_x},{start_y})->({cx},{cy})  "
            f"r={radius}m  v={speed}m/s  w={speed/radius:.3f}rad/s  "
            f"{n_laps} lap(s)  {n_states} states  ~{duration:.1f}s"
        )
        self.get_logger().info(
            f"Loop extents: "
            f"X=[{cx-radius:.2f},{cx+radius:.2f}]  "
            f"Y=[{cy-2*radius:.2f},{cy+2*radius:.2f}]  "
            f"wall clearances: X+-{5-radius:.1f}m  Y+-{5-2*radius:.1f}m"
        )

        # Latched so controller receives it even if it starts after this node
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(TrajMsg, traj_topic, qos)
        self._publish()
        self._timer = self.create_timer(1.0, self._publish)

    def _publish(self):
        msg = self._traj.to_msg()
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)

# ROS2 node entry point
def main(args=None):
    rclpy.init(args=args)
    node = Figure8Publisher()
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