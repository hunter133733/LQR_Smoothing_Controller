import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from nav_helpers_msgs.msg import StateActionPoint as PointMsg
from nav_helpers_msgs.msg import StateActionTrajectory as TrajMsg


def euler_from_quaternion(
    x: float, y: float, z: float, w: float
) -> Tuple[float, float, float]:
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - x * z)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ "
    Args: roll, pitch yaw
    Returns: Quaternion [w, x, y, z]
    """
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q


def yaw_to_quat(yaw):
    return quaternion_from_euler(0, 0, yaw)


def quat_to_yaw(q):
    return euler_from_quaternion(q.x, q.y, q.z, q.w)[2]


def float_to_duration(t):
    return Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))


def duration_to_float(d):
    return d.sec + d.nanosec * 1e-9


@dataclass
class StateActionTrajectory:
    states: np.ndarray  # (N+1, 3)
    actions: np.ndarray  # (N, 2)
    dt: float
    frame_id: str = "map"

    def __post_init__(self):
        assert self.states.shape[0] == self.actions.shape[0] + 1

    def to_msg(self, clock=None):
        msg = TrajMsg()
        msg.header.frame_id = self.frame_id

        # ----------------------------
        # Handle clock safely
        # ----------------------------
        if clock is not None:
            msg.header.stamp = clock.now().to_msg()
        else:
            # leave default stamp (0) or explicitly set zero time
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0

        N = self.actions.shape[0]

        for i in range(N + 1):
            pt = PointMsg()

            x, y, yaw = self.states[i]

            pt.pose.position.x = float(x)
            pt.pose.position.y = float(y)
            pt.pose.position.z = 0.0

            q = yaw_to_quat(yaw)
            pt.pose.orientation.w = q[0]
            pt.pose.orientation.x = q[1]
            pt.pose.orientation.y = q[2]
            pt.pose.orientation.z = q[3]

            if i < N:
                v, w = self.actions[i]
                pt.twist.linear.x = float(v)
                pt.twist.angular.z = float(w)
            else:
                pt.twist.linear.x = 0.0
                pt.twist.angular.z = 0.0

            pt.time_from_start = float_to_duration(i * self.dt)

            msg.points.append(pt)

        return msg

    @classmethod
    def from_msg(cls, msg: TrajMsg):
        states, actions, times = [], [], []

        for i, pt in enumerate(msg.points):
            p = pt.pose
            states.append([p.position.x, p.position.y, quat_to_yaw(p.orientation)])

            times.append(duration_to_float(pt.time_from_start))

            if i < len(msg.points) - 1:
                actions.append([pt.twist.linear.x, pt.twist.angular.z])

        states = np.array(states)
        actions = np.array(actions)
        dt = times[1] - times[0] if len(times) > 1 else 0.0

        return cls(states, actions, dt, msg.header.frame_id)

    def to_path(self, clock=None):
        path = Path()
        path.header.frame_id = self.frame_id

        for x, y, yaw in self.states:
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id

            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)

            q = yaw_to_quat(yaw)
            ps.pose.orientation.w = q[0]
            ps.pose.orientation.x = q[1]
            ps.pose.orientation.y = q[2]
            ps.pose.orientation.z = q[3]

            path.poses.append(ps)

        return path
