"""Live demo dashboard (plan Phase 11): per-(sensor x DOF) reliability,
trajectories (GT vs Fixed vs Adaptive EKF) and system state text panel.
"""
import math
import threading
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from asr_msgs.msg import FaultStatus, ReliabilityArray

CHAN_LABELS = [("encoder", "s_v"), ("encoder", "s_omega"),
               ("lidar", "s_v"), ("lidar", "s_omega"),
               ("imu", "s_omega")]
COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#17becf", "#2ca02c"]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Dashboard(Node):
    def __init__(self):
        super().__init__("asr_dashboard")
        self.lock = threading.Lock()
        self.rel = deque(maxlen=1200)   # (t, s0..s4, stationary)
        self.trajs = {"gt": [], "fixed": [], "adaptive": []}
        self.origins = {}
        self.faults = {}
        self.create_subscription(ReliabilityArray, "/asr/reliability", self.rel_cb, 10)
        self.create_subscription(Odometry, "/asr/ground_truth",
                                 lambda m: self.odom_cb(m, "gt"), 10)
        self.create_subscription(Odometry, "/asr/odom_fixed",
                                 lambda m: self.odom_cb(m, "fixed"), 10)
        self.create_subscription(Odometry, "/asr/odom_adaptive",
                                 lambda m: self.odom_cb(m, "adaptive"), 10)
        self.create_subscription(FaultStatus, "/asr/fault_status", self.fault_cb, 10)

    def rel_cb(self, msg):
        vals = {}
        for r in msg.sensors:
            vals[(r.sensor_name, "s_v")] = r.s_v
            vals[(r.sensor_name, "s_omega")] = r.s_omega
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.rel.append((t, *[vals.get(c, 1.0) for c in CHAN_LABELS],
                             float(msg.stationary)))

    def odom_cb(self, msg, key):
        p = msg.pose.pose
        x, y, yaw = p.position.x, p.position.y, yaw_of(p.orientation)
        with self.lock:
            if key not in self.origins:
                self.origins[key] = (x, y, yaw)
            x0, y0, a0 = self.origins[key]
            c, s = math.cos(-a0), math.sin(-a0)
            self.trajs[key].append((c * (x - x0) - s * (y - y0),
                                    s * (x - x0) + c * (y - y0)))
            if len(self.trajs[key]) > 6000:
                self.trajs[key] = self.trajs[key][-6000:]

    def fault_cb(self, msg):
        with self.lock:
            self.faults[(msg.sensor_name, msg.fault_type)] = msg.active


def main(args=None):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    rclpy.init(args=args)
    node = Dashboard()
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    plt.ion()
    fig = plt.figure("Adaptive Sensor Reliability", figsize=(12, 6))
    ax_r = fig.add_subplot(1, 2, 1)
    ax_t = fig.add_subplot(1, 2, 2)
    try:
        while rclpy.ok():
            with node.lock:
                rel = np.asarray(node.rel) if node.rel else None
                trajs = {k: np.asarray(v) for k, v in node.trajs.items() if v}
                faults = dict(node.faults)
            ax_r.clear()
            ax_t.clear()
            ax_r.set_title("Reliability (sensor x DOF)")
            ax_r.set_ylim(-0.05, 1.05)
            if rel is not None:
                t = rel[:, 0] - rel[0, 0]
                for i, (sen, dof) in enumerate(CHAN_LABELS):
                    ax_r.plot(t, rel[:, 1 + i], color=COLORS[i],
                              label=f"{sen} {dof}")
                ax_r.legend(loc="lower left", fontsize=8)
                cur = rel[-1, 1:6]
                degraded = (cur < 0.7).any()
                state = "DEGRADED" if degraded else "NORMAL"
                active = [f"{s}:{f}" for (s, f), a in faults.items() if a]
                ax_r.set_xlabel(
                    f"state: {state}   stationary: {bool(rel[-1, 6])}   "
                    f"faults: {', '.join(active) if active else '-'}")
            ax_t.set_title("Trajectory (aligned to first pose)")
            for key, style in (("gt", "k-"), ("fixed", "r--"), ("adaptive", "b-")):
                if key in trajs:
                    ax_t.plot(trajs[key][:, 0], trajs[key][:, 1], style,
                              label=key, linewidth=1.2)
            ax_t.legend(loc="best", fontsize=8)
            ax_t.axis("equal")
            plt.pause(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
