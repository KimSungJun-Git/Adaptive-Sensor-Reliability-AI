"""Scripted random driver for data collection: straights, arcs, spins and
stops (stationary scenario), with simple scan-based obstacle avoidance."""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from asr_core.params import MAX_ANG_VEL, MAX_LIN_VEL


def avoid_cmd(front, left, t, avoid_since, avoid_dir, back_after=3.0):
    """Obstacle-avoidance command while front < threshold. Pure rotation can
    deadlock in a concave corner (regression: a demo spawn point wedged
    against a wall spun in place for 80s straight) -- back up instead once
    rotation alone hasn't cleared it in `back_after` seconds.

    Returns (linear_x, angular_z, avoid_since', avoid_dir')."""
    if avoid_since is None:
        avoid_since = t
    if t - avoid_since > back_after:
        return -0.08, 0.0, avoid_since, avoid_dir
    avoid_dir = -1.0 if left < 0.6 else 1.0
    return 0.0, 0.9 * avoid_dir, avoid_since, avoid_dir


class DriverNode(Node):
    def __init__(self):
        super().__init__("asr_driver")
        self.declare_parameter("seed", 0)
        self.declare_parameter("duration", 90.0)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.duration = float(self.get_parameter("duration").value)
        self.pub = self.create_publisher(Twist, self.get_parameter("cmd_topic").value, 10)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value,
                                 self.scan_cb, qos_profile_sensor_data)
        self.front = 10.0
        self.left = 10.0
        self.got_scan = False
        self.t0 = None
        self.seg_end = 0.0
        self.cmd = (0.0, 0.0)
        self.avoid_dir = 1.0
        self.avoid_since = None  # set on entering avoidance; None once clear
        self.timer = self.create_timer(0.1, self.tick)

    def scan_cb(self, msg):
        r = np.asarray(msg.ranges)
        r = np.where(np.isfinite(r) & (r > msg.range_min), r, 10.0)
        n = r.size
        k = max(1, n // 12)  # +-30 deg front sector
        self.front = float(np.min(np.concatenate([r[:k], r[-k:]])))
        self.left = float(np.min(r[k:3 * k]))
        self.got_scan = True

    def now(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def tick(self):
        if not self.got_scan:  # robot not spawned / lidar not up yet
            return
        t = self.now()
        if t > self.duration:
            self.pub.publish(Twist())
            if t > self.duration + 2.0:
                raise SystemExit
            return
        if self.front < 0.45:  # obstacle: rotate away (or back out if stuck)
            lin, ang, self.avoid_since, self.avoid_dir = avoid_cmd(
                self.front, self.left, t, self.avoid_since, self.avoid_dir)
            m = Twist()
            m.linear.x, m.angular.z = lin, ang
            self.pub.publish(m)
            self.seg_end = min(self.seg_end, t)  # re-plan after clearing
            return
        self.avoid_since = None
        if t >= self.seg_end:
            kind = self.rng.choice(["straight", "arc", "spin", "stop"],
                                   p=[0.40, 0.35, 0.13, 0.12])
            if kind == "straight":
                self.cmd = (self.rng.uniform(0.10, MAX_LIN_VEL), 0.0)
            elif kind == "arc":
                self.cmd = (self.rng.uniform(0.08, 0.22),
                            self.rng.uniform(-0.8, 0.8))
            elif kind == "spin":
                self.cmd = (0.0, self.rng.uniform(0.4, 1.2) * self.rng.choice([-1, 1]))
            else:
                self.cmd = (0.0, 0.0)
            self.seg_end = t + self.rng.uniform(2.0, 6.0)
        m = Twist()
        m.linear.x = float(np.clip(self.cmd[0], 0.0, MAX_LIN_VEL))
        m.angular.z = float(np.clip(self.cmd[1], -MAX_ANG_VEL, MAX_ANG_VEL))
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(DriverNode())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
