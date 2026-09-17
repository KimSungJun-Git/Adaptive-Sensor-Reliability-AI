"""Online fault injector for the live demo (plan section 13).

Sits between Gazebo and the preprocessing nodes:
    /joint_states -> /asr/joint_states,  /imu -> /asr/imu,  /scan -> /asr/scan
and applies the scenario's faults inside their time window, publishing
ground-truth fault activity on /asr/fault_status.

Offline dataset generation uses the vectorized asr_core.faults instead; this
node shares the same formulas per message.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState, LaserScan

from asr_core.faults import scenario_events
from asr_core.params import WHEEL_RADIUS, WHEEL_SEPARATION
from asr_msgs.msg import FaultStatus


def wheel_fault_velocity(v_in, side, e, t, rng):
    """Faulted angular velocity [rad/s] for one wheel given one active
    encoder FaultEvent. `side`: +1.0 for the right wheel, -1.0 for the left.

    asr_core.faults.apply_encoder works on the already-computed v/omega
    scalars, so a channel="omega" bias just adds to omega directly. Here we
    only have raw per-wheel angular velocities (v=(vl+vr)/2*r,
    omega=(vr-vl)/b*r), so the same edit must be applied differently per
    channel: a "v" fault is symmetric (both wheels shift together, which
    moves the average but cancels in the difference); an "omega" fault must
    be antisymmetric (opposite sign per side, which moves the difference
    but cancels in the average)."""
    if e.channel == "omega":
        scale = side * WHEEL_SEPARATION / (2.0 * WHEEL_RADIUS)
    else:
        scale = 1.0 / WHEEL_RADIUS
    if e.ftype in ("slip", "scale") and e.channel != "omega":
        return v_in * (1.0 + e.mag)
    if e.ftype == "bias":
        return v_in + e.mag * scale
    if e.ftype == "drift":
        return v_in + e.mag * (t - e.t0) * scale
    if e.ftype == "noise":
        return v_in + rng.normal(0.0, e.mag * scale)
    return v_in


def imu_fault_step(w, ax, events, t, frozen, rng):
    """Faulted (angular_velocity.z, linear_acceleration.x) for one IMU sample
    given its currently-active fault events. `frozen`: the (w, ax) captured
    at dropout onset, or None; pass the returned value back in next call."""
    if any(e.ftype == "dropout" for e in events):
        if frozen is None:
            frozen = (w, ax)
        return frozen[0], frozen[1], frozen
    for e in events:
        if e.ftype == "bias":
            w += e.mag
        elif e.ftype == "bias_drift":
            w += e.mag * (t - e.t0)
        elif e.ftype == "noise":
            w += rng.normal(0.0, e.mag)
    return w, ax, None


class FaultInjectorNode(Node):
    def __init__(self):
        super().__init__("asr_fault_injector")
        self.declare_parameter("scenario", "slip")
        self.declare_parameter("t_end", 90.0)  # scenario timeline length
        self.declare_parameter("seed", 0)
        self.events = scenario_events(self.get_parameter("scenario").value,
                                      float(self.get_parameter("t_end").value),
                                      int(self.get_parameter("seed").value))
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.t0 = None
        self.joint_off = {}   # joint name -> accumulated position offset
        self.prev_js_t = None
        self.imu_frozen = None   # (w, ax) captured at dropout onset, else None
        self.pub_js = self.create_publisher(JointState, "/asr/joint_states", 10)
        self.pub_imu = self.create_publisher(Imu, "/asr/imu", qos_profile_sensor_data)
        self.pub_scan = self.create_publisher(LaserScan, "/asr/scan", qos_profile_sensor_data)
        self.pub_fault = self.create_publisher(FaultStatus, "/asr/fault_status", 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Imu, "/imu", self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, qos_profile_sensor_data)
        self.create_timer(0.5, self.status_tick)

    def rel_t(self, stamp):
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def active(self, sensor, t):
        return [e for e in self.events if e.sensor == sensor and e.t0 <= t < e.t1]

    # --- encoder (wheel level: velocity scaled, position kept consistent) ---
    def js_cb(self, msg):
        t = self.rel_t(msg.header.stamp)
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        dt = 0.0
        abs_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_js_t is not None:
            dt = max(abs_t - self.prev_js_t, 0.0)
        self.prev_js_t = abs_t
        for e in self.active("encoder", t):
            for i, name in enumerate(out.name):
                if "wheel" not in name or i >= len(out.velocity):
                    continue
                side = 1.0 if "right" in name else -1.0 if "left" in name else 0.0
                v_in = out.velocity[i]
                v_out = wheel_fault_velocity(v_in, side, e, t, self.rng)
                out.velocity[i] = v_out
                off = self.joint_off.get(name, 0.0) + (v_out - v_in) * dt
                self.joint_off[name] = off
                if i < len(out.position):
                    out.position[i] += off
        self.pub_js.publish(out)

    def imu_cb(self, msg):
        t = self.rel_t(msg.header.stamp)
        w, ax, self.imu_frozen = imu_fault_step(
            msg.angular_velocity.z, msg.linear_acceleration.x,
            self.active("imu", t), t, self.imu_frozen, self.rng)
        msg.angular_velocity.z = w
        msg.linear_acceleration.x = ax
        self.pub_imu.publish(msg)

    def scan_cb(self, msg):
        t = self.rel_t(msg.header.stamp)
        r = np.asarray(msg.ranges, dtype=np.float32)
        drop = False
        for e in self.active("lidar", t):
            if e.ftype == "dropout":
                drop = drop or (self.rng.random() < e.mag)
            elif e.ftype == "range_noise":
                r = r + self.rng.normal(0.0, e.mag, r.size).astype(np.float32)
            elif e.ftype == "sector":
                n = r.size
                w = int(n * e.mag)
                start = int(self.rng.integers(0, n))
                idx = (start + np.arange(w)) % n
                r[idx] = np.inf
        if drop:
            return
        msg.ranges = r.tolist()
        self.pub_scan.publish(msg)

    def status_tick(self):
        if self.t0 is None:
            return
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9 - self.t0
        for e in self.events:
            m = FaultStatus()
            m.header.stamp = now.to_msg()
            m.sensor_name = e.sensor
            m.channel = "scan" if e.sensor == "lidar" else e.channel
            m.fault_type = e.ftype
            m.active = bool(e.t0 <= t < e.t1)
            m.magnitude = float(e.mag)
            self.pub_fault.publish(m)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FaultInjectorNode())


if __name__ == "__main__":
    main()
