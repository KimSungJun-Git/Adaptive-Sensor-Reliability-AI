"""Adaptive EKF (plan section 16): same filter as the Shadow EKF but with
measurement covariance scaled by the AI reliability (bounded inflation,
plan section 15) and per-DOF rejection below s_reject.

mode:=fixed ignores reliability entirely -> the Fixed EKF comparison group.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from asr_core import params as P
from asr_core.ekf import EKF
from asr_core.smoothing import inflate
from asr_msgs.msg import ReliabilityArray


class AdaptiveEkfNode(Node):
    def __init__(self):
        super().__init__("asr_adaptive_ekf")
        self.declare_parameter("mode", "adaptive")     # adaptive | fixed
        self.declare_parameter("output_topic", "/asr/odom_adaptive")
        self.declare_parameter("alpha", P.ALPHA_INFLATION)
        self.declare_parameter("s_reject", P.S_REJECT)
        self.declare_parameter("imu_subsample", 4)
        self.mode = self.get_parameter("mode").value
        self.alpha = float(self.get_parameter("alpha").value)
        self.s_reject = float(self.get_parameter("s_reject").value)
        self.ekf = EKF(P.EKF_Q_DIAG, P.EKF_P0_DIAG)
        self.r0 = {s: np.array([P.NOMINAL_STD[c] ** 2 for c in P.SENSOR_CHANNELS[s]])
                   for s in P.SENSORS}
        self.s = np.ones(len(P.CHANNELS))
        self.imu_n = 0
        self.imu_sub_n = int(self.get_parameter("imu_subsample").value)
        self.pub = self.create_publisher(
            Odometry, self.get_parameter("output_topic").value, 10)
        self.create_subscription(ReliabilityArray, "/asr/reliability", self.rel_cb, 10)
        self.create_subscription(TwistWithCovarianceStamped, "/asr/encoder_twist",
                                 lambda m: self.meas(m, "encoder"), 20)
        self.create_subscription(TwistWithCovarianceStamped, "/asr/lidar_twist",
                                 lambda m: self.meas(m, "lidar"), 20)
        self.create_subscription(Imu, "/asr/imu", self.imu_cb, qos_profile_sensor_data)

    def rel_cb(self, msg):
        vals = {}
        for r in msg.sensors:
            vals[f"v_{r.sensor_name}"] = r.s_v
            vals[f"omega_{r.sensor_name}"] = r.s_omega
        self.s = np.array([vals.get(c, 1.0) for c in P.CHANNELS])

    def imu_cb(self, msg):
        self.imu_n += 1
        if self.imu_n % self.imu_sub_n:
            return
        self.meas(msg, "imu")

    def meas(self, msg, sensor):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if sensor == "imu":
            z = np.array([msg.angular_velocity.z])
        else:
            z = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        idx = P.SENSOR_STATE_IDX[sensor]
        if self.mode == "fixed":
            rd = self.r0[sensor]
            keep = np.ones(len(idx), bool)
        else:
            s = np.array([self.s[P.CHANNELS.index(c)]
                          for c in P.SENSOR_CHANNELS[sensor]])
            rd = inflate(self.r0[sensor], s, self.alpha)
            keep = s > self.s_reject  # measurement rejection (plan 15.3)
        self.ekf.predict(t)
        if keep.any():
            sel = np.flatnonzero(keep)
            self.ekf.update(z[sel], [idx[i] for i in sel], np.diag(rd[sel]))
        if sensor == "encoder":
            o = Odometry()
            o.header = msg.header
            o.header.frame_id = "odom_asr"
            o.child_frame_id = "base_footprint"
            x = self.ekf.x
            o.pose.pose.position.x, o.pose.pose.position.y = x[0], x[1]
            o.pose.pose.orientation.z = float(np.sin(x[2] / 2))
            o.pose.pose.orientation.w = float(np.cos(x[2] / 2))
            o.twist.twist.linear.x, o.twist.twist.angular.z = x[3], x[4]
            o.pose.covariance[0] = self.ekf.P[0, 0]
            o.pose.covariance[7] = self.ekf.P[1, 1]
            o.pose.covariance[35] = self.ekf.P[2, 2]
            o.twist.covariance[0] = self.ekf.P[3, 3]
            o.twist.covariance[35] = self.ekf.P[4, 4]
            self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(AdaptiveEkfNode())


if __name__ == "__main__":
    main()
