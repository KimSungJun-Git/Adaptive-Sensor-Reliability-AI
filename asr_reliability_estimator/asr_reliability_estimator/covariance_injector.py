"""Covariance Injector (plan section 31.1) — the robot_localization
deployment path.

Republishes the user's existing sensor topics with the same payload but a
reliability-scaled covariance, so a stock ekf_filter_node consumes them
unchanged:

    odom0: /wheel/odom/adaptive      # was /wheel/odom

Wheel odometry (nav_msgs/Odometry) and lidar twist
(geometry_msgs/TwistWithCovarianceStamped): twist covariance indices 0 (v)
and 35 (omega). IMU (sensor_msgs/Imu): angular_velocity_covariance index 8.
"""
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from asr_core import params as P
from asr_core.smoothing import inflate
from asr_msgs.msg import ReliabilityArray


class CovarianceInjector(Node):
    def __init__(self):
        super().__init__("asr_covariance_injector")
        self.declare_parameter("wheel_odom_in", "/odom")
        self.declare_parameter("imu_in", "/imu")
        self.declare_parameter("lidar_twist_in", "/asr/lidar_twist")
        self.declare_parameter("alpha", P.ALPHA_INFLATION)
        self.alpha = float(self.get_parameter("alpha").value)
        self.s = {c: 1.0 for c in P.CHANNELS}

        w_in = self.get_parameter("wheel_odom_in").value
        i_in = self.get_parameter("imu_in").value
        l_in = self.get_parameter("lidar_twist_in").value
        self.pub_w = self.create_publisher(Odometry, w_in.rstrip("/") + "/adaptive", 10)
        self.pub_i = self.create_publisher(Imu, i_in.rstrip("/") + "/adaptive",
                                           qos_profile_sensor_data)
        self.pub_l = self.create_publisher(TwistWithCovarianceStamped,
                                           l_in.rstrip("/") + "/adaptive", 10)
        self.create_subscription(ReliabilityArray, "/asr/reliability", self.rel_cb, 10)
        self.create_subscription(Odometry, w_in, self.wheel_cb, 20)
        self.create_subscription(Imu, i_in, self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(TwistWithCovarianceStamped, l_in, self.lidar_cb, 20)

    def rel_cb(self, msg):
        for r in msg.sensors:
            self.s[f"v_{r.sensor_name}"] = r.s_v
            self.s[f"omega_{r.sensor_name}"] = r.s_omega

    def _f(self, channel):
        return float(inflate([1.0], [self.s.get(channel, 1.0)], self.alpha)[0])

    def wheel_cb(self, msg):
        cov = list(msg.twist.covariance)
        cov[0] = max(cov[0], 1e-6) * self._f("v_encoder")
        cov[35] = max(cov[35], 1e-6) * self._f("omega_encoder")
        msg.twist.covariance = cov
        self.pub_w.publish(msg)

    def imu_cb(self, msg):
        cov = list(msg.angular_velocity_covariance)
        cov[8] = max(cov[8], 1e-9) * self._f("omega_imu")
        msg.angular_velocity_covariance = cov
        self.pub_i.publish(msg)

    def lidar_cb(self, msg):
        cov = list(msg.twist.covariance)
        cov[0] = max(cov[0], 1e-6) * self._f("v_lidar")
        cov[35] = max(cov[35], 1e-6) * self._f("omega_lidar")
        msg.twist.covariance = cov
        self.pub_l.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CovarianceInjector())


if __name__ == "__main__":
    main()
