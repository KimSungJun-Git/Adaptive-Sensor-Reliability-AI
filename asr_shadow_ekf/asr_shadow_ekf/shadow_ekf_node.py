"""Shadow EKF (plan section 8): fixed nominal R, exists only to produce
feedback-free Innovation / NIS for the Reliability AI. Never consumes
reliability, never feeds the Adaptive EKF.
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
from asr_msgs.msg import Innovation

DOF_NAMES = {"encoder": ["v", "omega"], "lidar": ["v", "omega"], "imu": ["omega"]}


class ShadowEkfNode(Node):
    def __init__(self):
        super().__init__("asr_shadow_ekf")
        self.declare_parameter("encoder_topic", "/asr/encoder_twist")
        self.declare_parameter("lidar_topic", "/asr/lidar_twist")
        self.declare_parameter("imu_topic", "/asr/imu")
        self.declare_parameter("imu_subsample", 4)  # 200 Hz -> 50 Hz
        self.ekf = EKF(P.EKF_Q_DIAG, P.EKF_P0_DIAG)
        self.r0 = {s: np.diag([P.NOMINAL_STD[c] ** 2 for c in P.SENSOR_CHANNELS[s]])
                   for s in P.SENSORS}
        self.imu_n = 0
        self.imu_sub_n = int(self.get_parameter("imu_subsample").value)
        self.pub_innov = self.create_publisher(Innovation, "/asr/shadow/innovation", 50)
        self.pub_odom = self.create_publisher(Odometry, "/asr/shadow/odom", 10)
        self.create_subscription(
            TwistWithCovarianceStamped, self.get_parameter("encoder_topic").value,
            lambda m: self.meas(m, "encoder"), 20)
        self.create_subscription(
            TwistWithCovarianceStamped, self.get_parameter("lidar_topic").value,
            lambda m: self.meas(m, "lidar"), 20)
        self.create_subscription(
            Imu, self.get_parameter("imu_topic").value, self.imu_cb,
            qos_profile_sensor_data)

    def imu_cb(self, msg):
        self.imu_n += 1
        if self.imu_n % self.imu_sub_n:
            return
        self.meas(msg, "imu")

    def meas(self, msg, sensor):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if sensor == "imu":
            z = [msg.angular_velocity.z]
        else:
            z = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        # ponytail: measurements are processed in arrival order; cross-sensor
        # jitter of a few ms is absorbed by predict()'s non-negative dt guard.
        self.ekf.predict(t)
        nu, S, nis = self.ekf.update(z, P.SENSOR_STATE_IDX[sensor], self.r0[sensor])
        out = Innovation()
        out.header = msg.header
        out.sensor_name = sensor
        out.dof = DOF_NAMES[sensor]
        out.innovation = [float(x) for x in nu]
        out.innovation_std = [float(x) for x in np.sqrt(np.diag(S))]
        out.nis = float(nis)
        self.pub_innov.publish(out)
        if sensor == "encoder":
            o = Odometry()
            o.header = msg.header
            o.header.frame_id = "odom_shadow"
            x = self.ekf.x
            o.pose.pose.position.x, o.pose.pose.position.y = x[0], x[1]
            o.pose.pose.orientation.z = float(np.sin(x[2] / 2))
            o.pose.pose.orientation.w = float(np.cos(x[2] / 2))
            o.twist.twist.linear.x, o.twist.twist.angular.z = x[3], x[4]
            self.pub_odom.publish(o)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ShadowEkfNode())


if __name__ == "__main__":
    main()
