"""LiDAR odometry: consecutive 2D ICP scan matching -> body twist.

Initial guess comes from the matcher's own constant-velocity model, never
from the Adaptive EKF (plan section 8.4).
"""
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from asr_core import params as P
from asr_core.icp import LidarOdometry
from asr_msgs.msg import LidarQuality


class LidarOdomNode(Node):
    def __init__(self):
        super().__init__("asr_lidar_odom")
        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("twist_topic", "/asr/lidar_twist")
        self.declare_parameter("quality_topic", "/asr/lidar_quality")
        self.declare_parameter("range_min", 0.12)
        self.declare_parameter("range_max", 3.4)
        self.lo = LidarOdometry(self.get_parameter("range_min").value,
                                self.get_parameter("range_max").value)
        self.pub_t = self.create_publisher(
            TwistWithCovarianceStamped, self.get_parameter("twist_topic").value, 10)
        self.pub_q = self.create_publisher(
            LidarQuality, self.get_parameter("quality_topic").value, 10)
        self.sub = self.create_subscription(
            LaserScan, self.get_parameter("input_topic").value, self.cb,
            qos_profile_sensor_data)

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = self.lo.step(t, list(msg.ranges), msg.angle_min, msg.angle_increment)
        if out is None:
            return
        q = LidarQuality()
        q.header = msg.header
        q.rms, q.match_ratio, q.valid = out["rms"], out["ratio"], out["valid"]
        q.degeneracy = out["degen"]
        self.pub_q.publish(q)
        if not out["valid"]:
            return
        tw = TwistWithCovarianceStamped()
        tw.header = msg.header
        tw.twist.twist.linear.x = out["v"]
        tw.twist.twist.angular.z = out["omega"]
        tw.twist.covariance[0] = P.NOMINAL_STD["v_lidar"] ** 2
        tw.twist.covariance[35] = P.NOMINAL_STD["omega_lidar"] ** 2
        self.pub_t.publish(tw)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LidarOdomNode())


if __name__ == "__main__":
    main()
