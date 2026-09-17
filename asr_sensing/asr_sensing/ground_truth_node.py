"""Gazebo ground truth -> nav_msgs/Odometry with body-frame twist.

Label generation and evaluation only — never an input to the estimators
(plan section 12.1). Requires the gazebo_ros_state world plugin.
"""
import math

import rclpy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from rclpy.node import Node


class GroundTruthNode(Node):
    def __init__(self):
        super().__init__("asr_ground_truth")
        self.declare_parameter("model_name", "waffle_pi")
        self.declare_parameter("output_topic", "/asr/ground_truth")
        self.model = self.get_parameter("model_name").value
        self.pub = self.create_publisher(
            Odometry, self.get_parameter("output_topic").value, 10)
        self.sub = self.create_subscription(
            ModelStates, "/gazebo/model_states", self.cb, 10)
        self.warned = False

    def cb(self, msg):
        try:
            i = msg.name.index(self.model)
        except ValueError:
            cands = [n for n in msg.name
                     if n not in ("ground_plane",) and "world" not in n]
            if not cands:
                if not self.warned:
                    self.get_logger().warn(f"model '{self.model}' not in {msg.name}")
                    self.warned = True
                return
            i = msg.name.index(cands[-1])
        pose, twist = msg.pose[i], msg.twist[i]
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # world-frame linear velocity -> body frame
        vx = math.cos(yaw) * twist.linear.x + math.sin(yaw) * twist.linear.y
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "odom_gt"
        out.child_frame_id = "base_footprint_gt"
        out.pose.pose = pose
        out.twist.twist.linear.x = vx
        out.twist.twist.angular.z = twist.angular.z
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GroundTruthNode())


if __name__ == "__main__":
    main()
