"""Wheel encoder preprocessing: /joint_states -> body twist (v, omega).

Uses reported wheel velocities when present, otherwise differentiates wheel
positions (what a real encoder does).
"""
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from asr_core import params as P


class EncoderOdomNode(Node):
    def __init__(self):
        super().__init__("asr_encoder_odom")
        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/asr/encoder_twist")
        self.declare_parameter("wheel_radius", P.WHEEL_RADIUS)
        self.declare_parameter("wheel_separation", P.WHEEL_SEPARATION)
        self.declare_parameter("left_joint", "wheel_left_joint")
        self.declare_parameter("right_joint", "wheel_right_joint")
        self.r = self.get_parameter("wheel_radius").value
        self.b = self.get_parameter("wheel_separation").value
        self.lj = self.get_parameter("left_joint").value
        self.rj = self.get_parameter("right_joint").value
        self.prev = None  # (t, pos_l, pos_r)
        self.pub = self.create_publisher(
            TwistWithCovarianceStamped,
            self.get_parameter("output_topic").value, 10)
        self.sub = self.create_subscription(
            JointState, self.get_parameter("input_topic").value, self.cb, 10)

    def cb(self, msg):
        try:
            il, ir = msg.name.index(self.lj), msg.name.index(self.rj)
        except ValueError:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        wl = wr = None
        if len(msg.velocity) > max(il, ir):
            wl, wr = msg.velocity[il], msg.velocity[ir]
        if (wl is None or (wl == 0.0 and wr == 0.0)) and len(msg.position) > max(il, ir):
            if self.prev is not None and t > self.prev[0]:
                dt = t - self.prev[0]
                wl_d = (msg.position[il] - self.prev[1]) / dt
                wr_d = (msg.position[ir] - self.prev[2]) / dt
                if wl is None or abs(wl_d) + abs(wr_d) > 0.0:
                    # prefer differentiation only when velocity field is absent
                    if wl is None:
                        wl, wr = wl_d, wr_d
            self.prev = (t, msg.position[il], msg.position[ir])
        if wl is None:
            return
        vl, vr = wl * self.r, wr * self.r
        v = (vr + vl) / 2.0
        w = (vr - vl) / self.b
        out = TwistWithCovarianceStamped()
        out.header = msg.header
        out.twist.twist.linear.x = float(v)
        out.twist.twist.angular.z = float(w)
        out.twist.covariance[0] = P.NOMINAL_STD["v_encoder"] ** 2
        out.twist.covariance[35] = P.NOMINAL_STD["omega_encoder"] ** 2
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderOdomNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
