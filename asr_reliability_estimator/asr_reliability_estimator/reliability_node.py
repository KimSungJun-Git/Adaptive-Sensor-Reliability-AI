"""Reliability estimator node: synchronized features -> AI inference (ONNX)
or rule baseline -> asymmetric smoothing + recovery gate -> ReliabilityArray.
"""
import json
import os

import numpy as np
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from ament_index_python.packages import get_package_share_directory
from asr_core import params as P
from asr_core.features import FeatureBuilder, StationaryDetector
from asr_core.hybrid import combine as hybrid_combine
from asr_core.rules import RuleBaseline, RuleZ, gates_ok
from asr_core.smoothing import ReliabilityFilter
from asr_msgs.msg import Innovation, LidarQuality, ReliabilityArray, SensorReliability

from .feature_builder import Series


def default_model_dir():
    return os.path.join(get_package_share_directory("asr_reliability_estimator"),
                        "model")


class ReliabilityNode(Node):
    def __init__(self):
        super().__init__("asr_reliability")
        self.declare_parameter("mode", "ai")            # ai | hybrid | rule | rule_z
        # "hybrid" (AI everywhere except a Rule-only v_lidar/omega_lidar) was
        # tried and rejected as the default: it nets *worse* overall despite
        # winning on lidar-only faults, because Rule's own residual vote
        # mis-attributes encoder drift/stuck to LiDAR and then rejects a
        # perfectly good LiDAR measurement -- the same shared-attribution
        # problem as AI's encoder channel getting contaminated by LiDAR
        # faults, just mirrored. See asr_core/hybrid.py and README.
        self.declare_parameter("model_dir", "")
        self.declare_parameter("tick_delay", 0.15)     # interpolation latency budget
        self.mode = self.get_parameter("mode").value
        mdir = self.get_parameter("model_dir").value or default_model_dir()
        with open(os.path.join(mdir, "stats.json")) as f:
            meta = json.load(f)
        self.stats = meta["stats"]
        self.session = None
        if self.mode in ("ai", "hybrid"):
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                os.path.join(mdir, "reliability.onnx"),
                providers=["CPUExecutionProvider"])
            self.f_mean = np.asarray(meta["feature_mean"], np.float32)
            self.f_std = np.asarray(meta["feature_std"], np.float32)
        self.rule = RuleBaseline(self.stats)
        self.rule_z = RuleZ(meta["feature_normal_mean"], meta["feature_normal_std"]) \
            if "feature_normal_mean" in meta else None

        self.enc = Series(2)
        self.imu = Series(3)      # w, ax, a_norm
        self.lid = Series(2)
        self.lidq = Series(3)     # rms, ratio, degeneracy
        self.innov = {"encoder": Series(3), "lidar": Series(3), "imu": Series(2)}
        self.fb = FeatureBuilder(P.TICK_RATE, P.WINDOW_SEC)
        self.stat = StationaryDetector(P.STAT_V_THR, P.STAT_W_THR, P.STAT_A_THR,
                                       P.STAT_HOLD_SEC, P.TICK_RATE)
        self.rf = ReliabilityFilter(len(P.CHANNELS), P.GAMMA_FALL, P.GAMMA_RISE,
                                    P.RECOVERY_TICKS, P.STATIONARY_FALL_LIMIT)
        self.delay = float(self.get_parameter("tick_delay").value)

        self.pub = self.create_publisher(ReliabilityArray, "/asr/reliability", 10)
        self.create_subscription(TwistWithCovarianceStamped, "/asr/encoder_twist",
                                 self.enc_cb, 20)
        self.create_subscription(TwistWithCovarianceStamped, "/asr/lidar_twist",
                                 self.lid_cb, 20)
        self.create_subscription(LidarQuality, "/asr/lidar_quality", self.lidq_cb, 20)
        self.create_subscription(Imu, "/asr/imu", self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(Innovation, "/asr/shadow/innovation",
                                 self.innov_cb, 50)
        self.create_timer(1.0 / P.TICK_RATE, self.tick)

    @staticmethod
    def _t(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def enc_cb(self, m):
        self.enc.push(self._t(m), m.twist.twist.linear.x, m.twist.twist.angular.z)

    def lid_cb(self, m):
        self.lid.push(self._t(m), m.twist.twist.linear.x, m.twist.twist.angular.z)

    def lidq_cb(self, m):
        self.lidq.push(self._t(m), m.rms, m.match_ratio, m.degeneracy)

    def imu_cb(self, m):
        a = m.linear_acceleration
        self.imu.push(self._t(m), m.angular_velocity.z, a.x,
                      float(np.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)))

    def innov_cb(self, m):
        s = m.sensor_name
        if s in ("encoder", "lidar"):
            self.innov[s].push(self._t(m), m.innovation[0], m.innovation[1], m.nis)
        else:
            self.innov[s].push(self._t(m), m.innovation[0], m.nis)

    def tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.delay
        enc = self.enc.interp(t)
        imu = self.imu.interp(t)
        lid = self.lid.last_at(t)
        if enc is None or imu is None or lid is None:
            return
        lidq = self.lidq.last_at(t) or (9.9, [9.9, 0.0, 0.0])
        i_enc = self.innov["encoder"].last_at(t) or (9.9, [0.0, 0.0, 2.0])
        i_lid = self.innov["lidar"].last_at(t) or (9.9, [0.0, 0.0, 2.0])
        i_imu = self.innov["imu"].last_at(t) or (9.9, [0.0, 1.0])
        stationary = self.stat.step(enc[0], imu[0], imu[2])
        sample = {
            "v_enc": enc[0], "w_enc": enc[1],
            "w_imu": imu[0], "ax_imu": imu[1],
            "v_lid": lid[1][0], "w_lid": lid[1][1], "lid_age": lid[0],
            "lid_rms": lidq[1][0], "lid_ratio": lidq[1][1], "lid_degen": lidq[1][2],
            "nu_v_enc": i_enc[1][0], "nu_w_enc": i_enc[1][1], "nis_enc": i_enc[1][2],
            "nu_v_lid": i_lid[1][0], "nu_w_lid": i_lid[1][1], "nis_lid": i_lid[1][2],
            "nu_w_imu": i_imu[1][0], "nis_imu": i_imu[1][1],
            "stationary": float(stationary),
        }
        feat = self.fb.push(sample)
        if feat is None:
            return
        if self.session is not None:
            x = ((feat.astype(np.float32) - self.f_mean) / self.f_std)[None, :]
            s_raw = self.session.run(None, {"features": x})[0][0]
            if self.mode == "hybrid":
                # v_lidar/omega_lidar: plain Rule beats the MLP on held-out
                # data (asr_core.hybrid) -- everything else keeps the MLP
                s_raw = hybrid_combine(s_raw, self.rule.s_raw(sample))
        elif self.mode == "rule_z" and self.rule_z is not None:
            s_raw = self.rule_z.s_raw(feat)
        else:
            s_raw = self.rule.s_raw(sample)
        s = self.rf.step(s_raw, gates_ok(sample, self.stats), stationary)

        out = ReliabilityArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.stationary = bool(stationary)
        vals = dict(zip(P.CHANNELS, s))
        for name, sv, sw in (("encoder", vals["v_encoder"], vals["omega_encoder"]),
                             ("lidar", vals["v_lidar"], vals["omega_lidar"]),
                             ("imu", -1.0, vals["omega_imu"])):
            r = SensorReliability()
            r.header = out.header
            r.sensor_name = name
            r.s_v, r.s_omega = float(sv), float(sw)
            out.sensors.append(r)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ReliabilityNode())


if __name__ == "__main__":
    main()
