"""External-data distribution-shift check (plan section 12.4 / 25 problem 5).

Takes rosbags that carry only wheel odometry (/odom twist) and /scan — e.g.
the ros2_auto_tuner jj0614 recordings — runs the same ICP lidar odometry and
compares the encoder-vs-lidar residual statistics and the rule-gate false
alarm rate against the sim training statistics. No IMU / GT: the full
5-channel model is not evaluated here, only the encoder-lidar path.

    python3 -m asr_evaluation.external_check --bags ~/ros2_auto_tuner/database/bags/jj0614 ...
"""
import argparse
import json
import pathlib

import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan

from asr_core.pipeline import lidar_odometry_offline


def load(path):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    odom, scans, scan_t, first = [], [], [], None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/odom":
            m = deserialize_message(data, Odometry)
            odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                         m.twist.twist.linear.x, m.twist.twist.angular.z))
        elif topic == "/scan":
            m = deserialize_message(data, LaserScan)
            first = first or m
            scan_t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            scans.append(np.asarray(m.ranges, np.float32))
    o = np.array(odom)
    return {"t_enc": o[:, 0], "v_enc": o[:, 1], "w_enc": o[:, 2],
            "scan_t": np.asarray(scan_t), "scans": scans,
            "angle_min": first.angle_min, "angle_increment": first.angle_increment,
            "range_min": max(first.range_min, 0.12), "range_max": min(first.range_max, 3.4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--stats", default="data/stats.json")
    ap.add_argument("--out", default="results/external_check.json")
    args = ap.parse_args()
    stats = json.load(open(args.stats))["stats"]
    report = {}
    for b in args.bags:
        run = load(pathlib.Path(b).expanduser())
        lid = lidar_odometry_offline(run)
        v_enc = np.interp(lid["t"], run["t_enc"], run["v_enc"])
        w_enc = np.interp(lid["t"], run["t_enc"], run["w_enc"])
        ok = lid["ratio"] > 0.5
        r_v, r_w = v_enc[ok] - lid["v"][ok], w_enc[ok] - lid["w"][ok]
        rep = {
            "n_scans": int(ok.sum()),
            "r_v_EL_std": float(r_v.std()), "r_w_EL_std": float(r_w.std()),
            "sim_r_v_EL_std": stats["r_v_EL"], "sim_r_w_EL_std": stats["r_w_EL"],
            "rule_gate_far_v": float((np.abs(r_v) > 3 * stats["r_v_EL"]).mean()),
            "rule_gate_far_w": float((np.abs(r_w) > 3 * stats["r_w_EL"]).mean()),
            "icp_rms_median": float(np.median(lid["rms"][ok])),
        }
        report[pathlib.Path(b).name] = rep
        print(pathlib.Path(b).name, json.dumps(rep, indent=None))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
