"""Analyze a bag recorded from the live pipeline (asr_demo.launch.py):
reliability rate (realtime check), per-channel reliability inside/outside
the injected fault window, and ATE of Fixed vs Adaptive EKF vs ground truth.

    python3 -m asr_evaluation.analyze_live --bag /tmp/asr_demo_bag
"""
import argparse
import math

import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message

from asr_core.metrics import ate
from asr_msgs.msg import FaultStatus, ReliabilityArray

CH = ["v_encoder", "omega_encoder", "v_lidar", "omega_lidar", "omega_imu"]


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def read_bag(path):
    """Returns (rel, faults, odom): rel is (n, 6) [t, *CH], faults is (n, 2)
    [t, active], odom is {"gt"|"fixed"|"adaptive": (n, 4) [t, x, y, yaw]}.
    Shared by the CLI report below and asr_evaluation.smoke_test_live."""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    rel, faults, odom = [], [], {"gt": [], "fixed": [], "adaptive": []}
    key = {"/asr/ground_truth": "gt", "/asr/odom_fixed": "fixed",
           "/asr/odom_adaptive": "adaptive"}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/asr/reliability":
            m = deserialize_message(data, ReliabilityArray)
            v = {}
            for s in m.sensors:
                v[f"v_{s.sensor_name}"] = s.s_v
                v[f"omega_{s.sensor_name}"] = s.s_omega
            rel.append((stamp(m), *[v[c] for c in CH]))
        elif topic == "/asr/fault_status":
            m = deserialize_message(data, FaultStatus)
            faults.append((stamp(m), m.active))
        elif topic in key:
            m = deserialize_message(data, Odometry)
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            odom[key[topic]].append((stamp(m), m.pose.pose.position.x,
                                     m.pose.pose.position.y, yaw))
    rel = np.array(rel)
    faults = np.array(faults)
    odom = {k: np.array(v) for k, v in odom.items()}
    return rel, faults, odom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    args = ap.parse_args()
    rel, f, odom = read_bag(args.bag)
    print(f"reliability msgs: {len(rel)}  rate: "
          f"{(len(rel) - 1) / (rel[-1, 0] - rel[0, 0]):.1f} Hz (target 20)")
    active = f[f[:, 1] > 0, 0]
    if active.size:
        t0, t1 = active.min(), active.max()
        inwin = (rel[:, 0] >= t0) & (rel[:, 0] <= t1)
        print(f"fault window {t0 - rel[0, 0]:.1f}s .. {t1 - rel[0, 0]:.1f}s")
        for i, c in enumerate(CH):
            print(f"  {c:14s} mean s in-fault {rel[inwin, 1 + i].mean():.2f}   "
                  f"outside {rel[~inwin, 1 + i].mean():.2f}   "
                  f"min in-fault {rel[inwin, 1 + i].min():.2f}")
    gt = odom["gt"]
    for k in ("fixed", "adaptive"):
        o = odom[k]
        if o.size == 0:
            continue
        t = o[:, 0]
        g = np.column_stack([np.interp(t, gt[:, 0], gt[:, 1]),
                             np.interp(t, gt[:, 0], gt[:, 2]),
                             np.interp(t, gt[:, 0], np.unwrap(gt[:, 3]))])
        print(f"ATE {k:9s}: {ate(g, o[:, 1:]):.3f} m over {t[-1] - t[0]:.0f}s")


if __name__ == "__main__":
    main()
