"""rosbag2 -> per-run npz with raw sensor streams (Phase 2 input).

    ros2 run asr_evaluation extract_bag -- --raw data/raw --out data/extracted
"""
import argparse
import math
import pathlib

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, JointState, LaserScan
from nav_msgs.msg import Odometry


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read_bag(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {"/joint_states": JointState, "/imu": Imu, "/scan": LaserScan,
             "/asr/ground_truth": Odometry}
    out = {k: [] for k in types}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in types:
            out[topic].append(deserialize_message(data, types[topic]))
    return out


def extract(path, imu_subsample=4):
    msgs = read_bag(path)
    run = {}

    # encoder: v, w from wheel joint velocities (fallback: differentiate pos)
    rows = []
    prev = None
    from asr_core.params import WHEEL_RADIUS, WHEEL_SEPARATION
    for m in msgs["/joint_states"]:
        try:
            il = m.name.index("wheel_left_joint")
            ir = m.name.index("wheel_right_joint")
        except ValueError:
            continue
        t = stamp(m)
        wl = wr = None
        if len(m.velocity) > max(il, ir):
            wl, wr = m.velocity[il], m.velocity[ir]
        if wl is None:
            if prev is not None and t > prev[0]:
                dt = t - prev[0]
                wl = (m.position[il] - prev[1]) / dt
                wr = (m.position[ir] - prev[2]) / dt
            prev = (t, m.position[il], m.position[ir])
            if wl is None:
                continue
        vl, vr = wl * WHEEL_RADIUS, wr * WHEEL_RADIUS
        rows.append((t, (vr + vl) / 2.0, (vr - vl) / WHEEL_SEPARATION))
    a = np.array(rows)
    run["t_enc"], run["v_enc"], run["w_enc"] = a[:, 0], a[:, 1], a[:, 2]

    rows = []
    for i, m in enumerate(msgs["/imu"]):
        if i % imu_subsample:
            continue
        acc = m.linear_acceleration
        rows.append((stamp(m), m.angular_velocity.z, acc.x,
                     math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)))
    a = np.array(rows)
    run["t_imu"], run["w_imu"], run["ax_imu"], run["a_norm"] = \
        a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    scans, scan_t = [], []
    first = msgs["/scan"][0]
    for m in msgs["/scan"]:
        scan_t.append(stamp(m))
        scans.append(np.asarray(m.ranges, np.float32))
    run["scan_t"] = np.asarray(scan_t)
    run["scans"] = np.vstack(scans)
    run["angle_min"] = first.angle_min
    run["angle_increment"] = first.angle_increment
    run["range_min"] = max(first.range_min, 0.12)
    run["range_max"] = min(first.range_max, 3.4)

    rows = []
    for m in msgs["/asr/ground_truth"]:
        q = m.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        rows.append((stamp(m), m.pose.pose.position.x, m.pose.pose.position.y,
                     yaw, m.twist.twist.linear.x, m.twist.twist.angular.z))
    a = np.array(rows)
    # gazebo_ros_state stamps can repeat; keep strictly increasing
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    a = a[keep]
    run["t_gt"], run["gt_x"], run["gt_y"] = a[:, 0], a[:, 1], a[:, 2]
    run["gt_yaw"], run["gt_v"], run["gt_w"] = a[:, 3], a[:, 4], a[:, 5]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/extracted")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for bag in sorted(p for p in pathlib.Path(args.raw).iterdir()
                      if (p / "metadata.yaml").exists()):
        if bag.suffix == ".partial":
            continue
        dst = out / f"{bag.name}.npz"
        if dst.exists():
            print(f"skip {dst}")
            continue
        print(f"extract {bag} -> {dst}")
        run = extract(bag)
        np.savez_compressed(dst, **run)
        n = run["t_enc"].size
        print(f"  enc {n} imu {run['t_imu'].size} scans {run['scan_t'].size} "
              f"gt {run['t_gt'].size} span {run['t_enc'][-1] - run['t_enc'][0]:.1f}s")


if __name__ == "__main__":
    main()
