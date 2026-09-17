"""Calibrate nominal measurement stds (params.NOMINAL_STD) from normal runs:
per-(sensor x DOF) error RMS against ground truth, pooled over all extracted
runs and reported per world (plan section 12.3). Prints the dict to paste
into asr_core/params.py; nothing is written automatically.

    python3 -m asr_evaluation.calibrate --extracted data/extracted
"""
import argparse
import pathlib

import numpy as np

from asr_core import params as P
from asr_core.labels import channel_errors
from asr_core.pipeline import build_tick_table, lidar_odometry_offline, run_ekf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="data/extracted")
    args = ap.parse_args()
    from .build_dataset import load_run
    per_world = {}
    for f in sorted(pathlib.Path(args.extracted).glob("*.npz")):
        run = load_run(f)
        lid = lidar_odometry_offline(run)
        shadow = run_ekf(run, lid, r_provider=None, log_updates=True)
        tb = build_tick_table(run, lid, shadow)
        err = channel_errors({k: tb[k] for k in ("v_enc", "w_enc", "v_lid", "w_lid", "w_imu")},
                             tb["gt_v"], tb["gt_w"])
        n_scan = len(run["scans"]) - 1
        per_world.setdefault(f.stem.rsplit("_", 1)[0], []).append(
            (err, 1.0 - lid["t"].size / max(n_scan, 1)))
    allerr = []
    for w, items in per_world.items():
        e = np.vstack([i[0] for i in items])
        allerr.append(e)
        rms = np.sqrt((e ** 2).mean(0))
        inv = np.mean([i[1] for i in items])
        print(f"{w:12s} runs={len(items):2d} icp_invalid={100 * inv:.1f}%  " +
              "  ".join(f"{c}={r:.4f}" for c, r in zip(P.CHANNELS, rms)))
    rms = np.sqrt((np.vstack(allerr) ** 2).mean(0))
    print("\npooled NOMINAL_STD suggestion (RMS, rounded up ~10%):")
    print("NOMINAL_STD = {")
    for c, r in zip(P.CHANNELS, rms):
        print(f'    "{c}": {float(np.ceil(r * 1.1 * 1000) / 1000):.3f},')
    print("}")


if __name__ == "__main__":
    main()
