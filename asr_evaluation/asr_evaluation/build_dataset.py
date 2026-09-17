"""Phase 4+5+6: offline fault injection on extracted runs, shadow-EKF replay,
feature + error/mask generation, per-run train/val/test split and normal-run
statistics (label sigmas, residual/innovation stds, feature normalization).

    ros2 run asr_evaluation build_dataset -- --extracted data/extracted --out data
"""
import argparse
import json
import pathlib

import numpy as np

from asr_core import params as P
from asr_core.faults import (SCENARIOS, STATIONARY_SCENARIOS, apply_encoder,
                             apply_imu, apply_lidar_scans, channel_masks,
                             find_stationary_window, scenario_events)
from asr_core.labels import channel_errors
from asr_core.pipeline import (build_features, build_tick_table,
                               lidar_odometry_offline, run_ekf)
from asr_core.rules import residual_stat_keys


def load_run(path):
    z = np.load(path)
    run = {k: z[k] for k in z.files}
    run["scans"] = [run["scans"][i] for i in range(run["scans"].shape[0])]
    for k in ("angle_min", "angle_increment", "range_min", "range_max"):
        run[k] = float(run[k])
    return run


def process_episode(run, events, lid_cache=None):
    """Apply events to raw streams, replay shadow EKF, build ticks/features."""
    t0 = float(run["t_enc"][0])
    ep = dict(run)  # shallow copy; arrays replaced below

    v, w, _, _ = apply_encoder(run["t_enc"] - t0, run["v_enc"], run["w_enc"], events)
    ep["v_enc"], ep["w_enc"] = v, w
    wi, ax, _ = apply_imu(run["t_imu"] - t0, run["w_imu"], run["ax_imu"], events)
    ep["w_imu"], ep["ax_imu"] = wi, ax

    lidar_faulted = any(e.sensor == "lidar" for e in events)
    if lidar_faulted or lid_cache is None:
        keep, scans, _ = apply_lidar_scans(run["scan_t"] - t0, run["scans"], events)
        ep["scan_t"] = run["scan_t"][keep]
        ep["scans"] = [s for s, k in zip(scans, keep) if k]
        lid = lidar_odometry_offline(ep)
    else:
        lid = lid_cache

    shadow = run_ekf(ep, lid, r_provider=None, log_updates=True)
    tb = build_tick_table(ep, lid, shadow)
    start, feats = build_features(tb)

    meas = {k: tb[k] for k in ("v_enc", "w_enc", "v_lid", "w_lid", "w_imu")}
    errors = channel_errors(meas, tb["gt_v"], tb["gt_w"])
    masks = channel_masks(tb["t"] - t0, events)
    return ep, lid, tb, start, feats, errors, masks


def split_runs(names, frac=(0.6, 0.2)):
    """Group by world prefix (<world>_<nn>), split each group 60/20/20."""
    groups = {}
    for n in sorted(names):
        groups.setdefault(n.rsplit("_", 1)[0], []).append(n)
    split = {"train": [], "val": [], "test": []}
    for g in groups.values():
        n = len(g)
        n_tr = max(1, int(round(n * frac[0])))
        n_va = max(1, int(round(n * frac[1]))) if n > 2 else 0
        split["train"] += g[:n_tr]
        split["val"] += g[n_tr:n_tr + n_va]
        split["test"] += g[n_tr + n_va:]
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="data/extracted")
    ap.add_argument("--out", default="data")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS + STATIONARY_SCENARIOS)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    epdir = out / "episodes"
    epdir.mkdir(parents=True, exist_ok=True)
    runs = sorted(pathlib.Path(args.extracted).glob("*.npz"))
    if not runs:
        raise SystemExit("no extracted runs found")

    # run-level split (plan section 17): no run straddles subsets; stratified
    # per world so every world appears in train / val / test
    split = split_runs([r.stem for r in runs])
    json.dump(split, open(out / "split.json", "w"), indent=1)
    print("split:", {k: len(v) for k, v in split.items()})

    normal_errors, normal_resid = [], {k: [] for k in residual_stat_keys()}
    normal_feats = []
    for rp in runs:
        run = load_run(rp)
        t0 = float(run["t_enc"][0])
        t_end = float(run["t_enc"][-1]) - t0
        # dedicated stationary scenarios (plan section 19, experiment 7) need
        # a genuine stationary segment of this run's ground truth, not the
        # default mid-run window; skip a run that never actually stops
        stat_win = find_stationary_window(run["t_gt"] - t0, run["gt_v"], run["gt_w"])
        lid_cache = None
        for sc in args.scenarios:
            is_stat = sc in STATIONARY_SCENARIOS
            if is_stat and stat_win is None:
                print(f"skip {rp.stem}__{sc}: no stationary segment in this run")
                continue
            dst = epdir / f"{rp.stem}__{sc}.npz"
            if dst.exists():
                if sc == "normal" and lid_cache is None:
                    z = np.load(dst)
                    lid_cache = {k[4:]: z[k] for k in z.files if k.startswith("lid_")}
                print(f"skip {dst.name}")
                continue
            # event times are relative to run start; process_episode shifts
            # every stream by t0 before applying them
            events = scenario_events(sc, t_end, seed=abs(hash(rp.stem + sc)) % 2**31,
                                     window=stat_win if is_stat else None)
            ep, lid, tb, start, feats, errors, masks = process_episode(
                run, events, lid_cache)
            if sc == "normal":
                lid_cache = lid
                if rp.stem in split["train"]:
                    normal_errors.append(errors)
                    normal_feats.append(feats)
                    for k in residual_stat_keys():
                        if k.startswith("r_"):
                            pair = {"r_v_EL": ("v_enc", "v_lid"),
                                    "r_w_EL": ("w_enc", "w_lid"),
                                    "r_w_EI": ("w_enc", "w_imu"),
                                    "r_w_LI": ("w_lid", "w_imu")}[k]
                            normal_resid[k].append(tb[pair[0]] - tb[pair[1]])
                        else:
                            col = {"nu_v_enc": "nu_v_enc", "nu_w_enc": "nu_w_enc",
                                   "nu_v_lid": "nu_v_lid", "nu_w_lid": "nu_w_lid",
                                   "nu_w_imu": "nu_w_imu"}[k]
                            normal_resid[k].append(tb[col])
            save = {f"tb_{k}": v for k, v in tb.items()}
            save.update({f"lid_{k}": v for k, v in lid.items()},
                        feat_start=start, features=feats, errors=errors,
                        masks=masks,
                        t_enc=ep["t_enc"], v_enc=ep["v_enc"], w_enc=ep["w_enc"],
                        t_imu=ep["t_imu"], w_imu=ep["w_imu"],
                        run=rp.stem, scenario=sc)
            np.savez_compressed(dst, **save)
            print(f"{dst.name}: ticks {tb['t'].size} feats {feats.shape}")

    if normal_errors:
        from asr_core.labels import estimate_sigma
        sigma = estimate_sigma(np.vstack(normal_errors))
        stats = {k: float(np.std(np.concatenate(v)))
                 for k, v in normal_resid.items()}
        nf = np.vstack(normal_feats)
        json.dump({"label_sigma": sigma.tolist(), "stats": stats,
                   "channels": P.CHANNELS,
                   "feature_normal_mean": nf.mean(0).tolist(),
                   "feature_normal_std": nf.std(0).tolist()},
                  open(out / "stats.json", "w"), indent=1)
        print("label sigma:", dict(zip(P.CHANNELS, np.round(sigma, 4))))
        print("residual stats:", {k: round(v, 4) for k, v in stats.items()})


if __name__ == "__main__":
    main()
