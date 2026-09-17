"""Hyperparameter sweep for reliability smoothing + covariance inflation
(plan sections 14.2 "gamma_fall/gamma_rise는 실험으로 조정", 15.3 "threshold는
실험으로 결정") over the deployed model — no retraining needed, since these
parameters are applied *after* ONNX inference (asr_core.smoothing /
asr_core.pipeline.run_adaptive).

A full grid over both axes is too slow (each combo replays the EKF over
every cached episode, ~10-15s each): instead this varies one axis at a time
around the current params.py defaults (coordinate sweep), on one
representative test run per world x 3 scenarios. Prints a ranked table and
writes results/smoothing_sweep.json; does not modify params.py itself.

    python3 -m asr_evaluation.sweep_smoothing --model asr_reliability_estimator/model
"""
import argparse
import json
import pathlib
import time

import numpy as np

from asr_core import params as P
from asr_core.metrics import ate, detection_metrics
from asr_core.pipeline import run_adaptive, tick_sample
from asr_core.rules import gates_ok
from asr_core.smoothing import ReliabilityFilter

SWEEP_SCENARIOS = ["normal", "slip", "imu_bias"]

SMOOTH_VARIANTS = [  # (gamma_fall, gamma_rise, recovery_ticks) -- one axis moved at a time
    (P.GAMMA_FALL, P.GAMMA_RISE, P.RECOVERY_TICKS),               # baseline
    (0.20, P.GAMMA_RISE, P.RECOVERY_TICKS),
    (0.45, P.GAMMA_RISE, P.RECOVERY_TICKS),
    (P.GAMMA_FALL, 0.80, P.RECOVERY_TICKS),
    (P.GAMMA_FALL, 0.95, P.RECOVERY_TICKS),
    (P.GAMMA_FALL, P.GAMMA_RISE, 5),
    (P.GAMMA_FALL, P.GAMMA_RISE, 15),
]
COV_VARIANTS = [  # (alpha, s_reject)
    (P.ALPHA_INFLATION, P.S_REJECT),                               # baseline
    (P.ALPHA_INFLATION, 0.05),
    (29.0, P.S_REJECT),
    (79.0, P.S_REJECT),
]


def load_episode(path):
    z = np.load(path, allow_pickle=True)
    tb = {k[3:]: z[k] for k in z.files if k.startswith("tb_")}
    lid = {k[4:]: z[k] for k in z.files if k.startswith("lid_")}
    run = {"t_enc": z["t_enc"], "v_enc": z["v_enc"], "w_enc": z["w_enc"],
           "t_imu": z["t_imu"], "w_imu": z["w_imu"]}
    return z, tb, lid, run


def s_raw_ai(sess, feats, meta, start, n_ticks):
    x = ((feats - np.asarray(meta["feature_mean"])) /
         np.asarray(meta["feature_std"])).astype(np.float32)
    s = sess.run(None, {"features": x})[0]
    out = np.ones((n_ticks, len(P.CHANNELS)))
    out[start:start + s.shape[0]] = s
    return out


def score_combo(cache, gf, gr, rt, sfl, alpha, s_reject):
    rows = []
    for scen, tb, lid, run, s_raw, masks, gate_arr, gt_pose in cache:
        rf = ReliabilityFilter(len(P.CHANNELS), gf, gr, rt, sfl)
        sm = np.array([rf.step(s_raw[i], gate_arr[i], bool(tb["stationary"][i]))
                       for i in range(tb["t"].size)])
        pose = run_adaptive(run, lid, tb, sm, alpha=alpha, s_reject=s_reject)
        a = ate(gt_pose, pose)
        far = chatter = n_ch = 0.0
        for i in range(len(P.CHANNELS)):
            if scen == "normal" or masks[:, i].any():
                d = detection_metrics(tb["t"], sm[:, i], masks[:, i])
                far += d["far"] if scen == "normal" else 0.0
                chatter += d["chatter_per_min"]
                n_ch += 1
        rows.append((scen, a, far, chatter / max(n_ch, 1)))
    fault_ate = np.mean([r[1] for r in rows if r[0] != "normal"])
    normal_ate = np.mean([r[1] for r in rows if r[0] == "normal"])
    far = np.mean([r[2] for r in rows if r[0] == "normal"])
    chatter = np.mean([r[3] for r in rows])
    score = fault_ate + 5.0 * normal_ate + 2.0 * far  # lower is better
    return {"fault_ate": round(float(fault_ate), 4), "normal_ate": round(float(normal_ate), 4),
            "far": round(float(far), 4), "chatter_per_min": round(float(chatter), 3),
            "score": round(float(score), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="asr_reliability_estimator/model")
    ap.add_argument("--out", default="results/smoothing_sweep.json")
    ap.add_argument("--runs-per-world", type=int, default=1)
    args = ap.parse_args()
    data = pathlib.Path(args.data)
    mdir = pathlib.Path(args.model)
    meta = json.load(open(mdir / "stats.json"))
    stats = meta["stats"]
    split = meta["split"]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(mdir / "reliability.onnx"),
                                providers=["CPUExecutionProvider"])

    by_world = {}
    for r in split["test"]:
        by_world.setdefault(r.rsplit("_", 1)[0], []).append(r)
    runs = [r for g in by_world.values() for r in sorted(g)[:args.runs_per_world]]
    files = [f for f in sorted((data / "episodes").glob("*.npz"))
             if f.stem.split("__")[0] in runs and f.stem.split("__")[1] in SWEEP_SCENARIOS]
    if not files:
        raise SystemExit("no matching test episodes for sweep")
    print(f"sweeping over {len(files)} episodes from runs {runs}: {SWEEP_SCENARIOS}")

    # cache raw AI output, recovery gate (param-independent!) and GT pose once
    cache = []
    for f in files:
        z, tb, lid, run = load_episode(f)
        start, n = int(z["feat_start"]), tb["t"].size
        s_raw = s_raw_ai(sess, z["features"], meta, start, n)
        gate_arr = np.array([gates_ok(tick_sample(tb, i), stats) for i in range(n)])
        gt_pose = np.column_stack([tb["gt_x"], tb["gt_y"], tb["gt_yaw"]])
        cache.append((f.stem.split("__")[1], tb, lid, run, s_raw, z["masks"], gate_arr, gt_pose))

    results = []
    t0 = time.time()
    for gf, gr, rt in SMOOTH_VARIANTS:
        r = score_combo(cache, gf, gr, rt, P.STATIONARY_FALL_LIMIT,
                        P.ALPHA_INFLATION, P.S_REJECT)
        r.update({"axis": "smooth", "gamma_fall": gf, "gamma_rise": gr, "recovery_ticks": rt,
                  "alpha": P.ALPHA_INFLATION, "s_reject": P.S_REJECT})
        results.append(r)
        print(f"[smooth] gf={gf} gr={gr} rt={rt:2d} -> {r} ({time.time()-t0:.0f}s elapsed)")
    for alpha, s_reject in COV_VARIANTS:
        r = score_combo(cache, P.GAMMA_FALL, P.GAMMA_RISE, P.RECOVERY_TICKS,
                        P.STATIONARY_FALL_LIMIT, alpha, s_reject)
        r.update({"axis": "cov", "gamma_fall": P.GAMMA_FALL, "gamma_rise": P.GAMMA_RISE,
                  "recovery_ticks": P.RECOVERY_TICKS, "alpha": alpha, "s_reject": s_reject})
        results.append(r)
        print(f"[cov]    alpha={alpha:4.0f} rej={s_reject:.2f} -> {r} ({time.time()-t0:.0f}s elapsed)")

    baseline = next(r for r in results if r["gamma_fall"] == P.GAMMA_FALL
                    and r["gamma_rise"] == P.GAMMA_RISE and r["recovery_ticks"] == P.RECOVERY_TICKS
                    and r["alpha"] == P.ALPHA_INFLATION and r["s_reject"] == P.S_REJECT)
    ranked = sorted(results, key=lambda r: r["score"])
    print(f"\ncurrent defaults score = {baseline['score']}")
    print("=== ranked (best first) ===")
    for r in ranked:
        tag = " <- current default" if r is baseline else ""
        print(f"  score={r['score']:.4f}  gf={r['gamma_fall']} gr={r['gamma_rise']} "
              f"rt={r['recovery_ticks']:2d} alpha={r['alpha']:4.0f} rej={r['s_reject']:.2f}"
              f"  fault_ate={r['fault_ate']:.3f} normal_ate={r['normal_ate']:.4f} "
              f"far={r['far']:.3f}{tag}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"baseline": baseline, "ranked": ranked}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
