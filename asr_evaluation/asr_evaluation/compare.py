"""Phase 9+10: Fixed EKF vs Rule vs RuleZ vs AI-Adaptive EKF on the test
episodes of a trained model set (plan sections 20-21).

Every AI number is the mean over the seed models given in --models (same
split), reported with its std, so the spread across training runs is part
of the result. Produces results/<tag>/{comparison.csv, detection.csv,
summary.md, ate_by_scenario.png, episode_*.png}.

    python3 -m asr_evaluation.compare --models data/models/main/seed_*
    python3 -m asr_evaluation.compare --models data/models/holdout_house/seed_0 \\
        --scenarios normal slip enc_drift imu_bias lidar_sector combo
"""
import argparse
import json
import pathlib

import numpy as np

from asr_core import params as P
from asr_core.hybrid import combine as hybrid_combine
from asr_core.labels import make_labels
from asr_core.metrics import _align_first, ate, detection_metrics, rpe
from asr_core.pipeline import run_adaptive, smooth_reliability, tick_sample
from asr_core.rules import RuleBaseline, RuleZ, gates_ok

KEY_SCENARIOS = ["slip", "combo"]  # per-episode plots


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


def s_raw_rules(tb, feats, meta, start):
    rb = RuleBaseline(meta["stats"])
    rz = RuleZ(meta["feature_normal_mean"], meta["feature_normal_std"])
    n = tb["t"].size
    out_b, out_z = np.ones((n, len(P.CHANNELS))), np.ones((n, len(P.CHANNELS)))
    for i in range(start, n):
        out_b[i] = rb.s_raw(tick_sample(tb, i))
        out_z[i] = rz.s_raw(feats[i - start])
    return out_b, out_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", nargs="+", default=["asr_reliability_estimator/model"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None, help="results subfolder (default: model tag)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    args = ap.parse_args()
    data = pathlib.Path(args.data)

    import onnxruntime as ort
    models = [pathlib.Path(m) for m in args.models]
    metas = [json.load(open(m / "stats.json")) for m in models]
    sessions = [ort.InferenceSession(str(m / "reliability.onnx"),
                                     providers=["CPUExecutionProvider"]) for m in models]
    meta = metas[0]
    split = meta["split"]
    tag = args.tag or meta.get("tag", "main")
    res = pathlib.Path(args.results) / tag
    res.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted((data / "episodes").glob("*.npz"))
             if f.stem.split("__")[0] in split["test"]
             and (args.scenarios is None or f.stem.split("__")[1] in args.scenarios)]
    if not files:
        raise SystemExit("no test episodes")
    print(f"[{tag}] {len(files)} test episodes, {len(models)} AI model(s)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj_rows, det_rows, plotted = [], [], set()
    for f in files:
        z, tb, lid, run = load_episode(f)
        run_name, scenario = f.stem.split("__")
        start, n = int(z["feat_start"]), tb["t"].size
        gt_pose = np.column_stack([tb["gt_x"], tb["gt_y"], tb["gt_yaw"]])
        masks = z["masks"]
        # ground-truth degradation (injected or natural: bumps, real slip,
        # corridor scan matching). Used for the strict false-alarm region and
        # as the honest detection target next to the injected mask.
        gt_bad = make_labels(z["errors"], np.asarray(meta["label_sigma"])) < 0.5

        s_b, s_z = s_raw_rules(tb, z["features"], meta, start)
        s_sets = {
            "fixed": np.ones((n, len(P.CHANNELS))),
            "rule": smooth_reliability(tb, s_b, meta["stats"], gates_ok),
            "rule_z": smooth_reliability(tb, s_z, meta["stats"], gates_ok),
        }
        for k, (sess, m) in enumerate(zip(sessions, metas)):
            s_ai_k = s_raw_ai(sess, z["features"], m, start, n)
            s_sets[f"ai_{k}"] = smooth_reliability(tb, s_ai_k, meta["stats"], gates_ok)
            # hybrid: AI everywhere except LiDAR channels, where the plain
            # Rule wins on held-out validation data (see asr_core.hybrid)
            s_sets[f"hybrid_{k}"] = smooth_reliability(
                tb, hybrid_combine(s_ai_k, s_b), meta["stats"], gates_ok)

        row = {"episode": f.stem, "run": run_name, "world": run_name.rsplit("_", 1)[0],
               "scenario": scenario}
        poses = {}
        for mode, s_ticks in s_sets.items():
            reject = -1.0 if mode == "fixed" else P.S_REJECT
            pose = run_adaptive(run, lid, tb, s_ticks, s_reject=reject)
            poses[mode] = pose
            row[f"ate_{mode}"] = ate(gt_pose, pose)
            row[f"rpe_{mode}"] = rpe(gt_pose, pose, int(P.TICK_RATE))[0]
        for tag_, prefix in (("ai", "ai_"), ("hybrid", "hybrid_")):
            vals_ate = [row[f"ate_{prefix}{k}"] for k in range(len(models))]
            vals_rpe = [row[f"rpe_{prefix}{k}"] for k in range(len(models))]
            row[f"ate_{tag_}"] = float(np.mean(vals_ate))
            row[f"ate_{tag_}_std"] = float(np.std(vals_ate))
            row[f"rpe_{tag_}"] = float(np.mean(vals_rpe))
            row[f"rpe_{tag_}_std"] = float(np.std(vals_rpe))
        traj_rows.append(row)
        print(f"{f.stem}: fixed={row['ate_fixed']:.3f} rule={row['ate_rule']:.3f} "
              f"rule_z={row['ate_rule_z']:.3f} ai={row['ate_ai']:.3f}±{row['ate_ai_std']:.3f} "
              f"hybrid={row['ate_hybrid']:.3f}±{row['ate_hybrid_std']:.3f}")

        modes = (["rule", "rule_z"] + [f"ai_{k}" for k in range(len(models))]
                + [f"hybrid_{k}" for k in range(len(models))])
        for mode in modes:
            for i, c in enumerate(P.CHANNELS):
                if not masks[:, i].any() and scenario != "normal":
                    continue
                d = detection_metrics(tb["t"], s_sets[mode][:, i], masks[:, i],
                                      exclude=gt_bad[:, i])
                g = detection_metrics(tb["t"], s_sets[mode][:, i], gt_bad[:, i])
                mode_name = ("ai" if mode.startswith("ai_")
                            else "hybrid" if mode.startswith("hybrid_") else mode)
                det_rows.append({"episode": f.stem, "scenario": scenario, "world": row["world"],
                                 "mode": mode_name,
                                 "seed": mode, "channel": c, **d,
                                 "f1_gt": g["f1"], "auroc_gt": g["auroc"]})

        if scenario in KEY_SCENARIOS and scenario not in plotted:
            plotted.add(scenario)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            t = tb["t"] - tb["t"][0]
            for i, c in enumerate(P.CHANNELS):
                axes[0].plot(t, s_sets["ai_0"][:, i], label=c)
                if masks[:, i].any():
                    axes[0].fill_between(t, 0, 1, where=masks[:, i], alpha=0.1, color="red")
            axes[0].set_title(f"AI reliability — {f.stem}")
            axes[0].set_ylim(-0.05, 1.05)
            axes[0].legend(fontsize=7)
            g = _align_first(gt_pose)
            axes[1].plot(g[:, 0], g[:, 1], "k-", label="GT")
            for mode, style in (("fixed", "r--"), ("rule_z", "g-."), ("ai_0", "b-"),
                               ("hybrid_0", "m:")):
                p = _align_first(poses[mode])
                axes[1].plot(p[:, 0], p[:, 1], style,
                             label=f"{mode} (ATE {row['ate_' + mode]:.3f})")
            axes[1].legend(fontsize=8)
            axes[1].axis("equal")
            axes[1].set_title("trajectories")
            fig.tight_layout()
            fig.savefig(res / f"episode_{scenario}.png", dpi=130)
            plt.close(fig)

    import pandas as pd
    tdf, ddf = pd.DataFrame(traj_rows), pd.DataFrame(det_rows)
    tdf.to_csv(res / "comparison.csv", index=False)
    ddf.to_csv(res / "detection.csv", index=False)

    cols = ["ate_fixed", "ate_rule", "ate_rule_z", "ate_ai", "ate_ai_std",
            "ate_hybrid", "ate_hybrid_std",
            "rpe_fixed", "rpe_rule", "rpe_rule_z", "rpe_ai", "rpe_hybrid"]
    by_sc = tdf.groupby("scenario")[cols].mean()
    by_world = tdf.groupby("world")[cols].mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    by_sc[["ate_fixed", "ate_rule", "ate_rule_z", "ate_ai", "ate_hybrid"]].plot.bar(
        ax=ax, yerr=[np.zeros(len(by_sc))] * 3 + [by_sc["ate_ai_std"].values,
                                                  by_sc["ate_hybrid_std"].values], capsize=2)
    ax.set_ylabel("ATE RMSE [m]")
    ax.set_title(f"Fixed vs Rule vs RuleZ vs AI vs Hybrid ({tag}; {len(models)} seed(s))")
    fig.tight_layout()
    fig.savefig(res / "ate_by_scenario.png", dpi=130)

    det_cols = ["f1", "auroc", "f1_gt", "auroc_gt", "latency", "recovery", "far",
                "far_clean", "chatter_per_min", "sat_frac"]
    fault_det = ddf[ddf["scenario"] != "normal"].dropna(subset=["f1"])
    by_mode = fault_det.groupby("mode")[det_cols].mean()
    far_normal = ddf[ddf["scenario"] == "normal"].groupby("mode")[
        ["far", "far_clean", "chatter_per_min", "sat_frac"]].mean()
    with open(res / "summary.md", "w") as fh:
        fh.write(f"# Fixed vs Rule vs RuleZ vs AI vs Hybrid Adaptive EKF — {tag}\n\n")
        fh.write(f"test runs: {split['test']}\nAI models: {[str(m) for m in models]}\n")
        fh.write("Hybrid = AI for every channel except v_lidar/omega_lidar, where the plain "
                 "Rule is used instead (asr_core.hybrid, chosen on the validation split)\n\n")
        fh.write("## ATE / RPE by scenario (mean over test episodes; *_std = spread over seeds)\n\n")
        fh.write("```\n" + by_sc.round(4).to_string() + "\n```\n\n")
        fh.write("## ATE / RPE by world\n\n```\n" + by_world.round(4).to_string() + "\n```\n\n")
        fh.write("## Detection on faulted channels (fault episodes; *_gt = vs ground-truth "
                 "degradation instead of the injected mask; sat_frac = fraction of ticks "
                 "with R pinned near R_max, plan section 21.4)\n\n")
        fh.write("```\n" + by_mode.round(4).to_string() + "\n```\n\n")
        fh.write("## False alarms on normal episodes (all channels; far_clean excludes "
                 "ticks where ground truth says the sensor really was wrong)\n\n")
        fh.write("```\n" + far_normal.round(4).to_string() + "\n```\n\n")
        fh.write("(generated by asr_evaluation.compare)\n")
    print(f"\nwrote {res}/")
    print(by_sc[["ate_fixed", "ate_rule", "ate_rule_z", "ate_ai", "ate_hybrid"]].round(4))
    print(by_mode.round(3))


if __name__ == "__main__":
    main()
