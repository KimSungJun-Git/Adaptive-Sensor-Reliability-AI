"""Phase 7: model training and selection.

Order per plan section 11: rule baselines -> Isolation Forest -> MLP
regression (primary), trained over several seeds so every reported number
carries a spread. Exports each seed to data/models/<tag>/seed_<k>/ and the
best-validation seed to asr_reliability_estimator/model/ for deployment.

    python3 -m asr_evaluation.train                        # stratified split
    python3 -m asr_evaluation.train --holdout house        # leave-one-world-out
"""
import argparse
import json
import pathlib
import shutil
import time

import numpy as np

from asr_core import params as P
from asr_core.features import FEATURE_NAMES
from asr_core.labels import estimate_sigma, make_labels, robust_sigma
from asr_core.metrics import auroc
from asr_core.pipeline import tick_sample
from asr_core.rules import RuleBaseline, RuleZ, residual_stat_keys

RESID_PAIRS = {"r_v_EL": ("v_enc", "v_lid"), "r_w_EL": ("w_enc", "w_lid"),
               "r_w_EI": ("w_enc", "w_imu"), "r_w_LI": ("w_lid", "w_imu")}


def world_of(run):
    return run.rsplit("_", 1)[0]


def make_split(data, holdout=None):
    """Run-level split. Default: data/split.json (stratified per world).
    holdout=<world>: that world's runs are the test set, the rest is split
    80/20 train/val per world."""
    if holdout is None:
        return json.load(open(data / "split.json"))
    runs = sorted({f.stem.split("__")[0] for f in (data / "episodes").glob("*.npz")})
    test = [r for r in runs if world_of(r) == holdout]
    rest = [r for r in runs if world_of(r) != holdout]
    if not test or not rest:
        raise SystemExit(f"holdout world '{holdout}' not found or nothing left")
    split = {"train": [], "val": [], "test": test}
    for w in sorted({world_of(r) for r in rest}):
        g = [r for r in rest if world_of(r) == w]
        n_va = max(1, int(round(0.2 * len(g)))) if len(g) > 2 else 0
        split["train"] += g[:len(g) - n_va]
        split["val"] += g[len(g) - n_va:]
    return split


def episode_files(data, split):
    eps = {"train": [], "val": [], "test": []}
    for f in sorted((data / "episodes").glob("*.npz")):
        run = f.stem.split("__")[0]
        for k, runs in split.items():
            if run in runs:
                eps[k].append(f)
    return eps


def normal_stats(train_files):
    """Everything derived from *normal train* episodes only, so a holdout
    world never leaks into thresholds: label sigmas, instantaneous residual /
    innovation stds (RuleBaseline + recovery gate), windowed feature
    mean/std (RuleZ)."""
    errs, feats, resid = [], [], {k: [] for k in residual_stat_keys()}
    for f in train_files:
        if not f.stem.endswith("__normal"):
            continue
        z = np.load(f, allow_pickle=True)
        errs.append(z["errors"])
        feats.append(z["features"])
        for k in residual_stat_keys():
            if k.startswith("r_"):
                a, b = RESID_PAIRS[k]
                resid[k].append(z[f"tb_{a}"] - z[f"tb_{b}"])
            else:
                resid[k].append(z[f"tb_{k}"])
    nf = np.vstack(feats)
    # residual / feature scales are p95-based like the labels: normal data is
    # heavy-tailed, and a std-based 3-sigma gate would be too loose to fire
    med = np.median(nf, axis=0)
    return {
        "label_sigma": estimate_sigma(np.vstack(errs)).tolist(),
        "stats": {k: float(robust_sigma(np.concatenate(v))) for k, v in resid.items()},
        "feature_normal_mean": med.tolist(),
        "feature_normal_std": robust_sigma(nf - med, axis=0).tolist(),
    }


def stack(files, sigma):
    Xs, ys, ms = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        start = int(z["feat_start"])
        Xs.append(z["features"])
        ys.append(make_labels(z["errors"][start:], sigma))
        ms.append(z["masks"][start:])
    return np.vstack(Xs), np.vstack(ys), np.vstack(ms)


def eval_channels(name, s_pred, y_mask, y_label=None, thr=0.5):
    """Per-channel detection vs the injected-fault mask and, if labels are
    given, vs the ground-truth label (label < thr = sensor really degraded,
    injected or not) - the latter is the honest target for lidar, whose
    injected degradations are largely absorbed by the scan matcher."""
    def score(m):
        det = s_pred[:, i] < thr
        tp, fp, fn = (det & m).sum(), (det & ~m).sum(), (~det & m).sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        # `prec and rec` looks like a "both defined" guard but isn't: nan is
        # truthy in Python, so a channel with zero predicted positives (a
        # real, not-rare case for an underused channel) slipped nan into f1
        # instead of the intended 0.0 fallback, then into np.mean() over
        # seeds in mlp_mean/mlp_std below.
        f1 = (2 * prec * rec / (prec + rec)
              if np.isfinite(prec) and np.isfinite(rec) and (prec or rec) else 0.0)
        return round(float(f1), 3), (round(auroc(1 - s_pred[:, i], m), 3) if m.any() else None)
    rows = {}
    for i, c in enumerate(P.CHANNELS):
        f1, auc = score(y_mask[:, i])
        rows[c] = {"f1": f1, "auroc": auc}
        if y_label is not None:
            f1g, aucg = score(y_label[:, i] < thr)
            rows[c].update({"f1_gt": f1g, "auroc_gt": aucg})
    print(f"[{name}] " + "  ".join(
        f"{c}: f1={v['f1']} auc={v['auroc']}" + (f" | gt f1={v['f1_gt']} auc={v['auroc_gt']}"
                                                 if "f1_gt" in v else "")
        for c, v in rows.items()))
    return rows


def rule_series(files, ns):
    rb, rz = RuleBaseline(ns["stats"]), RuleZ(ns["feature_normal_mean"],
                                              ns["feature_normal_std"])
    out_b, out_z, masks = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        start = int(z["feat_start"])
        tb = {k[3:]: z[k] for k in z.files if k.startswith("tb_")}
        out_b.append(np.array([rb.s_raw(tick_sample(tb, i))
                               for i in range(start, tb["t"].size)]))
        out_z.append(np.array([rz.s_raw(x) for x in z["features"]]))
        masks.append(z["masks"][start:])
    return np.vstack(out_b), np.vstack(out_z), np.vstack(masks)


def train_mlp(Xtr, ytr, Xva, yva, seed, epochs, dev):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, len(P.CHANNELS)), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(yva, dtype=torch.float32, device=dev)

    def loss_fn(pred, target):
        w = 1.0 + 4.0 * (1.0 - target)  # emphasize degraded samples
        return ((pred - target) ** 2 * w).mean()

    best, best_state, patience = np.inf, None, 0
    g = torch.Generator(device=dev).manual_seed(seed)
    n = Xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=g)
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xv), yv))
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def train_autoencoder(X_normal, seed, epochs, dev):
    """Plan 11.4: undercomplete autoencoder on normal-only windowed features;
    reconstruction error is the anomaly score. Compared against, never
    deployed (plan 11.5: MLP is the primary candidate)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = X_normal.shape[1]
    ae = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(),
                       nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, d)).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xt = torch.tensor(X_normal, device=dev)
    n = Xt.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev, generator=g)
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            ((ae(Xt[b]) - Xt[b]) ** 2).mean().backward()
            opt.step()
    return ae


def autoencoder_scores(ae, X, dev):
    import torch
    with torch.no_grad():
        recon = ae(torch.tensor(X, device=dev)).cpu().numpy()
    return ((X - recon) ** 2).mean(1)


def export(model, n_in, out_dir, meta):
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model.cpu(), torch.zeros(1, n_in), str(out_dir / "reliability.onnx"),
                      input_names=["features"], output_names=["s"],
                      dynamic_axes={"features": {0: "batch"}}, opset_version=18,
                      external_data=False)
    json.dump(meta, open(out_dir / "stats.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-dir", default="asr_reliability_estimator/model")
    ap.add_argument("--models-out", default="data/models")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--holdout", default=None, help="world name to hold out as test")
    args = ap.parse_args()
    data = pathlib.Path(args.data)
    tag = f"holdout_{args.holdout}" if args.holdout else "main"
    split = make_split(data, args.holdout)
    eps = episode_files(data, split)
    ns = normal_stats(eps["train"])
    sigma = np.asarray(ns["label_sigma"])
    print(f"[{tag}] runs train/val/test = "
          f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])}")

    Xtr, ytr, mtr = stack(eps["train"], sigma)
    Xva, yva, mva = stack(eps["val"], sigma) if eps["val"] else (Xtr, ytr, mtr)
    Xte, yte, mte = stack(eps["test"], sigma)
    print(f"samples train {Xtr.shape} val {Xva.shape} test {Xte.shape}")
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: ((X - mean) / std).astype(np.float32)
    results = {"tag": tag, "split": split}

    # --- A. rule baselines (plain + strong windowed z-score) -----------------
    s_rule, s_rz, m_rule = rule_series(eps["test"], ns)
    results["rule"] = eval_channels("rule", s_rule, m_rule, yte)
    results["rule_z"] = eval_channels("rule_z", s_rz, m_rule, yte)

    # --- B. Isolation Forest (global anomaly score) --------------------------
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=200, random_state=0).fit(norm(Xtr[~mtr.any(1)]))
    results["isolation_forest"] = {
        "any_fault_auroc": round(auroc(-iso.score_samples(norm(Xte)), mte.any(1)), 3)}
    print(f"[iforest] any-fault AUROC = {results['isolation_forest']['any_fault_auroc']}")

    # --- C. MLP regression over seeds (primary, plan 11.3) --------------------
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta_base = {"stats": ns["stats"], "feature_mean": mean.tolist(),
                 "feature_std": std.tolist(), "label_sigma": sigma.tolist(),
                 "feature_normal_mean": ns["feature_normal_mean"],
                 "feature_normal_std": ns["feature_normal_std"],
                 "channels": P.CHANNELS, "feature_names": FEATURE_NAMES,
                 "split": split, "tag": tag}
    per_seed, best_seed, best_val = [], None, np.inf
    for seed in args.seeds:
        t0 = time.time()
        model, vl, n_ep = train_mlp(norm(Xtr), ytr, norm(Xva), yva, seed, args.epochs, dev)
        with torch.no_grad():
            s_mlp = model(torch.tensor(norm(Xte), device=dev)).cpu().numpy()
        r = eval_channels(f"mlp seed {seed}", s_mlp, mte, yte)
        r["val_loss"] = round(vl, 5)
        r["epochs"] = n_ep
        r["label_rmse"] = round(float(np.sqrt(((s_mlp - yte) ** 2).mean())), 4)
        per_seed.append(r)
        out_dir = pathlib.Path(args.models_out) / tag / f"seed_{seed}"
        export(model, Xtr.shape[1], out_dir, {**meta_base, "seed": seed})
        print(f"  seed {seed}: val {vl:.5f} after {n_ep} ep, {time.time() - t0:.0f}s -> {out_dir}")
        if vl < best_val:
            best_val, best_seed = vl, seed
    results["mlp_seeds"] = dict(zip(map(str, args.seeds), per_seed))
    keys = ("f1", "auroc", "f1_gt", "auroc_gt")
    results["mlp_mean"] = {c: {k: round(float(np.mean([r[c][k] for r in per_seed
                                                       if r[c][k] is not None])), 3)
                               for k in keys} for c in P.CHANNELS}
    results["mlp_std"] = {c: {k: round(float(np.std([r[c][k] for r in per_seed
                                                     if r[c][k] is not None])), 3)
                              for k in keys} for c in P.CHANNELS}
    print("[mlp mean±std] " + "  ".join(
        f"{c}: f1={results['mlp_mean'][c]['f1']}±{results['mlp_std'][c]['f1']}"
        for c in P.CHANNELS))

    # --- D. Autoencoder (comparison-only, plan 11.4/11.5) ---------------------
    ae = train_autoencoder(norm(Xtr[~mtr.any(1)]), args.seeds[0], args.epochs, dev)
    err_te = autoencoder_scores(ae, norm(Xte), dev)
    err_normal_tr = autoencoder_scores(ae, norm(Xtr[~mtr.any(1)]), dev)
    # calibration (plan 11.4: reconstruction error has no physical unit) --
    # same exp(-.) shape as the ground-truth labels, scaled so a typical
    # normal-driving reconstruction error maps to a score near the AUROC
    # ranking is calibration-invariant; the threshold-based F1 is not, so
    # both are reported.
    scale = np.percentile(err_normal_tr, 95) / 1.959964
    ae_score = np.exp(-err_te / max(scale, 1e-9))
    any_fault = mte.any(1)
    det = ae_score < 0.5
    tp, fp, fn = (det & any_fault).sum(), (det & ~any_fault).sum(), (~det & any_fault).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    results["autoencoder"] = {
        "any_fault_auroc": round(auroc(err_te, any_fault), 3),
        "any_fault_f1_at_0.5": (round(2 * prec * rec / (prec + rec), 3)
                                if np.isfinite(prec) and np.isfinite(rec) and (prec or rec)
                                else 0.0),
        "recon_error_p95_normal": round(float(np.percentile(err_normal_tr, 95)), 5),
    }
    print(f"[autoencoder] any-fault AUROC = {results['autoencoder']['any_fault_auroc']}  "
          f"F1@0.5 = {results['autoencoder']['any_fault_f1_at_0.5']}")

    # --- deploy best seed + inference benchmark (CPU realtime constraint) ---
    if not args.holdout:
        src = pathlib.Path(args.models_out) / tag / f"seed_{best_seed}"
        mdir = pathlib.Path(args.model_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        for name in ("reliability.onnx", "stats.json"):
            shutil.copy(src / name, mdir / name)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(mdir / "reliability.onnx"),
                                    providers=["CPUExecutionProvider"])
        x1 = norm(Xte[:1])
        for _ in range(50):
            sess.run(None, {"features": x1})
        t0 = time.perf_counter()
        for _ in range(1000):
            sess.run(None, {"features": x1})
        us = (time.perf_counter() - t0) * 1e3
        results["onnx_inference_us"] = round(us, 1)
        results["deployed_seed"] = best_seed
        print(f"deployed seed {best_seed} -> {mdir}; ONNX inference {us:.0f} us/sample")

    json.dump(results, open(data / f"train_results_{tag}.json", "w"), indent=1,
              default=str)


if __name__ == "__main__":
    main()
