"""Evaluation metrics (plan section 21): ATE / RPE / RMSE for trajectories,
precision / recall / F1 / AUROC / latency / recovery / false-alarm rate for
fault detection."""
import numpy as np
from scipy.stats import rankdata


def _align_first(pose):
    """Express trajectory relative to its first pose (n,3: x,y,yaw)."""
    x0, y0, th0 = pose[0]
    c, s = np.cos(-th0), np.sin(-th0)
    R = np.array([[c, -s], [s, c]])
    xy = (pose[:, :2] - [x0, y0]) @ R.T
    yaw = np.angle(np.exp(1j * (pose[:, 2] - th0)))
    return np.column_stack([xy, yaw])


def ate(gt_pose, est_pose):
    """Absolute trajectory error (position RMSE) after first-pose alignment."""
    g, e = _align_first(np.asarray(gt_pose)), _align_first(np.asarray(est_pose))
    return float(np.sqrt(((g[:, :2] - e[:, :2]) ** 2).sum(axis=1).mean()))


def rpe(gt_pose, est_pose, delta):
    """Relative pose error over a window of `delta` samples: translation RMSE
    and yaw RMSE of the relative-motion mismatch."""
    g, e = np.asarray(gt_pose), np.asarray(est_pose)
    n = len(g) - delta
    if n < 1:
        return float("nan"), float("nan")
    def rel(p):
        d_xy = p[delta:, :2] - p[:-delta, :2]
        c, s = np.cos(-p[:-delta, 2]), np.sin(-p[:-delta, 2])
        local = np.column_stack([c * d_xy[:, 0] - s * d_xy[:, 1],
                                 s * d_xy[:, 0] + c * d_xy[:, 1]])
        dyaw = np.angle(np.exp(1j * (p[delta:, 2] - p[:-delta, 2])))
        return local, dyaw
    (gl, gy), (el, ey) = rel(g), rel(e)
    t_rmse = float(np.sqrt(((gl - el) ** 2).sum(axis=1).mean()))
    y_rmse = float(np.sqrt((np.angle(np.exp(1j * (gy - ey))) ** 2).mean()))
    return t_rmse, y_rmse


def auroc(scores, labels):
    """Mann-Whitney AUROC; scores higher = more anomalous, labels bool."""
    labels = np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _segments(mask):
    m = np.asarray(mask, bool).astype(int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def saturation_frac(s, sat_s=0.05):
    """Fraction of ticks where the reliability is low enough that R sits
    within ~1% of R_max (plan section 21.4: covariance saturation
    frequency). inflate()'s f(s) = 1 + alpha*(1-s) reaches 99% of its max
    range 1+alpha at s <= 0.01; sat_s=0.05 is a slightly looser, still
    meaningfully "pinned at the ceiling" cutoff."""
    return float(np.mean(np.asarray(s) <= sat_s))


def detection_metrics(t, s, mask, thr=0.5, recover_s=0.8, recover_hold=1.0,
                      guard=5.0, exclude=None, sat_s=0.05):
    """Per-channel detection metrics. s: reliability series, mask: injected
    fault activity. Detection = s < thr. `exclude`: ticks where the sensor is
    known (from ground truth) to be genuinely wrong although no fault was
    injected (bumps, wheel slip against obstacles); they are left out of the
    false-alarm region and reported separately as far_clean."""
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    mask = np.asarray(mask, bool)
    det = s < thr
    tp = float((det & mask).sum())
    fp = float((det & ~mask).sum())
    fn = float((~det & mask).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")

    # event timing
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.05
    hold_n = max(1, int(round(recover_hold / dt)))
    latencies, recoveries = [], []
    for a, b in _segments(mask):
        hit = np.flatnonzero(det[a:b])
        latencies.append(t[a + hit[0]] - t[a] if hit.size else float("nan"))
        rec_t = float("nan")
        for i in range(b, len(s) - hold_n):
            if mask[i]:
                break
            if (s[i:i + hold_n] >= recover_s).all():
                rec_t = t[i] - t[b - 1]
                break
        recoveries.append(rec_t)

    # clean region = outside fault windows + guard seconds after each
    clean = ~mask.copy()
    for a, b in _segments(mask):
        g = b + int(round(guard / dt))
        clean[a:min(g, len(clean))] = False
    far = float(det[clean].mean()) if clean.any() else float("nan")
    crossings = np.abs(np.diff(det[clean].astype(int))).sum() if clean.sum() > 1 else 0
    minutes = max(clean.sum() * dt / 60.0, 1e-9)
    if exclude is not None:
        strict = clean & ~np.asarray(exclude, bool)
        far_clean = float(det[strict].mean()) if strict.any() else float("nan")
    else:
        far_clean = far

    return {
        "precision": prec, "recall": rec, "f1": f1,
        "auroc": auroc(1.0 - s, mask),
        "latency": float(np.nanmean(latencies)) if latencies else float("nan"),
        "recovery": float(np.nanmean(recoveries)) if recoveries else float("nan"),
        "far": far,
        "far_clean": far_clean,
        "chatter_per_min": float(crossings / minutes),
        "sat_frac": saturation_frac(s, sat_s),
        "n_events": len(latencies),
        "missed": int(np.sum(np.isnan(latencies))) if latencies else 0,
    }
