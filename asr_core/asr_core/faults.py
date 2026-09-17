"""Fault injection (plan section 13).

Offline, vectorized: applied to extracted run arrays when building the
training / evaluation datasets. Magnitudes are deliberately subtle — the goal
is faults a plain threshold struggles with, not `encoder = 9999`.

The online demo injector node reuses the same formulas per-message for the
demo subset (slip / drift / imu bias / lidar sector).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FaultEvent:
    sensor: str            # encoder | imu | lidar
    ftype: str
    t0: float
    t1: float
    mag: float = 0.0
    channel: str = "v"     # encoder/imu channel-level faults: 'v' | 'omega'
    seed: int = 0

    def window(self, t):
        return (t >= self.t0) & (t < self.t1)


def _chan(v, w, channel):
    return v if channel == "v" else w


def apply_encoder(t, v, w, events):
    """Returns (v', w', mask_v, mask_w). Slip is symmetric wheel spin:
    v inflated, omega untouched (plan section 7.2)."""
    v, w = v.copy(), w.copy()
    mv = np.zeros(t.size, bool)
    mw = np.zeros(t.size, bool)
    for e in events:
        if e.sensor != "encoder":
            continue
        win = e.window(t)
        rng = np.random.default_rng(e.seed)
        x = _chan(v, w, e.channel)
        if e.ftype == "bias":
            x[win] += e.mag
        elif e.ftype == "scale":
            x[win] *= 1.0 + e.mag
        elif e.ftype == "drift":
            x[win] += e.mag * (t[win] - e.t0)
        elif e.ftype == "spike":
            hits = win & (rng.random(t.size) < 0.06)
            x[hits] += rng.choice([-1.0, 1.0], hits.sum()) * e.mag
        elif e.ftype == "stuck":
            idx = np.flatnonzero(win)
            if idx.size:
                x[idx] = x[idx[0]]
        elif e.ftype == "noise":
            x[win] += rng.normal(0.0, e.mag, win.sum())
        elif e.ftype == "slip":
            v[win] *= 1.0 + e.mag
        elif e.ftype == "quant":
            x[win] = np.round(x[win] / e.mag) * e.mag
        else:
            raise ValueError(f"unknown encoder fault {e.ftype}")
        if e.ftype == "slip" or e.channel == "v":
            mv[win] = True
        else:
            mw[win] = True
    return v, w, mv, mw


def apply_imu(t, w, ax, events):
    """Returns (w', ax', mask_w)."""
    w, ax = w.copy(), ax.copy()
    mw = np.zeros(t.size, bool)
    for e in events:
        if e.sensor != "imu":
            continue
        win = e.window(t)
        rng = np.random.default_rng(e.seed)
        if e.ftype == "bias":
            w[win] += e.mag
        elif e.ftype == "bias_drift":
            w[win] += e.mag * (t[win] - e.t0)
        elif e.ftype == "noise":
            w[win] += rng.normal(0.0, e.mag, win.sum())
        elif e.ftype == "spike":
            hits = win & (rng.random(t.size) < 0.06)
            w[hits] += rng.choice([-1.0, 1.0], hits.sum()) * e.mag
        elif e.ftype == "dropout":
            idx = np.flatnonzero(win)
            if idx.size and idx[0] > 0:
                w[idx] = w[idx[0] - 1]
                ax[idx] = ax[idx[0] - 1]
        else:
            raise ValueError(f"unknown imu fault {e.ftype}")
        mw[win] = True
    return w, ax, mw


def apply_lidar_scans(scan_t, scans, events):
    """scans: list of range arrays (same length as scan_t).
    Returns (keep_mask, scans', fault_mask_per_scan)."""
    keep = np.ones(len(scans), bool)
    mask = np.zeros(len(scans), bool)
    out = [s.copy() for s in scans]
    for e in events:
        if e.sensor != "lidar":
            continue
        win = e.window(np.asarray(scan_t))
        rng = np.random.default_rng(e.seed)
        for i in np.flatnonzero(win):
            r = out[i]
            if e.ftype == "dropout":
                if rng.random() < e.mag:
                    keep[i] = False
            elif e.ftype == "range_noise":
                out[i] = r + rng.normal(0.0, e.mag, r.size)
            elif e.ftype == "sector":
                n = r.size
                w = int(n * min(max(e.mag, 0.0), 1.0))
                start = rng.integers(0, n)
                idx = (start + np.arange(w)) % n
                r2 = r.copy()
                r2[idx] = np.inf
                out[i] = r2
            else:
                raise ValueError(f"unknown lidar fault {e.ftype}")
        mask |= win
    return keep, out, mask


def channel_masks(t, events):
    """Ground-truth fault activity per (sensor x DOF) channel for detection
    metrics, aligned with params.CHANNELS order."""
    from .params import CHANNELS
    m = {c: np.zeros(t.size, bool) for c in CHANNELS}
    for e in events:
        win = e.window(t)
        if e.sensor == "encoder":
            ch = "v_encoder" if (e.channel == "v" or e.ftype == "slip") else "omega_encoder"
            m[ch] |= win
        elif e.sensor == "imu":
            m["omega_imu"] |= win
        elif e.sensor == "lidar":
            m["v_lidar"] |= win
            m["omega_lidar"] |= win
    return np.column_stack([m[c] for c in CHANNELS])


# --- scenario library (plan section 19) -------------------------------------
def scenario_events(name, t_end, seed=0, window=None):
    """Standard experiment scenarios. Fault window sits mid-run by default;
    `window=(t0, t1)` overrides it (used by the dedicated stationary
    scenarios, which target a run's actual stationary segment instead)."""
    a, b = window if window is not None else (0.25 * t_end, 0.60 * t_end)
    S = {
        "normal": [],
        "slip": [FaultEvent("encoder", "slip", a, b, 0.8, seed=seed)],
        "enc_drift": [FaultEvent("encoder", "drift", a, b, 0.012, "v", seed)],
        "enc_bias": [FaultEvent("encoder", "bias", a, b, 0.06, "v", seed)],
        "enc_noise": [FaultEvent("encoder", "noise", a, b, 0.08, "v", seed)],
        "enc_stuck": [FaultEvent("encoder", "stuck", a, b, 0.0, "v", seed)],
        "enc_scale": [FaultEvent("encoder", "scale", a, b, 0.12, "v", seed)],
        "enc_spike": [FaultEvent("encoder", "spike", a, b, 0.15, "v", seed)],
        "enc_quant": [FaultEvent("encoder", "quant", a, b, 0.05, "v", seed)],
        # omega-channel encoder faults (asymmetric wheel error) so that
        # s_omega_encoder has positive examples too
        "enc_w_bias": [FaultEvent("encoder", "bias", a, b, 0.12, "omega", seed)],
        "enc_w_drift": [FaultEvent("encoder", "drift", a, b, 0.02, "omega", seed)],
        "enc_w_noise": [FaultEvent("encoder", "noise", a, b, 0.15, "omega", seed)],
        "imu_bias": [FaultEvent("imu", "bias", a, b, 0.15, "omega", seed)],
        "imu_drift": [FaultEvent("imu", "bias_drift", a, b, 0.01, "omega", seed)],
        "imu_noise": [FaultEvent("imu", "noise", a, b, 0.12, "omega", seed)],
        "imu_spike": [FaultEvent("imu", "spike", a, b, 0.30, "omega", seed)],
        "imu_dropout": [FaultEvent("imu", "dropout", a, b, 0.0, "omega", seed)],
        "lidar_sector": [FaultEvent("lidar", "sector", a, b, 0.70, seed=seed)],
        "lidar_noise": [FaultEvent("lidar", "range_noise", a, b, 0.15, seed=seed)],
        "lidar_dropout": [FaultEvent("lidar", "dropout", a, b, 0.8, seed=seed)],
        "combo": [FaultEvent("lidar", "sector", a, b, 0.70, seed=seed),
                  FaultEvent("encoder", "slip", (a + b) / 2, b, 0.8, seed=seed + 1)],
        # Plan section 19, experiment 7: fault injected into a genuine
        # stationary segment (not the default mid-run window) -- tests both
        # that the fault is still caught despite stationary damping, and
        # that surrounding stationary silence stays quiet. `window` is
        # required for these two; build_dataset locates it per run.
        "stationary_imu_bias": [FaultEvent("imu", "bias", a, b, 0.10, "omega", seed)],
        "stationary_enc_quant": [FaultEvent("encoder", "quant", a, b, 0.03, "v", seed)],
    }
    return S[name]


def find_stationary_window(gt_t, gt_v, gt_w, min_dur=3.0, v_thr=0.02, w_thr=0.03):
    """Longest continuous run of near-zero ground-truth motion, for the
    dedicated stationary scenarios. Returns (t0, t1) or None if no segment
    of at least `min_dur` seconds exists in this run."""
    still = (np.abs(gt_v) < v_thr) & (np.abs(gt_w) < w_thr)
    d = np.diff(np.concatenate([[0], still.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if not len(starts):
        return None
    durations = gt_t[np.minimum(ends, len(gt_t) - 1)] - gt_t[starts]
    i = int(np.argmax(durations))
    if durations[i] < min_dur:
        return None
    t0, t1 = gt_t[starts[i]], gt_t[np.minimum(ends[i], len(gt_t) - 1)]
    pad = 0.5  # keep the fault fully inside the still segment
    return (t0 + pad, t1 - pad)


SCENARIOS = ["normal", "slip", "enc_drift", "enc_bias", "enc_noise", "enc_stuck",
             "enc_scale", "enc_spike", "enc_quant",
             "enc_w_bias", "enc_w_drift", "enc_w_noise",
             "imu_bias", "imu_drift", "imu_noise", "imu_spike", "imu_dropout",
             "lidar_sector", "lidar_noise", "lidar_dropout", "combo"]

# subset of SCENARIOS whose fault window must be a genuine stationary segment
# rather than the default mid-run window; skipped for a run if none is found
STATIONARY_SCENARIOS = ["stationary_imu_bias", "stationary_enc_quant"]
