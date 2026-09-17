"""Offline replay pipeline shared by dataset building, training and the
Fixed / Rule / AI EKF comparison (plan sections 17, 20).

A `run` is a dict of raw streams extracted from a rosbag:
    t_enc, v_enc, w_enc          encoder twist (native rate)
    t_imu, w_imu, ax_imu         imu (subsampled ~50 Hz)
    scan_t, scans, angle_min, angle_increment, range_min, range_max
    t_gt, gt_x, gt_y, gt_yaw, gt_v, gt_w
Everything downstream (shadow EKF, ticks, features, adaptive EKF) is
recomputed here so fault injection on the raw streams propagates exactly as
it would online.
"""
import numpy as np

from . import params as P
from .ekf import EKF
from .features import FeatureBuilder, StationaryDetector
from .icp import LidarOdometry
from .smoothing import ReliabilityFilter, inflate


def lidar_odometry_offline(run):
    lo = LidarOdometry(run.get("range_min", 0.12), run.get("range_max", 3.4))
    rows = []
    for t, r in zip(run["scan_t"], run["scans"]):
        out = lo.step(float(t), r, run["angle_min"], run["angle_increment"])
        if out is not None and out["valid"]:  # invalid matches never reach the EKF
            rows.append((t, out["v"], out["omega"], out["rms"], out["ratio"],
                         out["degen"]))
    if not rows:
        raise ValueError("no lidar odometry produced")
    a = np.array(rows)
    return {"t": a[:, 0], "v": a[:, 1], "w": a[:, 2], "rms": a[:, 3],
            "ratio": a[:, 4], "degen": a[:, 5]}


def _measurements(run, lid):
    """Sorted (t, sensor, z) measurement stream for EKF replay."""
    ms = []
    for t, v, w in zip(run["t_enc"], run["v_enc"], run["w_enc"]):
        ms.append((float(t), "encoder", (v, w)))
    for t, v, w in zip(lid["t"], lid["v"], lid["w"]):
        ms.append((float(t), "lidar", (v, w)))
    for t, w in zip(run["t_imu"], run["w_imu"]):
        ms.append((float(t), "imu", (w,)))
    ms.sort(key=lambda m: m[0])
    return ms


def run_ekf(run, lid, r_provider=None, log_updates=False):
    """Replay the EKF over the measurement stream.

    r_provider(sensor, t) -> (R_diag, keep_mask) or None for fixed nominal R.
    Returns state log (t, x[5]) and, if log_updates, per-sensor innovation logs.
    """
    ekf = EKF(P.EKF_Q_DIAG, P.EKF_P0_DIAG)
    logs = {s: [] for s in P.SENSORS}
    states = []
    for t, sensor, z in _measurements(run, lid):
        ekf.predict(t)
        idx = P.SENSOR_STATE_IDX[sensor]
        r0d = np.array([P.NOMINAL_STD[c] ** 2 for c in P.SENSOR_CHANNELS[sensor]])
        keep = np.ones(len(idx), bool)
        if r_provider is not None:
            rd, keep = r_provider(sensor, t)
        else:
            rd = r0d
        if keep.any():
            sel = np.flatnonzero(keep)
            nu, S, nis = ekf.update(np.asarray(z)[sel],
                                    [idx[i] for i in sel], np.diag(rd[sel]))
            if log_updates:
                full_nu = np.full(len(idx), np.nan)
                full_nu[sel] = nu
                logs[sensor].append((t, *full_nu, nis))
        states.append((t, *ekf.x))
    out = {"states": np.array(states)}
    if log_updates:
        for s in P.SENSORS:
            out[s] = np.array(logs[s]) if logs[s] else np.zeros((0, 2 + len(P.SENSOR_STATE_IDX[s])))
    return out


def _ffill_at(ticks, t_src, values, fill):
    """Last value of `values` at or before each tick (forward fill)."""
    j = np.searchsorted(t_src, ticks, side="right") - 1
    out = np.where(j >= 0, values[np.clip(j, 0, None)], fill)
    return out, j


def build_tick_table(run, lid, shadow):
    """Uniform tick timeline with interpolated sensors, held lidar odometry,
    forward-filled shadow innovations / NIS, GT and stationary flag."""
    t0 = max(run["t_enc"][0], run["t_imu"][0], lid["t"][0], run["t_gt"][0])
    t1 = min(run["t_enc"][-1], run["t_imu"][-1], lid["t"][-1], run["t_gt"][-1])
    ticks = np.arange(t0 + 0.2, t1, 1.0 / P.TICK_RATE)

    tb = {"t": ticks}
    tb["v_enc"] = np.interp(ticks, run["t_enc"], run["v_enc"])
    tb["w_enc"] = np.interp(ticks, run["t_enc"], run["w_enc"])
    tb["w_imu"] = np.interp(ticks, run["t_imu"], run["w_imu"])
    tb["ax_imu"] = np.interp(ticks, run["t_imu"], run["ax_imu"])
    tb["a_norm"] = np.interp(ticks, run["t_imu"], run["a_norm"]) if "a_norm" in run \
        else np.full(ticks.size, 9.81)

    lid_cols = {"v": ("v_lid", 0.0), "w": ("w_lid", 0.0), "rms": ("lid_rms", 9.9),
                "ratio": ("lid_ratio", 0.0), "degen": ("lid_degen", 0.0)}
    for k, (name, fill) in lid_cols.items():
        tb[name], j = _ffill_at(ticks, lid["t"], lid[k], fill)
    tb["lid_age"] = ticks - np.where(j >= 0, lid["t"][np.clip(j, 0, None)], ticks - 9.9)

    # shadow innovations / NIS, forward-filled per sensor
    nu_names = {"encoder": ["nu_v_enc", "nu_w_enc"], "lidar": ["nu_v_lid", "nu_w_lid"],
                "imu": ["nu_w_imu"]}
    nis_names = {"encoder": "nis_enc", "lidar": "nis_lid", "imu": "nis_imu"}
    for s in P.SENSORS:
        log = shadow[s]
        dof = len(P.SENSOR_STATE_IDX[s])
        if log.shape[0] == 0:
            for name in nu_names[s]:
                tb[name] = np.zeros(ticks.size)
            tb[nis_names[s]] = np.full(ticks.size, float(dof))
            continue
        for d, name in enumerate(nu_names[s]):
            col = np.nan_to_num(log[:, 1 + d])
            tb[name], _ = _ffill_at(ticks, log[:, 0], col, 0.0)
        tb[nis_names[s]], _ = _ffill_at(ticks, log[:, 0], log[:, -1], float(dof))

    # ground truth at ticks
    tb["gt_x"] = np.interp(ticks, run["t_gt"], run["gt_x"])
    tb["gt_y"] = np.interp(ticks, run["t_gt"], run["gt_y"])
    tb["gt_yaw"] = np.interp(ticks, run["t_gt"], np.unwrap(run["gt_yaw"]))
    tb["gt_v"] = np.interp(ticks, run["t_gt"], run["gt_v"])
    tb["gt_w"] = np.interp(ticks, run["t_gt"], run["gt_w"])

    det = StationaryDetector(P.STAT_V_THR, P.STAT_W_THR, P.STAT_A_THR,
                             P.STAT_HOLD_SEC, P.TICK_RATE)
    tb["stationary"] = np.array([
        det.step(tb["v_enc"][i], tb["w_imu"][i], tb["a_norm"][i])
        for i in range(ticks.size)], float)
    return tb


def tick_sample(tb, i):
    keys = ["v_enc", "w_enc", "w_imu", "ax_imu", "v_lid", "w_lid",
            "nu_v_enc", "nu_w_enc", "nu_v_lid", "nu_w_lid", "nu_w_imu",
            "nis_enc", "nis_lid", "nis_imu",
            "lid_rms", "lid_ratio", "lid_degen", "lid_age", "stationary"]
    return {k: tb[k][i] for k in keys}


def build_features(tb):
    """Returns (start_index, features (m, n_feat)) — features exist once the
    window is full."""
    fb = FeatureBuilder(P.TICK_RATE, P.WINDOW_SEC)
    rows, start = [], None
    for i in range(tb["t"].size):
        f = fb.push(tick_sample(tb, i))
        if f is not None:
            if start is None:
                start = i
            rows.append(f)
    return start, np.asarray(rows)


def smooth_reliability(tb, s_raw_ticks, stats, gates_fn):
    """Apply the asymmetric filter + recovery gate over the tick timeline.
    s_raw_ticks: (n_ticks, 5) raw model output (1.0 where undefined)."""
    rf = ReliabilityFilter(len(P.CHANNELS), P.GAMMA_FALL, P.GAMMA_RISE,
                           P.RECOVERY_TICKS, P.STATIONARY_FALL_LIMIT)
    out = np.ones_like(s_raw_ticks)
    for i in range(tb["t"].size):
        ok = gates_fn(tick_sample(tb, i), stats)
        out[i] = rf.step(s_raw_ticks[i], ok, bool(tb["stationary"][i]))
    return out


def run_adaptive(run, lid, tb, s_ticks, alpha=P.ALPHA_INFLATION,
                 s_reject=P.S_REJECT):
    """Adaptive EKF replay: R scaled by the (causal) smoothed reliability at
    the latest tick before each measurement. Returns pose at tick times."""
    ticks = tb["t"]
    ch_idx = {s: [P.CHANNELS.index(c) for c in P.SENSOR_CHANNELS[s]]
              for s in P.SENSORS}
    r0 = {s: np.array([P.NOMINAL_STD[c] ** 2 for c in P.SENSOR_CHANNELS[s]])
          for s in P.SENSORS}

    def r_provider(sensor, t):
        i = np.searchsorted(ticks, t, side="right") - 1
        if i < 0:
            return r0[sensor], np.ones(len(r0[sensor]), bool)
        s = s_ticks[i, ch_idx[sensor]]
        return inflate(r0[sensor], s, alpha), s > s_reject

    out = run_ekf(run, lid, r_provider=r_provider)
    st = out["states"]  # (m, 6): t, x, y, yaw, v, w
    pose = np.column_stack([
        np.interp(ticks, st[:, 0], st[:, 1]),
        np.interp(ticks, st[:, 0], st[:, 2]),
        np.interp(ticks, st[:, 0], np.unwrap(st[:, 3])),
    ])
    return pose
