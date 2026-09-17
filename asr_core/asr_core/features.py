"""Temporal feature builder (plan section 9).

Consumes tick-rate samples (already time-synchronized / interpolated by the
caller: the reliability node online, the dataset builder offline) and emits a
fixed-order feature vector once the window is full.
"""
from collections import deque

import numpy as np

RESIDUALS = {
    "r_v_EL": ("v_enc", "v_lid"),
    "r_w_EL": ("w_enc", "w_lid"),
    "r_w_EI": ("w_enc", "w_imu"),
    "r_w_LI": ("w_lid", "w_imu"),
}
INNOVATIONS = ["nu_v_enc", "nu_w_enc", "nu_v_lid", "nu_w_lid", "nu_w_imu"]
NIS = ["nis_enc", "nis_lid", "nis_imu"]
RAW = ["v_enc", "w_enc", "w_imu", "ax_imu", "v_lid", "w_lid"]

# Plan section 9: the model sees sensor *relationships* only — residuals,
# shadow innovations / NIS, lidar match quality, stationary flag. Raw
# velocities are deliberately excluded: an inflated encoder reading above the
# platform's top speed would otherwise be a simulation-only shortcut.
FEATURE_NAMES = (
    [f"{r}_{s}" for r in RESIDUALS for s in ("mean", "std", "absmean", "slope")]
    + [f"{n}_mean" for n in INNOVATIONS]
    # NIS is chi-square distributed and explodes under faults (1e5+); window
    # stats are taken on log1p(NIS) so the model sees a well-conditioned scale
    + [f"log_{n}_{s}" for n in NIS for s in ("mean", "max", "var")]
    + ["lid_rms_mean", "lid_ratio_mean", "lid_degen_mean", "lid_age", "stationary"]
)

SAMPLE_KEYS = RAW + INNOVATIONS + NIS + ["lid_rms", "lid_ratio", "lid_degen",
                                         "lid_age", "stationary"]


class FeatureBuilder:
    def __init__(self, tick_rate, window_sec):
        self.n = max(2, int(round(tick_rate * window_sec)))
        self.dt = 1.0 / tick_rate
        self.buf = {k: deque(maxlen=self.n) for k in SAMPLE_KEYS}
        for name, (a, b) in RESIDUALS.items():
            self.buf[name] = deque(maxlen=self.n)
        # precomputed abscissa for the slope fit
        self._x = np.arange(self.n) * self.dt
        self._x = self._x - self._x.mean()
        self._xx = float((self._x ** 2).sum())

    def push(self, sample):
        """sample: dict with SAMPLE_KEYS. Returns feature vector or None
        while the window is still filling."""
        for k in SAMPLE_KEYS:
            self.buf[k].append(float(sample[k]))
        for name, (a, b) in RESIDUALS.items():
            self.buf[name].append(float(sample[a]) - float(sample[b]))
        if len(self.buf["v_enc"]) < self.n:
            return None
        f = []
        for name in RESIDUALS:
            r = np.asarray(self.buf[name])
            f += [r.mean(), r.std(), np.abs(r).mean(),
                  float((self._x * (r - r.mean())).sum() / self._xx)]
        for name in INNOVATIONS:
            f.append(np.mean(self.buf[name]))
        for name in NIS:
            a = np.log1p(np.maximum(np.asarray(self.buf[name]), 0.0))
            f += [a.mean(), a.max(), a.var()]
        f += [np.mean(self.buf["lid_rms"]), np.mean(self.buf["lid_ratio"]),
              np.mean(self.buf["lid_degen"]),
              self.buf["lid_age"][-1], self.buf["stationary"][-1]]
        return np.asarray(f, float)


class StationaryDetector:
    """Plan section 10: sustained low-motion condition.

    Gazebo's simulated IMU accelerometer is far noisier per-sample than a
    real one (std ~1-2 m/s^2 at rest here) — thresholding the raw ||a|| - g
    reading directly almost never fires. a_norm is smoothed with a short EMA
    (time constant ~hold_sec/3) before the threshold check; v_enc/w_imu need
    no smoothing since wheel/gyro readings are already clean at rest.
    """

    def __init__(self, v_thr, w_thr, a_thr, hold_sec, tick_rate, g=9.81):
        self.v_thr, self.w_thr, self.a_thr, self.g = v_thr, w_thr, a_thr, g
        self.need = max(1, int(round(hold_sec * tick_rate)))
        self.count = 0
        self.a_alpha = 1.0 - np.exp(-3.0 / max(self.need, 1))  # ~hold/3 EMA
        self.a_ema = g

    def step(self, v_enc, w_imu, a_norm):
        self.a_ema += self.a_alpha * (a_norm - self.a_ema)
        if (abs(v_enc) < self.v_thr and abs(w_imu) < self.w_thr
                and abs(self.a_ema - self.g) < self.a_thr):
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.need
