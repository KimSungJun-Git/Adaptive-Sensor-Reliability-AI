"""Ground-truth based reliability labels (plan section 12).

s = exp(-e^2 / (2 sigma_eff^2)), sigma_eff = LABEL_SIGMA_FACTOR * sigma_normal,
with sigma_normal estimated per (sensor x DOF) channel from normal runs.
"""
import numpy as np

from .params import CHANNELS, LABEL_SIGMA_FACTOR


def channel_errors(meas, gt_v, gt_w):
    """meas: dict with v_enc, w_enc, v_lid, w_lid, w_imu arrays on the tick
    timeline. Returns (n_ticks, 5) absolute errors in CHANNELS order."""
    e = {
        "v_encoder": np.abs(meas["v_enc"] - gt_v),
        "omega_encoder": np.abs(meas["w_enc"] - gt_w),
        "v_lidar": np.abs(meas["v_lid"] - gt_v),
        "omega_lidar": np.abs(meas["w_lid"] - gt_w),
        "omega_imu": np.abs(meas["w_imu"] - gt_w),
    }
    return np.column_stack([e[c] for c in CHANNELS])


P95_TO_SIGMA = 1.0 / 1.959964  # |N(0,s)| 95th percentile = 1.96 s


def robust_sigma(x, axis=0):
    """Gaussian-equivalent sigma from the 95th percentile of |x|. Normal-run
    errors are heavy-tailed (bumps, corridor scan matching): RMS is dominated
    by that tail while the median is ~0 whenever the robot is slow, so both
    mis-state what "nominal" means. p95 tracks the core of the distribution."""
    return np.percentile(np.abs(np.asarray(x, float)), 95, axis=axis) * P95_TO_SIGMA + 1e-6


def estimate_sigma(normal_errors):
    """normal_errors: (n, 5) stacked over all normal runs -> per-channel
    label sigma. (R0 for the EKF uses the conservative RMS instead, params.)"""
    return robust_sigma(normal_errors, axis=0)


def make_labels(errors, sigma, factor=LABEL_SIGMA_FACTOR):
    sig = np.maximum(np.asarray(sigma, float) * factor, 1e-6)
    return np.exp(-(errors ** 2) / (2.0 * sig ** 2))
