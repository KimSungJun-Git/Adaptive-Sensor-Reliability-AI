"""Asymmetric reliability smoothing + recovery gate (plan section 14) and
bounded covariance inflation (plan section 15)."""
import numpy as np


class ReliabilityFilter:
    """Fast attack (fall), gated slow release (rise), stationary damping."""

    def __init__(self, n_channels, gamma_fall, gamma_rise, recovery_ticks,
                 stationary_fall_limit):
        self.gf, self.gr = gamma_fall, gamma_rise
        self.need = recovery_ticks
        self.stat_limit = stationary_fall_limit
        self.s = np.ones(n_channels)
        self.ok = np.zeros(n_channels, int)

    def step(self, s_raw, gate_ok, stationary):
        """s_raw: raw AI output per channel; gate_ok: per-channel bool,
        residual+NIS+consistency normal this tick (recovery gate)."""
        s_raw = np.clip(np.asarray(s_raw, float), 0.0, 1.0)
        gate_ok = np.asarray(gate_ok, bool)
        self.ok = np.where(gate_ok, self.ok + 1, 0)
        falling = s_raw < self.s
        fall = self.gf * self.s + (1.0 - self.gf) * s_raw
        if stationary:  # plan section 10: limit reliability drop when parked
            fall = np.maximum(fall, self.s - self.stat_limit)
        rise = np.where(self.ok >= self.need,
                        self.gr * self.s + (1.0 - self.gr) * s_raw, self.s)
        self.s = np.clip(np.where(falling, fall, rise), 0.0, 1.0)
        return self.s.copy()


def inflate(r0_diag, s, alpha):
    """R_adapted = R0 * (1 + alpha * (1 - s)), bounded in [R0, R0*(1+alpha)]."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    f = 1.0 + alpha * (1.0 - s)
    return np.asarray(r0_diag, float) * np.clip(f, 1.0, 1.0 + alpha)
