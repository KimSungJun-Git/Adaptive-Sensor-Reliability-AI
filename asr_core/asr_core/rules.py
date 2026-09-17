"""Rule-based baselines (plan section 11.1 / comparison group B) and the
recovery-gate check shared with the AI path (plan section 14.3).

Two rule variants are kept so the AI comparison is honest:

RuleBaseline  instantaneous 3-sigma gates on pairwise residuals (the plain
              "|residual| > threshold" of plan 11.1)
RuleZ         the strong baseline: z-score gates on the *same windowed
              feature vector the AI sees* (residual abs-mean, log-NIS mean,
              innovation mean), with NIS-attributed DOF flagging. If the AI
              cannot beat this, it is not needed.
"""
import numpy as np

from .features import FEATURE_NAMES
from .params import GATE_NIS_MAX, RULE_INNOV_SIGMA, RULE_RESID_SIGMA, RULE_S_LOW

_FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


# residual keys as produced from a tick sample
def _residuals(sample):
    return {
        "r_v_EL": sample["v_enc"] - sample["v_lid"],
        "r_w_EL": sample["w_enc"] - sample["w_lid"],
        "r_w_EI": sample["w_enc"] - sample["w_imu"],
        "r_w_LI": sample["w_lid"] - sample["w_imu"],
    }


def _vote(p, enc_v_score, lid_v_score, nis_bad=None, nu_z=None):
    """Shared majority-vote logic. p: dict of residual-abnormal flags.
    Returns bad flags in CHANNELS order."""
    enc_w = p["r_w_EL"] and p["r_w_EI"]
    lid_w = p["r_w_EL"] and p["r_w_LI"]
    imu_w = p["r_w_EI"] and p["r_w_LI"]
    enc_v = lid_v = False
    if p["r_v_EL"]:
        enc_v, lid_v = enc_v_score >= lid_v_score, lid_v_score > enc_v_score
    if nis_bad:  # NIS gating: abnormal sensor -> flag its worst DOF
        if nis_bad["encoder"]:
            if nu_z["nu_v_enc"] >= nu_z["nu_w_enc"]:
                enc_v = True
            else:
                enc_w = True
        if nis_bad["lidar"]:
            if nu_z["nu_v_lid"] >= nu_z["nu_w_lid"]:
                lid_v = True
            else:
                lid_w = True
        if nis_bad["imu"]:
            imu_w = True
    return [enc_v, enc_w, lid_v, lid_w, imu_w]


class RuleBaseline:
    """Instantaneous residual gates; v-channel tie broken by the larger
    normalized shadow innovation."""

    def __init__(self, stats, k_resid=RULE_RESID_SIGMA, k_innov=RULE_INNOV_SIGMA,
                 s_low=RULE_S_LOW):
        self.stats = stats
        self.k_r, self.k_i, self.s_low = k_resid, k_innov, s_low

    def s_raw(self, sample):
        st = self.stats
        p = {k: abs(v) > self.k_r * st[k] for k, v in _residuals(sample).items()}
        ne = abs(sample["nu_v_enc"]) / max(st["nu_v_enc"], 1e-9)
        nl = abs(sample["nu_v_lid"]) / max(st["nu_v_lid"], 1e-9)
        return np.where(_vote(p, ne, nl), self.s_low, 1.0)


class RuleZ:
    """Windowed z-score rule on the AI feature vector (strong baseline)."""

    def __init__(self, feature_normal_mean, feature_normal_std, k=3.0,
                 s_low=RULE_S_LOW):
        self.mu = np.asarray(feature_normal_mean, float)
        self.sd = np.maximum(np.asarray(feature_normal_std, float), 1e-9)
        self.k, self.s_low = k, s_low

    def s_raw(self, feat):
        z = (np.asarray(feat, float) - self.mu) / self.sd
        hi = lambda name: z[_FI[name]] > self.k
        p = {r: hi(f"{r}_absmean") for r in ("r_v_EL", "r_w_EL", "r_w_EI", "r_w_LI")}
        nis_bad = {"encoder": hi("log_nis_enc_mean"), "lidar": hi("log_nis_lid_mean"),
                   "imu": hi("log_nis_imu_mean")}
        nu_z = {n: abs(z[_FI[f"{n}_mean"]]) for n in
                ("nu_v_enc", "nu_w_enc", "nu_v_lid", "nu_w_lid", "nu_w_imu")}
        return np.where(_vote(p, nu_z["nu_v_enc"], nu_z["nu_v_lid"], nis_bad, nu_z),
                        self.s_low, 1.0)


def gates_ok(sample, stats, k=RULE_RESID_SIGMA):
    """Recovery gate per channel: every residual touching the channel is
    normal AND the sensor NIS is normal (plan 14.3). Returns (5,) bool in
    CHANNELS order."""
    r = _residuals(sample)
    ok = {k_: abs(v) <= k * stats[k_] for k_, v in r.items()}
    nis_ok = {
        "encoder": sample["nis_enc"] <= GATE_NIS_MAX["encoder"],
        "lidar": sample["nis_lid"] <= GATE_NIS_MAX["lidar"],
        "imu": sample["nis_imu"] <= GATE_NIS_MAX["imu"],
    }
    return np.array([
        ok["r_v_EL"] and nis_ok["encoder"],                     # v_encoder
        ok["r_w_EL"] and ok["r_w_EI"] and nis_ok["encoder"],    # omega_encoder
        ok["r_v_EL"] and nis_ok["lidar"],                       # v_lidar
        ok["r_w_EL"] and ok["r_w_LI"] and nis_ok["lidar"],      # omega_lidar
        ok["r_w_EI"] and ok["r_w_LI"] and nis_ok["imu"],        # omega_imu
    ])


def residual_stat_keys():
    return ["r_v_EL", "r_w_EL", "r_w_EI", "r_w_LI",
            "nu_v_enc", "nu_w_enc", "nu_v_lid", "nu_w_lid", "nu_w_imu"]
