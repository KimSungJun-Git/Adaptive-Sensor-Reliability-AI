"""Self-checks for asr_core: fails loudly if any core math breaks."""
import numpy as np

from asr_core import params as P
from asr_core.ekf import EKF
from asr_core.faults import (FaultEvent, apply_encoder, apply_imu,
                             channel_masks, find_stationary_window,
                             scenario_events, SCENARIOS, STATIONARY_SCENARIOS)
from asr_core.features import FeatureBuilder, StationaryDetector, FEATURE_NAMES
from asr_core.hybrid import combine as hybrid_combine
from asr_core.icp import icp
from asr_core.metrics import ate, detection_metrics, rpe, saturation_frac
from asr_core.smoothing import ReliabilityFilter, inflate


def test_ekf_tracks_and_nis_calibrated():
    rng = np.random.default_rng(0)
    # Q matched to the constant-velocity truth so NIS calibration is testable
    ekf = EKF([1e-6, 1e-6, 1e-6, 1e-4, 1e-4], P.EKF_P0_DIAG)
    v_true, w_true = 0.2, 0.3
    sig_v, sig_w = 0.02, 0.01
    nis_log = []
    for k in range(1, 400):
        t = k * 0.02
        ekf.predict(t)
        z = [v_true + rng.normal(0, sig_v), w_true + rng.normal(0, sig_w)]
        _, _, nis = ekf.update(z, [3, 4], np.diag([sig_v ** 2, sig_w ** 2]))
        nis_log.append(nis)
    assert abs(ekf.x[3] - v_true) < 0.02
    assert abs(ekf.x[4] - w_true) < 0.02
    assert 1.0 < np.mean(nis_log[50:]) < 3.5  # ~chi2(2) mean = 2


def test_icp_recovers_known_motion():
    rng = np.random.default_rng(1)
    # square room walls
    a = np.linspace(-1, 1, 200)
    pts = np.vstack([np.column_stack([a, np.ones_like(a)]),
                     np.column_stack([a, -np.ones_like(a)]),
                     np.column_stack([np.ones_like(a), a]),
                     np.column_stack([-np.ones_like(a), a])])
    dx, dy, dyaw = 0.05, 0.01, 0.04
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s], [s, c]])
    cur = (pts - [dx, dy]) @ R  # inverse transform: robot moved by (dx,dy,dyaw)
    cur += rng.normal(0, 0.002, cur.shape)
    ex, ey, eyaw, rms, ratio, degen = icp(pts, cur)
    assert abs(ex - dx) < 0.01 and abs(ey - dy) < 0.01 and abs(eyaw - dyaw) < 0.01
    assert ratio > 0.9 and rms < 0.02 and degen > 0.5  # square room: well conditioned
    # corridor: two parallel walls only -> translation along x unobservable
    wall = np.vstack([np.column_stack([a, np.ones_like(a)]),
                      np.column_stack([a, -np.ones_like(a)])])
    *_, degen_c = icp(wall, wall + rng.normal(0, 0.002, wall.shape))
    assert degen_c < 0.05


def test_stationary_detector_survives_imu_noise():
    # Gazebo's simulated accelerometer is noisy at rest (std ~1-2 m/s^2);
    # the raw ||a|| - g threshold alone would never latch True (regression:
    # a real Gazebo run showed 0% detection over an 8s known-stationary
    # window before the EMA smoothing was added).
    rng = np.random.default_rng(2)
    det = StationaryDetector(P.STAT_V_THR, P.STAT_W_THR, P.STAT_A_THR,
                             P.STAT_HOLD_SEC, P.TICK_RATE)
    hits = [det.step(0.0, 0.0, 9.81 + rng.normal(0, 1.3)) for _ in range(60)]
    assert hits[-1] is True  # settles True despite per-sample accel noise
    # moving robot (v clearly nonzero) must never be flagged stationary
    det2 = StationaryDetector(P.STAT_V_THR, P.STAT_W_THR, P.STAT_A_THR,
                              P.STAT_HOLD_SEC, P.TICK_RATE)
    moving = [det2.step(0.13, 0.0, 9.81 + rng.normal(0, 1.3)) for _ in range(60)]
    assert not any(moving)


def test_slip_only_hits_encoder_v():
    t = np.arange(0, 10, 0.05)
    v = np.full_like(t, 0.2)
    w = np.full_like(t, 0.1)
    ev = [FaultEvent("encoder", "slip", 3, 6, 0.8)]
    v2, w2, mv, mw = apply_encoder(t, v, w, ev)
    win = (t >= 3) & (t < 6)
    assert np.allclose(v2[win], 0.36) and np.allclose(v2[~win], 0.2)
    assert np.allclose(w2, w) and mv.any() and not mw.any()
    m = channel_masks(t, ev)
    assert m[:, 0].any() and not m[:, 1:].any()


def test_all_scenarios_build_without_error():
    # every named scenario, including the newly-wired unused fault types,
    # must produce a nonempty event list that apply_* accepts without raising
    t = np.arange(0, 30, 0.05)
    for sc in SCENARIOS[1:]:  # skip "normal" (empty by design)
        events = scenario_events(sc, t_end=30.0, seed=0)
        assert events, sc
        v, w, mv, mw = apply_encoder(t, np.full_like(t, 0.15), np.full_like(t, 0.1), events)
        wi, ax, mw2 = apply_imu(t, np.full_like(t, 0.1), np.full_like(t, 0.0), events)
        assert np.isfinite(v).all() and np.isfinite(w).all()
        assert np.isfinite(wi).all() and np.isfinite(ax).all()
        if any(e.sensor == "encoder" for e in events):
            assert mv.any() or mw.any(), sc
        if any(e.sensor == "imu" for e in events):
            assert mw2.any(), sc


def test_stationary_window_found_and_placed_correctly():
    # a run that sits still for the middle third, moves elsewhere
    t = np.arange(0, 30, 0.1)
    v = np.where((t > 10) & (t < 20), 0.0, 0.2)
    w = np.zeros_like(t)
    win = find_stationary_window(t, v, w, min_dur=3.0)
    assert win is not None
    assert 10.0 <= win[0] < win[1] <= 20.0  # inside the still segment, padded
    # a run that never stops has no window
    assert find_stationary_window(t, np.full_like(t, 0.2), w, min_dur=3.0) is None
    for sc in STATIONARY_SCENARIOS:
        events = scenario_events(sc, t_end=30.0, seed=0, window=win)
        assert events and events[0].t0 == win[0] and events[0].t1 == win[1]


def test_hybrid_uses_rule_for_lidar_only():
    # CHANNELS order: v_encoder, omega_encoder, v_lidar, omega_lidar, omega_imu
    # Rule's raw output is binary (1.0 fine / 0.2 bad); hybrid sharpens a
    # "bad" flag on the LiDAR channels so it can cross S_REJECT (regression:
    # a soft 0.2 floor never rejects, which blew up one hard episode 6x).
    s_ai = np.array([0.9, 0.8, 0.9, 0.8, 0.9])
    s_rule = np.array([1.0, 1.0, 0.2, 0.2, 1.0])
    out = hybrid_combine(s_ai, s_rule)
    np.testing.assert_allclose(out, [0.9, 0.8, 0.02, 0.02, 0.9])
    # LiDAR channels always come from Rule (never blended with AI): Rule
    # judging them fine reads as 1.0, not as a pass-through of AI's opinion
    s_rule_fine = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(hybrid_combine(s_ai, s_rule_fine),
                               [0.9, 0.8, 1.0, 1.0, 0.9])
    # batched (n_ticks, 5) form must broadcast the same way
    batch = hybrid_combine(np.tile(s_ai, (3, 1)), np.tile(s_rule, (3, 1)))
    assert batch.shape == (3, 5)
    np.testing.assert_allclose(batch[1], out)


def test_saturation_frac():
    assert saturation_frac(np.array([0.0, 0.0, 1.0, 1.0])) == 0.5
    assert saturation_frac(np.ones(10)) == 0.0


def test_smoothing_fast_fall_gated_rise():
    rf = ReliabilityFilter(1, P.GAMMA_FALL, P.GAMMA_RISE, 5, 0.02)
    for _ in range(10):
        s = rf.step([0.1], [False], False)
    assert s[0] < 0.15  # fast attack
    for i in range(4):
        s = rf.step([1.0], [True], False)
    held = s[0]
    assert held < 0.15  # held until gate satisfied
    for _ in range(60):
        s = rf.step([1.0], [True], False)
    assert s[0] > 0.9  # recovered


def test_inflation_bounded():
    r = inflate([0.01], [0.0], alpha=49.0)
    assert np.isclose(r[0], 0.5)
    assert np.isclose(inflate([0.01], [1.0], 49.0)[0], 0.01)
    # out-of-range s (e.g. the covariance injector's unused IMU v-channel
    # sentinel, -1.0) must clip to a bound, not propagate a negative or
    # huge multiplier -- regression once caught in a duplicate, since-
    # removed reimplementation of this exact formula.
    assert np.isclose(inflate([0.01], [-1.0], 49.0)[0], 0.5)
    assert np.isclose(inflate([0.01], [2.0], 49.0)[0], 0.01)


def test_features_shape():
    fb = FeatureBuilder(20.0, 1.0)
    sample = {k: 0.0 for k in
              ["v_enc", "w_enc", "w_imu", "ax_imu", "v_lid", "w_lid",
               "nu_v_enc", "nu_w_enc", "nu_v_lid", "nu_w_lid", "nu_w_imu",
               "nis_enc", "nis_lid", "nis_imu", "lid_rms", "lid_ratio",
               "lid_degen", "lid_age", "stationary"]}
    out = None
    for _ in range(20):
        out = fb.push(sample)
    assert out is not None and out.size == len(FEATURE_NAMES)


def test_metrics_sane():
    pose = np.column_stack([np.linspace(0, 5, 100), np.zeros(100), np.zeros(100)])
    assert ate(pose, pose) < 1e-9
    assert rpe(pose, pose, 20)[0] < 1e-9
    t = np.arange(0, 60, 0.05)
    mask = (t > 20) & (t < 30)
    s = np.where(mask, 0.1, 0.95)
    d = detection_metrics(t, s, mask)
    assert d["f1"] > 0.99 and d["far"] == 0.0 and d["latency"] < 0.1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
