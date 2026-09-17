*[한국어로 보기](README.ko.md)*

# Adaptive Sensor Reliability AI

**Real-time per-(sensor × DOF) reliability estimation from multi-sensor consistency, feeding an Adaptive EKF**

An AI model analyzes the Innovation/NIS of an independent **Shadow EKF** — computed
from the relationships between LiDAR odometry, IMU, and wheel encoder
measurements — to estimate **reliability per (sensor × degree-of-freedom)**, then
feeds that estimate into the EKF measurement covariance as a **bounded
inflation**. The result is stable state estimation through sensor faults such
as wheel slip, without discarding sensors that are still healthy.

```text
LiDAR + IMU + Encoder
        ↓
 Sensor Residuals ──┐
        ↓           │
 Shadow EKF (fixed R) → Innovation / NIS
        ↓
 Reliability AI (ONNX MLP)   s_v_encoder, s_ω_encoder, s_v_lidar, s_ω_lidar, s_ω_imu
        ↓
 Asymmetric Smoothing + Recovery Gate
        ↓
 Bounded R Inflation:  R = R₀·[1 + α(1−s)],  R₀ ≤ R ≤ R₀(1+α)
        ↓
 Adaptive EKF  (or robot_localization + Covariance Injector)
```

Core design ideas (full rationale in the
[development plan](Adaptive_Sensor_Reliability_AI_개발계획서.md), Korean):

- **DOF-separated reliability** — during wheel slip, only `s_v_encoder` drops;
  `s_ω_encoder` and `s_v_lidar` stay high. Healthy information is never
  thrown away just because one channel of one sensor is bad.
- **Shadow EKF** — the Innovation/NIS fed to the AI is computed by a
  *fixed*-covariance filter, so there is no Adaptive EKF → AI feedback loop
  (reliability latching onto its own past decisions).
- **The AI never edits sensor values** — it only adjusts covariance.

## Packages

| Package | Contents |
|---|---|
| `asr_core` | ROS-free core: EKF, 2D point-to-line ICP (+ degeneracy), features, fault injection, smoothing, rule baselines, metrics |
| `asr_msgs` | `SensorReliability`, `ReliabilityArray`, `Innovation`, `FaultStatus`, `LidarQuality` |
| `asr_sensing` | encoder twist, ICP LiDAR odometry, Gazebo ground truth, driver, fault injector |
| `asr_shadow_ekf` | fixed-R Shadow EKF (publishes Innovation/NIS) |
| `asr_reliability_estimator` | feature sync + ONNX inference + smoothing + Adaptive EKF + **Covariance Injector** |
| `asr_bringup` | Gazebo (TurtleBot3 waffle_pi) launch files, demo, dashboard |
| `asr_evaluation` | rosbag extraction, calibration, fault dataset generation, multi-seed training (+ autoencoder comparison), Fixed/Rule/RuleZ/AI comparison, leave-one-world-out, smoothing hyperparameter sweep |

## Quick Start (Simulation Demo)

Requirements: ROS2 Humble, Gazebo Classic 11, `ros-humble-turtlebot3-gazebo`,
`pip install onnxruntime`.

```bash
colcon build --symlink-install && source install/setup.bash
ros2 launch asr_bringup asr_demo.launch.py scenario:=slip
```

Around 22 seconds into the Wheel Slip scenario, watch the dashboard: only
`encoder s_v` drops, and the Adaptive EKF trajectory stays closer to ground
truth than the Fixed EKF. `scenario:=` accepts
`normal | slip | enc_drift | imu_bias | lidar_sector | combo | ...`.

### Online Pipeline Smoke Test

The `asr_core` (12) + `asr_sensing` (6) + `asr_reliability_estimator` (7)
unit tests — 25 total — never touch ROS or Gazebo, so they can't catch a
broken topic name, a message-shape mismatch between nodes, or a node that
silently stops publishing. `smoke_test_live` actually launches the demo and
checks that the reliability publish rate holds and that the right
(sensor × DOF) reliability actually drops during each scenario's fault
window — slow (~1 minute per scenario), but the only end-to-end check worth
putting in a release checklist. The known intermittent gzserver
startup-crash is handled with the same retry as `record_runs`.

```bash
python3 -m asr_evaluation.smoke_test_live --scenario slip   # one scenario
python3 -m asr_evaluation.smoke_test_live --all             # all 10 scenarios
```

## Real-Robot Deployment — robot_localization Covariance Injector

This project does not force you onto its own EKF. The Covariance Injector
republishes your existing sensor topics with the **same message type**, only
with reliability-scaled covariance, so a stock `ekf_filter_node` config needs
just a one-line topic change:

```yaml
odom0: /odom/adaptive        # was: /odom
imu0: /imu/adaptive
twist0: /lidar_twist/adaptive
```

Full chain, in simulation:

```bash
ros2 launch asr_bringup rl_demo.launch.py scenario:=slip
```

This path had never been exercised live before (`smoke_test_live` only
covers the self-contained EKF path via `asr_demo.launch.py`; the Covariance
Injector + a real `robot_localization`/`ekf_node` is separate) — verifying it
directly confirmed clean startup (`ros-humble-robot-localization` 3.5.4),
correct republishing of `/odom/adaptive`, `/asr/imu/adaptive`, and
`/asr/lidar_twist/adaptive` with the same message types as the originals, and
a valid pose from `/odometry/filtered`. The injected covariance itself stayed
small and sane (`v_encoder` twist covariance around 1e-4–1e-3), and as the
slip fault window began, `ekf_filter_node`'s own reported x/y position
covariance grew from 78 to 200 (x-variance) over 20 seconds — at first this
looked like a broken covariance matrix (the x-y cross term exceeded the
x-variance), but the 2×2 submatrix determinant
(199.9 × 10038.3 − 1357.98² ≈ 162,854 > 0) confirms it's a valid, if strongly
correlated, positive-semidefinite matrix (only vx and vω are fused, not x/y
directly, so a correlation coefficient near 0.96 is expected). In other
words: "once the encoder can no longer be trusted, the robot's own belief
about its position correctly becomes less certain too" — the intended
behavior — holds up mathematically in the real `robot_localization` stack,
not just in this project's own EKF. Note that
`asr_bringup/config/robot_localization.yaml` uses robot_localization's
default `process_noise_covariance` rather than a robot-specific tuning; for a
real robot you should tune that to your own dynamics (a standard
`robot_localization` concern, unrelated to this project's AI/reliability
system).

## Training / Experiment Reproduction (Phases 1→10)

Additional requirements: `pip install torch scikit-learn pandas` (needed for
MLP training, the Isolation Forest comparison, and CSV aggregation — unlike
the `onnxruntime` requirement above, none of this is needed to run the demo,
only this training pipeline. `asr_evaluation/package.xml` declares
`python3-sklearn`/`python3-pandas` so `rosdep install` resolves those too,
but `torch` has no rosdep key — it's PyPI-only — so it needs a separate pip
install). `docker/Dockerfile.dev` already has all of this preinstalled; see
[Docker](#docker) below.

```bash
source install/setup.bash
for w in tb3_world house stage4; do                                          # Phase 1: 3 worlds
  python3 -m asr_evaluation.record_runs --world $w --n 10 --duration 90 --out data/raw
done
python3 -m asr_evaluation.extract_bag --raw data/raw --out data/extracted   # Phase 2
python3 -m asr_evaluation.calibrate                                         # print NOMINAL_STD suggestions
python3 -m asr_evaluation.build_dataset                                     # Phase 4-6
asr_evaluation/run_experiments.sh                                           # Phase 7-10, full matrix
```

- `build_dataset`: run-level, per-world stratified split (no time-series
  leakage), injecting **23 scenarios** (normal / encoder v·ω bias·drift·
  noise·stuck·scale·spike·quant·slip / IMU bias·drift·noise·spike·dropout /
  LiDAR sector·noise·dropout / combined / **2 stationary-only experiments**)
  offline onto recorded normal driving. The stationary experiments locate an
  actually-stationary segment in each run's ground truth and inject the
  fault only there (`find_stationary_window`).
- `train`: Rule (instantaneous residual) · **RuleZ** (z-score + NIS gating
  over the same windowed features the AI sees — a strong rule baseline) ·
  IsolationForest · **Autoencoder** (trained on normal data only,
  reconstruction error calibrated into an anomaly score) · MLP × 3 seeds →
  best seed exported to ONNX for deployment. `--holdout <world>` for
  leave-one-world-out.
- `compare`: Fixed / Rule / RuleZ / AI (seed mean ± std) ATE·RPE, detection
  F1/AUROC/latency/recovery, false-alarm rate (`far_clean` excludes segments
  where the sensor was genuinely wrong per ground truth — e.g. a collision —
  not just where a fault was injected), `sat_frac` (fraction of ticks where
  covariance sits near R_max, plan section 21.4).
- `sweep_smoothing`: sweeps `GAMMA_FALL/RISE`, `RECOVERY_TICKS`,
  `ALPHA_INFLATION`, `S_REJECT` around their current defaults without
  retraining (smoothing/covariance parameters live downstream of ONNX
  inference). Results are already reflected in `asr_core/params.py`.

## Results (3 worlds × 32 runs × 23 scenarios, per-world stratified run split + leave-one-world-out)

Data was collected in three Gazebo worlds — `tb3_world` (default TurtleBot3
arena), `house`, and `stage4` (an obstacle-heavy stage) — 10–12 runs each.
**AI numbers are a 3-seed mean** (`results/main/summary.md`). Smoothing and
covariance hyperparameters are the values found by `sweep_smoothing`
(`RECOVERY_TICKS` 10→5, `S_REJECT` 0.10→0.05 — without retraining, this
improved normal-driving ATE by 65%, fault ATE by 16%, and the false-alarm
rate by 21%, all at once; see `results/smoothing_sweep.json`).

### Per-scenario ATE RMSE [m] — Fixed → Rule → RuleZ → **AI**

RuleZ is a strong rule baseline: z-score + NIS gating over the exact same
windowed features the AI sees (much stronger than Rule's plain instantaneous
threshold).

| Scenario | Fixed | Rule | RuleZ | **AI (±seed std)** |
|---|---:|---:|---:|---:|
| normal | 0.012 | 0.012 | 0.011 | **0.032 ± 0.001** |
| slip (encoder v ×1.8) | 1.281 | 1.285 | 1.289 | **0.309 ± 0.026** |
| enc_drift | 2.430 | 2.464 | 2.468 | **0.324 ± 0.079** |
| enc_bias | 0.750 | 0.750 | 0.750 | **0.165 ± 0.013** |
| enc_w_bias / enc_w_drift | 0.510 / 1.171 | 0.327 / 0.399 | 0.097 / 0.267 | **0.086 / 0.038** |
| enc_scale / enc_spike / enc_quant | 0.193 / 0.020 / 0.058 | 0.192 / 0.020 / 0.057 | 0.193 / 0.023 / 0.056 | 0.189 / 0.040 / 0.063 |
| enc_stuck | 0.991 | 1.000 | 1.001 | **0.366 ± 0.041** |
| imu_bias / imu_drift | 2.816 / 2.199 | 2.831 / 2.153 | 2.761 / 2.072 | **0.084 / 0.095** |
| imu_dropout | 2.497 | 2.215 | 2.508 | **0.820 ± 0.180** |
| imu_spike | 0.058 | 0.058 | 0.061 | 0.062 |
| combo (lidar sector + slip) | 0.680 | 0.689 | 0.688 | **0.500 ± 0.051** |
| lidar_sector / lidar_noise / lidar_dropout | 0.015 / 0.015 / 0.013 | **0.014 / 0.013 / 0.012** | **0.012 / 0.012 / 0.011** | 0.067 / 0.149 / 0.123 |
| enc_noise / enc_w_noise / imu_noise | 0.062 / 0.036 / 0.107 | 0.062 / 0.027 / 0.110 | 0.061 / 0.012 / 0.108 | 0.073 / 0.028 / **0.045** |
| stationary_imu_bias (IMU bias while stopped) | 0.454 | 0.456 | 0.504 | **0.107 ± 0.038** |
| stationary_enc_quant (encoder quantization while stopped) | 0.013 | 0.013 | 0.012 | 0.036 |

Detection (averaged over faulted channels): AI F1 **0.63** / AUROC **0.86** /
sat_frac (fraction of time R sits near R_max) 11.2%, vs Rule F1 0.37 / AUROC
0.73 / sat_frac 0% (the rule baseline is a fixed `s=0.2`, so it never fully
saturates), RuleZ F1 0.54 / AUROC 0.81. Normal-driving false-alarm rate
(far_clean): AI 2.3% / Rule 2.0% / RuleZ 1.4%. ONNX CPU inference: 15 µs.

AI-candidate comparison (whole dataset, any-fault AUROC):
**Autoencoder 0.865** (trained on normal data only, reconstruction error
calibrated) > Isolation Forest 0.821. Neither is deployed (plan section
11.5: MLP is the primary candidate), but the Autoencoder doing better than
expected is recorded honestly here regardless.

### Leave-One-World-Out — Generalization to a World Never Seen During Training

Training with `--holdout <world>` excludes that world entirely, then
evaluates only on it (mean ATE over normal/slip/enc_drift/enc_w_drift/
imu_bias/lidar_sector/combo — 7 scenarios; see `results/holdout_*/summary.md`):

| World held out of training | Fixed | Rule | RuleZ | **AI** | far_clean (AI) |
|---|---:|---:|---:|---:|---:|
| tb3_world | 1.239 | 1.151 | 1.087 | **0.263** | 1.9% |
| house | 1.477 | 1.408 | 1.291 | **0.574** | 7.3% |
| stage4 | 0.917 | 0.850 | 0.841 | **0.110** | 6.3% |

Even in a world never seen during training, the AI cuts ATE **2.6–8×**
relative to Fixed — evidence that the per-DOF reliability estimate learned a
general pattern of sensor-relationship behavior rather than memorizing one
world's geometry. The false-alarm rate (far_clean) is higher on every
holdout than on `main` (2.3%), and notably higher on house/stage4 (6–7%)
than tb3_world (1.9%) — narrower corridors and more obstacles mean more room
for miscalibration on genuinely unseen geometry.

### Honest Limitations

**On LiDAR-only faults and purely-noise/spike/quantization-scale
perturbations, RuleZ or even Fixed often beats the AI**
(`lidar_sector` 0.067 vs Fixed 0.015; `enc_spike` 0.040 vs Fixed 0.020). The
root cause, traced directly: the Shadow EKF fuses encoder, lidar, and IMU
into **one shared state**, so a LiDAR-only fault contaminates the whole state
estimate, and the encoder's innovation/NIS — computed against that
contaminated prediction — grows right along with it (measured: during
`lidar_sector`, `s_v_encoder` drops to 0.56 even though the encoder itself
was never touched). The AI can't distinguish this indirect contamination
signal from a genuine direct fault, so it ends up suspecting both channels;
RuleZ, by contrast, explicitly compares NIS magnitudes and gets this case
right. This means the assumption "each sensor's Innovation/NIS reflects only
that sensor's own state" doesn't perfectly hold in a fused-filter
architecture — a methodological limitation, not a bug. **The AI's advantage
is overwhelming on systematic (bias/drift/dropout-type) faults, while
explicit rules still win in attribution-ambiguous single-sensor / pure-noise
situations** — exactly the distinction the project's core three-way
Fixed-vs-Rule-vs-AI comparison was designed to surface. (`enc_stuck` used to
be worse than Fixed before hyperparameter tuning, but reversed to a 2.7×
improvement after adjusting `RECOVERY_TICKS`/`S_REJECT`.)

**A hybrid experiment (choose AI or Rule per channel) — tried, and rejected
after validation.** Seeing the limitation above, the obvious next question
was "why not just use Rule for the LiDAR channels?" — actually implemented
(`asr_core/hybrid.py`) and validated on 159 test episodes. The net effect
was **negative**. Looking only at the LiDAR channel's own held-out
performance (F1_gt), Rule clearly wins over AI (omega_lidar F1 0.89 vs
0.66) — but the real ATE tells a different story: Rule's own simple residual
vote (`RuleBaseline._vote`) sometimes misattributes a genuine encoder fault
to LiDAR, and then rejects perfectly good LiDAR measurements —
`enc_drift` got more than twice as bad (0.32→0.68m). This is the exact mirror
of the AI's encoder channel getting contaminated by a LiDAR fault: **neither
method can perfectly identify "which of this correlated sensor pair is
actually the culprit," and each one's failure just shows up on a different
channel.** Sharpening the binary (1.0/0.2) rule signal to 0.02 so it
actually crosses the rejection threshold fixed one dropout episode
(`stage4_08`, 1.86→0.32m) but made `enc_drift`, `enc_stuck`, and `combo`
worse by more, so the net effect got worse still (12–16 of 23 scenarios
improved, but the remaining few large regressions outweighed them).
**The deployment default therefore stays pure AI (`mode:=ai`)** —
`asr_core/hybrid.py` and `mode:=hybrid` remain in the code, tests, and the
`ate_hybrid` column of `asr_evaluation.compare`, but are not the default.
That an intuitively-obvious idea — "just use the simpler method per channel"
— turned out net-negative is itself quantitative evidence that this
project's core underlying problem (attribution ambiguity between correlated
sensor pairs) doesn't go away just by switching methodology.

**Four real bugs were found and fixed during development** (all reused
below):

1. `StationaryDetector` was completely disabled — its acceleration threshold
   (0.30 m/s²) was far tighter than the Gazebo simulated IMU's actual noise
   (instantaneous std 1.1–3.8 m/s²), so detection rate was 0% even during a
   confirmed 8-second stationary segment. Fixed by adding a short EMA to
   `||a||` and recalibrating the threshold to 1.5, plus a regression test
   (`test_stationary_detector_survives_imu_noise`). With this bug present,
   plan section 10's design goal — not under-detecting faults while
   stationary — could not actually be verified.
2. Found during live verification: `asr_demo.launch.py`'s spawn point sat
   close enough to an obstacle that the scripted driver's avoidance logic
   (rotate in place on obstacle detection) could deadlock permanently in a
   concave corner — measured: 80 seconds spent entirely inside a 0.4m×0.4m
   box, essentially zero net travel. The 32 offline training runs each used
   a different seed and explored 2.3–4.5m normally, so the dataset results
   are unaffected — but this specific combination of demo spawn point and
   seed was only found by running live. Fixed by backing out after 3 seconds
   stuck in avoidance, with a regression test in
   `asr_sensing/test/test_driver.py`.
3. Found while running all 10 scenarios live via `smoke_test_live` (below):
   the online demo's fault injector (`fault_injector_node.py`) was
   implemented differently from the offline code (`asr_core.faults`), so
   `enc_w_bias`/`enc_w_drift` (encoder omega-channel faults) **had no
   effect at all** — the same bias was added to both wheels, leaving the
   difference (omega) untouched and only shifting the average (v). Fixed
   by applying opposite signs per wheel (+right, −left) so only the
   difference (omega) moves and the average (v) stays put
   (`wheel_fault_velocity`, extracted as a pure function with a
   `test_omega_fault_shifts_difference_not_average` regression test).
4. Found the same way: the `imu_dropout` scenario was **not implemented at
   all** online (only bias/noise existed; messages passed straight through
   with no dropout branch). Fixed by freezing and re-publishing the last
   good value while dropped. Bugs 3 and 4 do **not** affect the offline
   dataset results — `asr_core.faults` was correct from the start; the
   issue was confined to the online-demo-only code path.

**Stationary-only experiments** (plan section 19, experiment 7): locates the
actually-stationary segment (median 4.3s) in each run's ground truth and
injects the fault only there. `stationary_imu_bias` shows the AI beating
Fixed by 4.2× — a result that was impossible to verify while bug 1 above
was still present. Conversely, `stationary_enc_quant` shows no clear gap
between methods since absolute error is already small for both (0.01–0.04m).

**Live demo verification** (after fixing bug 2, real-time headless Gazebo,
`asr_demo.launch.py scenario:=slip`, 99s): reliability held at 20.0 Hz;
during slip, `s_v_encoder` dropped precisely from 0.87 to 0.20 while
`s_v_lidar` (0.97) and `s_ω_imu` (0.86) stayed essentially unchanged —
confirming the DOF-separated design works as intended in a live environment.
That single episode's ATE (fixed 1.65 vs adaptive 1.74, reproducible via
`asr_evaluation.analyze_live`) showed no clear advantage, though — unlike
the 730-episode dataset average, one obstacle-heavy live run is dominated by
noise and by the difficulty of the avoidance maneuver itself, so
trustworthy performance claims should rest on the dataset results (table
above), not a single live run. The dashboard GUI was verified by actually
capturing the screen via Xvfb with the real nodes running (reliability
curve, GT/Fixed/Adaptive trajectories, and status text all render
correctly).

After fixing bugs 3 and 4, `smoke_test_live --all` (10 scenarios: slip,
enc_drift, enc_bias, enc_w_bias, enc_w_drift, imu_bias, imu_drift,
imu_dropout, lidar_sector, lidar_dropout) **passes completely** — the only
end-to-end test that live-confirms each scenario's faulted channel actually
drops. The `imu_cb` dropout-freeze logic was later extracted into a pure
function (`imu_fault_step`, same style as `wheel_fault_velocity`) with 2 fast
unit tests (`asr_sensing/test/test_driver.py`). `imu_dropout`'s `MIN_GAP`
threshold was tuned to 0.10 after observing borderline-noisy live gaps
(unlike `lidar_dropout`, this isn't an inherently weak signal, just run-to-run
timing noise — confirmed by rerunning before touching the threshold, not
after). The same pass filled in a test gap in `asr_reliability_estimator`:
the online time-sync ring buffer (`Series`, plan section 18) had no tests at
all before, so it got 7 fast, ROS-free unit tests
(`asr_reliability_estimator/test/test_feature_builder.py`). An initial
extraction of `covariance_injector._f`'s covariance-inflation formula into
its own function turned out to duplicate `asr_core.smoothing.inflate`
exactly, so `_f` now just calls `inflate` directly, and the duplicate test
file was removed — only the genuinely new case (an out-of-range `s`, e.g.
IMU's unused v-channel sentinel of −1.0) was folded into
`asr_core/test/test_core.py`'s existing `test_inflation_bounded`. There are
now 25 unit tests total across 3 packages (`asr_core` 12 + `asr_sensing` 6 +
`asr_reliability_estimator` 7), all passing.

External-data check (`asr_evaluation.external_check`, `~/ros2_auto_tuner` TB3
bags): no IMU/GT, so unusable for training or ATE evaluation — only the
encoder–LiDAR residual could be compared, and in a cylinder-field world ICP
underestimates v, giving a residual std several times the sim value —
meaning `NOMINAL_STD` needs recalibrating whenever this moves to a different
environment.

Plots: `results/ate_by_scenario.png`, `results/main/episode_slip.png`,
`results/main/episode_combo.png`.

## Docker

```bash
docker build -f docker/Dockerfile.runtime -t asr:runtime .   # lightweight, for real-robot deployment
docker build -f docker/Dockerfile.dev -t asr:dev .           # Gazebo + PyTorch, for reproduction/retraining
```

`Dockerfile.runtime` has been **built and actually run** on both Humble and
**Jazzy** (`--build-arg ROS_DISTRO=jazzy`). Jazzy (Ubuntu 24.04) enforces PEP
668 (externally-managed-environment), which broke a plain `pip3 install`
outright — but Humble's (22.04) older pip doesn't understand the
`--break-system-packages` flag at all, so it can't unconditionally be added
either. Fixed by trying the plain form first and retrying with
`--break-system-packages` only on failure, so the same Dockerfile builds
cleanly on both distributions. This is a real, verified check of plan
section 31.5's requirement that deployment packages default-target Jazzy.

## License

Apache-2.0
