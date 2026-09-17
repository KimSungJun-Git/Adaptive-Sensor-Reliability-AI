"""Central defaults for the ASR pipeline.

ROS nodes expose these as ROS parameters (same names, same defaults);
offline tools (training / evaluation) import them directly so both paths
share one source of truth.
"""
import numpy as np

# --- robot (TurtleBot3 waffle_pi) -----------------------------------------
WHEEL_RADIUS = 0.033          # [m]
WHEEL_SEPARATION = 0.287      # [m]
MAX_LIN_VEL = 0.26            # [m/s]
MAX_ANG_VEL = 1.82            # [rad/s]

# --- (sensor x DOF) channels, fixed order used everywhere ------------------
# IMU provides no direct v, so there is no v_imu channel (plan section 7.1).
CHANNELS = ["v_encoder", "omega_encoder", "v_lidar", "omega_lidar", "omega_imu"]
SENSORS = ["encoder", "lidar", "imu"]
# measurement layout per sensor: state indices (3 = v, 4 = omega)
SENSOR_STATE_IDX = {"encoder": [3, 4], "lidar": [3, 4], "imu": [4]}
SENSOR_CHANNELS = {
    "encoder": ["v_encoder", "omega_encoder"],
    "lidar": ["v_lidar", "omega_lidar"],
    "imu": ["omega_imu"],
}

# --- nominal measurement std (normal driving, used for fixed R0) -----------
# Calibrated with `asr_evaluation.calibrate` (plan 12.3): pooled error RMS
# vs Gazebo ground truth over 25 normal runs in tb3_world / house / stage4,
# rounded up ~10%. Includes physically real events (bumps, corridor ICP), so
# it is deliberately conservative. Re-run when robot / world / lidar change.
NOMINAL_STD = {
    "v_encoder": 0.013,
    "omega_encoder": 0.078,
    "v_lidar": 0.088,
    "omega_lidar": 0.234,
    "omega_imu": 0.015,
}

def r0(sensor):
    """Fixed nominal measurement covariance for a sensor."""
    return np.diag([NOMINAL_STD[c] ** 2 for c in SENSOR_CHANNELS[sensor]])

# --- EKF -------------------------------------------------------------------
EKF_Q_DIAG = [1e-4, 1e-4, 1e-4, 0.4, 1.0]   # per second, state [x y yaw v w]
EKF_P0_DIAG = [1e-3, 1e-3, 1e-3, 0.1, 0.1]

# --- feature pipeline --------------------------------------------------------
TICK_RATE = 20.0              # [Hz] feature / reliability update rate
WINDOW_SEC = 1.0              # temporal feature window
LIDAR_MAX_AGE = 0.6           # [s] hold lidar odom at most this long

# --- stationary detection (plan section 10) ---------------------------------
STAT_V_THR = 0.01             # [m/s]
STAT_W_THR = 0.02             # [rad/s]
STAT_A_THR = 1.50             # [m/s^2] | EMA(||a||) - g | -- see features.StationaryDetector
STAT_HOLD_SEC = 0.5

# --- reliability smoothing (plan section 14) ---------------------------------
# gamma/recovery_ticks/s_reject tuned via asr_evaluation.sweep_smoothing
# (plan 14.2/15.3: "실험으로 조정/결정") -- see results/smoothing_sweep.json.
# RECOVERY_TICKS 10->5 and S_REJECT 0.10->0.05 together cut normal-driving
# ATE 65% and fault ATE 16% vs the untuned defaults, with no tradeoff found
# on far/chatter; gamma_fall/gamma_rise/alpha stayed at their original values
# (each candidate tested was flat-to-worse on the sweep's combined score).
GAMMA_FALL = 0.30
GAMMA_RISE = 0.90
RECOVERY_TICKS = 5            # consecutive normal ticks before rise allowed
STATIONARY_FALL_LIMIT = 0.02  # max per-tick drop while stationary

# --- adaptive covariance scaling (plan section 15) ---------------------------
ALPHA_INFLATION = 49.0        # R_max = R0 * (1 + alpha) = 50 x R0
S_REJECT = 0.05               # s <= 0.05 -> measurement rejection candidate

# --- recovery gate thresholds (in sigma units) -------------------------------
GATE_RESIDUAL_SIGMA = 3.0
GATE_NIS_MAX = {"encoder": 9.5, "lidar": 9.5, "imu": 6.6}  # ~chi2 97.5%

# --- rule baseline ------------------------------------------------------------
RULE_INNOV_SIGMA = 3.0
RULE_RESID_SIGMA = 3.0
RULE_S_LOW = 0.2

# --- label generation (plan section 12) ---------------------------------------
# s = exp(-e^2 / (2 * (LABEL_SIGMA_FACTOR * sigma_normal)^2)); the factor is a
# ponytail: calibration knob so that normal-driving error maps to s ~ 0.95.
LABEL_SIGMA_FACTOR = 3.0
