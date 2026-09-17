"""Constant-velocity unicycle EKF, state x = [x, y, yaw, v, omega].

Used both as the fixed-R Shadow EKF (innovation / NIS source) and as the
Adaptive EKF (R scaled by reliability). All measurements are linear in the
state (v / omega picks), only the prediction is nonlinear.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class EKF:
    def __init__(self, q_diag, p0_diag):
        self.q_diag = np.asarray(q_diag, float)
        self.p0_diag = np.asarray(p0_diag, float)
        self.reset()

    def reset(self, x0=None, t0=None):
        self.x = np.zeros(5) if x0 is None else np.asarray(x0, float).copy()
        self.P = np.diag(self.p0_diag).copy()
        self.t = t0

    def predict(self, t):
        """Propagate to time t. Non-positive dt is ignored (out-of-order guard)."""
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0.0:
            return
        dt = min(dt, 0.5)  # sensor gap guard: cap blind propagation
        x, y, yaw, v, w = self.x
        c, s = np.cos(yaw), np.sin(yaw)
        self.x = np.array([x + v * c * dt, y + v * s * dt, wrap(yaw + w * dt), v, w])
        F = np.eye(5)
        F[0, 2] = -v * s * dt
        F[0, 3] = c * dt
        F[1, 2] = v * c * dt
        F[1, 3] = s * dt
        F[2, 4] = dt
        self.P = F @ self.P @ F.T + np.diag(self.q_diag) * dt
        self.t = t

    def update(self, z, idx, R):
        """Update with z measuring state components idx (e.g. [3,4] = v,omega).

        Returns (innovation, S, NIS) computed against the pre-update prediction.
        """
        z = np.atleast_1d(np.asarray(z, float))
        idx = list(idx)
        nu = z - self.x[idx]
        S = self.P[np.ix_(idx, idx)] + R
        Sinv = np.linalg.inv(S)
        nis = float(nu @ Sinv @ nu)
        K = self.P[:, idx] @ Sinv
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        H = np.zeros((len(idx), 5))
        H[range(len(idx)), idx] = 1.0
        IKH = np.eye(5) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form
        return nu, S, nis
