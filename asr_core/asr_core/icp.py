"""2D point-to-line ICP scan matcher for LiDAR odometry.

Point-to-line (PL-ICP, plan section 5.3) instead of point-to-point: a
current point is only pulled *perpendicular* to the local surface of the
previous scan, so sliding along walls and sparse sampling of curved surfaces
no longer bias the translation. The 3x3 Gauss-Newton Hessian also yields a
geometric degeneracy score (corridor / feature-poor environments) that is
exported as a lidar-quality feature.

The initial guess must come from a constant-velocity model or the Shadow
EKF, never from the Adaptive EKF (plan section 8.4).
"""
import numpy as np
from scipy.spatial import cKDTree


def scan_to_points(ranges, angle_min, angle_increment, range_min=0.12, range_max=3.4):
    r = np.asarray(ranges, float)
    n = r.size
    ang = angle_min + angle_increment * np.arange(n)
    ok = np.isfinite(r) & (r > range_min) & (r < range_max)
    return np.column_stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])


def point_normals(pts, tree, k=6):
    """Unit normal per point from PCA of its k nearest neighbours."""
    k = min(k, len(pts))
    _, j = tree.query(pts, k=k)
    nb = pts[j]                                   # (n, k, 2)
    c = nb - nb.mean(axis=1, keepdims=True)
    a = (c[:, :, 0] ** 2).sum(1)
    b = (c[:, :, 0] * c[:, :, 1]).sum(1)
    d = (c[:, :, 1] ** 2).sum(1)
    tr, det = a + d, a * d - b * b
    lam = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))  # smallest eigenvalue
    nx, ny = b, lam - a
    # b == 0: covariance already diagonal, eigenvector is an axis; the
    # smallest-variance axis is y when the points spread along x (a > d)
    alt = np.abs(b) < 1e-12
    nx = np.where(alt, np.where(a > d, 0.0, 1.0), nx)
    ny = np.where(alt, np.where(a > d, 1.0, 0.0), ny)
    nrm = np.hypot(nx, ny)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    return np.column_stack([nx / nrm, ny / nrm])


def icp(prev_pts, cur_pts, init=(0.0, 0.0, 0.0), max_iter=30, tol=1e-5,
        max_corr=0.5, prev_tree=None, prev_normals=None):
    """Estimate T = (dx, dy, dyaw) mapping current-frame points into the
    previous frame, i.e. the robot motion prev -> cur in the previous body
    frame.

    Returns (dx, dy, dyaw, rms, match_ratio, degeneracy).
    rms         point-to-line residual RMS [m] of the final correspondences
    match_ratio fraction of current points with a correspondence < max_corr
    degeneracy  lambda_min / lambda_max of the translation Hessian in [0, 1];
                ~0 in a corridor (motion along the wall unobservable)
    """
    if len(prev_pts) < 20 or len(cur_pts) < 20:
        return 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0
    tree = prev_tree or cKDTree(prev_pts)
    normals = prev_normals if prev_normals is not None else point_normals(prev_pts, tree)
    dx, dy, dyaw = init
    rms, ratio, degen = float("inf"), 0.0, 0.0
    for _ in range(max_iter):
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s], [s, c]])
        t = np.array([dx, dy])
        q = cur_pts @ R.T + t
        dist, j = tree.query(q, distance_upper_bound=max_corr)
        ok = np.isfinite(dist)
        ratio = float(ok.mean())
        if ok.sum() < 10:
            return 0.0, 0.0, 0.0, float("inf"), ratio, 0.0
        p, n, qk = prev_pts[j[ok]], normals[j[ok]], q[ok]
        r = ((qk - p) * n).sum(1)                      # signed point-to-line
        J = np.column_stack([n[:, 0], n[:, 1],
                             n[:, 1] * qk[:, 0] - n[:, 0] * qk[:, 1]])
        H = J.T @ J
        g = J.T @ r
        delta = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        rms = float(np.sqrt((r * r).mean()))
        w = np.linalg.eigvalsh(H[:2, :2])
        degen = float(w[0] / w[1]) if w[1] > 0 else 0.0
        # compose the increment on top of the current transform
        cd, sd = np.cos(delta[2]), np.sin(delta[2])
        Rd = np.array([[cd, -sd], [sd, cd]])
        t = Rd @ t + delta[:2]
        dx, dy, dyaw = float(t[0]), float(t[1]), float(dyaw + delta[2])
        if np.abs(delta).max() < tol:
            break
    return dx, dy, dyaw, rms, ratio, degen


class LidarOdometry:
    """Consecutive scan matching -> body-frame v, omega (plan section 5.3).

    A match whose implied motion exceeds the platform limits (v_max, w_max,
    with margin) is a convergence failure, not a measurement: it is reported
    as invalid so the EKF skips it and the lidar-age feature grows.
    """

    def __init__(self, range_min=0.12, range_max=3.4, v_max=0.6, w_max=3.0):
        self.range_min, self.range_max = range_min, range_max
        self.v_max, self.w_max = v_max, w_max
        self.prev_pts = None
        self.prev_tree = None
        self.prev_normals = None
        self.prev_t = None
        self.prev_T = (0.0, 0.0, 0.0)

    def _set_prev(self, pts, t):
        self.prev_pts, self.prev_t = pts, t
        if len(pts) >= 20:
            self.prev_tree = cKDTree(pts)
            self.prev_normals = point_normals(pts, self.prev_tree)
        else:
            self.prev_tree = self.prev_normals = None

    def step(self, t, ranges, angle_min, angle_increment):
        """Returns dict(v, omega, rms, ratio, degen, dt, valid) or None on the
        first scan."""
        pts = scan_to_points(ranges, angle_min, angle_increment,
                             self.range_min, self.range_max)
        if self.prev_pts is None:
            self._set_prev(pts, t)
            return None
        dt = t - self.prev_t
        if dt <= 0.0:
            return None
        plausible = lambda r: (np.isfinite(r[3]) and abs(r[0] / dt) <= self.v_max
                               and abs(r[2] / dt) <= self.w_max)
        # constant-velocity initial guess, zero-motion guess as a second start
        res = icp(self.prev_pts, pts, init=self.prev_T,
                  prev_tree=self.prev_tree, prev_normals=self.prev_normals)
        if not plausible(res):
            res = icp(self.prev_pts, pts, init=(0.0, 0.0, 0.0),
                      prev_tree=self.prev_tree, prev_normals=self.prev_normals)
        dx, dy, dyaw, rms, ratio, degen = res
        self._set_prev(pts, t)
        if not plausible(res):
            self.prev_T = (0.0, 0.0, 0.0)
            return {"v": 0.0, "omega": 0.0, "rms": 9.9, "ratio": ratio,
                    "degen": 0.0, "dt": dt, "valid": False}
        self.prev_T = (dx, dy, dyaw)
        # dx = forward displacement in previous body frame -> signed v
        return {"v": dx / dt, "omega": dyaw / dt, "rms": rms, "ratio": ratio,
                "degen": degen, "dt": dt, "valid": True}
