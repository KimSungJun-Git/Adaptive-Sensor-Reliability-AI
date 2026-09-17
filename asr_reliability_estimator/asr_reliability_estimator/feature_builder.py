"""Online time synchronization (plan section 18): per-stream ring buffers,
timestamp-based interpolation at tick time, sample-and-hold for lidar."""
from collections import deque

import numpy as np


class Series:
    """Ring buffer of (t, value...) rows with linear interpolation."""

    def __init__(self, dims, maxlen=400):
        self.dims = dims
        self.buf = deque(maxlen=maxlen)

    def push(self, t, *values):
        if self.buf and t <= self.buf[-1][0]:
            return
        self.buf.append((t, *values))

    def interp(self, t):
        """Linear interpolation at t; None if the buffer cannot cover t."""
        if len(self.buf) < 2 or t < self.buf[0][0] or t > self.buf[-1][0]:
            return None
        a = np.asarray(self.buf)
        return [float(np.interp(t, a[:, 0], a[:, 1 + d])) for d in range(self.dims)]

    def last_at(self, t):
        """Latest row with stamp <= t, else None. Returns (age, values)."""
        for row in reversed(self.buf):
            if row[0] <= t:
                return t - row[0], list(row[1:])
        return None
