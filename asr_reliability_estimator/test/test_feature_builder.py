"""Self-check for Series (plan section 18: online time sync feeding the AI
model's features). Pure Python/numpy, no ROS -- and, unlike almost every
other file in this package, never covered by any test before now. Two
behaviors matter most: rejecting out-of-order/duplicate stamps (a reordered
message must not corrupt interpolation), and refusing to extrapolate
(returning None outside the buffered range rather than silently feeding the
model a stale or made-up value).
"""
import numpy as np

from asr_reliability_estimator.feature_builder import Series


def test_push_ignores_out_of_order_and_duplicate_timestamps():
    s = Series(dims=1)
    s.push(1.0, 10.0)
    s.push(1.0, 999.0)   # duplicate stamp -> dropped
    s.push(0.5, 999.0)   # out of order -> dropped
    s.push(2.0, 20.0)
    assert list(s.buf) == [(1.0, 10.0), (2.0, 20.0)]


def test_interp_linear_between_samples():
    s = Series(dims=1)
    s.push(0.0, 0.0)
    s.push(2.0, 10.0)
    assert np.isclose(s.interp(1.0)[0], 5.0)


def test_interp_none_outside_buffer_range():
    s = Series(dims=1)
    s.push(0.0, 0.0)
    s.push(2.0, 10.0)
    assert s.interp(-0.1) is None
    assert s.interp(2.1) is None


def test_interp_none_with_fewer_than_two_samples():
    s = Series(dims=1)
    assert s.interp(0.0) is None
    s.push(0.0, 5.0)
    assert s.interp(0.0) is None


def test_last_at_sample_and_hold():
    s = Series(dims=2)
    s.push(0.0, 1.0, 2.0)
    s.push(1.0, 3.0, 4.0)
    age, vals = s.last_at(1.5)
    assert np.isclose(age, 0.5)
    assert vals == [3.0, 4.0]


def test_last_at_none_before_first_sample():
    s = Series(dims=1)
    s.push(1.0, 5.0)
    assert s.last_at(0.5) is None


def test_ring_buffer_evicts_oldest():
    s = Series(dims=1, maxlen=3)
    for i in range(5):
        s.push(float(i), float(i))
    assert [row[0] for row in s.buf] == [2.0, 3.0, 4.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
