"""Self-check for the driver's obstacle-avoidance escape logic.

Regression: pure in-place rotation deadlocked for 80s straight when the demo
spawn point was wedged near a wall (asr_world.world, seed=42) -- the live
reliability signal looked fine offline but every online metric was garbage
because the robot never actually drove anywhere.
"""
import numpy as np

from asr_core.faults import FaultEvent
from asr_sensing.driver_node import avoid_cmd
from asr_sensing.fault_injector_node import imu_fault_step, wheel_fault_velocity


def test_rotates_then_backs_out_if_stuck():
    since, adir = None, 1.0
    lin, ang, since, adir = avoid_cmd(front=0.2, left=1.0, t=10.0,
                                      avoid_since=since, avoid_dir=adir)
    assert lin == 0.0 and ang != 0.0  # rotates first, doesn't reverse yet

    lin, ang, since, adir = avoid_cmd(front=0.2, left=1.0, t=10.5,
                                      avoid_since=since, avoid_dir=adir)
    assert lin == 0.0 and since == 10.0  # avoid_since latched, still rotating

    lin, ang, since, adir = avoid_cmd(front=0.2, left=1.0, t=13.6,
                                      avoid_since=since, avoid_dir=adir)
    assert lin < 0.0 and ang == 0.0  # 3.6s stuck -> backs straight out


def test_avoid_dir_follows_left_clearance():
    _, ang, *_ = avoid_cmd(front=0.2, left=0.3, t=0.0, avoid_since=None, avoid_dir=1.0)
    assert ang < 0.0  # left blocked -> turn the other way
    _, ang, *_ = avoid_cmd(front=0.2, left=1.0, t=0.0, avoid_since=None, avoid_dir=1.0)
    assert ang > 0.0  # left clear -> default direction


def test_omega_fault_shifts_difference_not_average():
    # regression: the online injector used to apply every encoder fault
    # symmetrically to both wheels, so a channel="omega" bias never actually
    # changed omega at all (smoke_test_live caught this: reliability never
    # dropped for enc_w_bias/enc_w_drift in a live run).
    rng = np.random.default_rng(0)
    e = FaultEvent("encoder", "bias", 0.0, 10.0, mag=0.1, channel="omega")
    vl = wheel_fault_velocity(1.0, -1.0, e, t=1.0, rng=rng)
    vr = wheel_fault_velocity(1.0, +1.0, e, t=1.0, rng=rng)
    assert abs((vr - vl) - (0.0)) > 1e-9          # difference (~omega) moved
    assert abs((vr + vl) / 2 - 1.0) < 1e-9        # average (~v) untouched


def test_v_fault_shifts_average_not_difference():
    rng = np.random.default_rng(0)
    e = FaultEvent("encoder", "bias", 0.0, 10.0, mag=0.1, channel="v")
    vl = wheel_fault_velocity(1.0, -1.0, e, t=1.0, rng=rng)
    vr = wheel_fault_velocity(1.0, +1.0, e, t=1.0, rng=rng)
    assert abs(vr - vl) < 1e-9                    # difference (~omega) untouched
    assert (vr + vl) / 2 > 1.0 + 1e-9             # average (~v) moved


def test_imu_dropout_freezes_then_releases_cleanly():
    # regression: imu_dropout was unimplemented online (smoke_test_live caught
    # omega_imu never dropping for the imu_dropout scenario in a live run).
    rng = np.random.default_rng(0)
    e = FaultEvent("imu", "dropout", 0.0, 10.0, mag=1.0, channel="omega")
    w, ax, frozen = imu_fault_step(0.5, 0.1, [e], t=1.0, frozen=None, rng=rng)
    assert (w, ax) == (0.5, 0.1)
    # raw readings keep changing underneath, but output stays frozen
    w, ax, frozen = imu_fault_step(9.0, 9.0, [e], t=2.0, frozen=frozen, rng=rng)
    assert (w, ax) == (0.5, 0.1)
    # once the window ends, fresh values pass through and frozen clears
    w, ax, frozen = imu_fault_step(9.0, 9.0, [], t=11.0, frozen=frozen, rng=rng)
    assert (w, ax) == (9.0, 9.0)
    assert frozen is None


def test_imu_bias_adds_to_omega_only():
    e = FaultEvent("imu", "bias", 0.0, 10.0, mag=0.05, channel="omega")
    w, ax, frozen = imu_fault_step(0.2, 1.0, [e], t=1.0, frozen=None,
                                    rng=np.random.default_rng(0))
    assert abs(w - 0.25) < 1e-9
    assert ax == 1.0
    assert frozen is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
