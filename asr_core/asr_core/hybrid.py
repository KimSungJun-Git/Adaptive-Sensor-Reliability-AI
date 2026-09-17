"""Per-channel reliability source selection: use whichever method the data
says is better for that channel, instead of committing to one method
everywhere.

Decided on the *validation* split (never the test split used to report
results) via asr_evaluation.train's per-channel F1/AUROC-vs-ground-truth
comparison: the plain instantaneous-residual Rule beats both the MLP and
RuleZ specifically on the LiDAR channels (omega_lidar F1 0.89 vs AI 0.66 vs
RuleZ 0.43) -- LiDAR's own residual is a clean, immediate signal that a 1s
smoothed window (RuleZ) or the MLP (contaminated by the shared Shadow EKF
state, see README's "정직한 한계") only muddies. Every other channel keeps
the MLP, which wins there by a wide margin.

This is not "AI vs rules" -- it is one number per channel, picked by
evidence, not by allegiance to either method.
"""
import numpy as np

from .params import CHANNELS, RULE_S_LOW, S_REJECT

RULE_CHANNELS = {"v_lidar", "omega_lidar"}
RULE_MASK = np.array([c in RULE_CHANNELS for c in CHANNELS])

# RuleBaseline's raw output is binary: 1.0 (fine) or RULE_S_LOW=0.2 (bad).
# 0.2 sits *above* S_REJECT (0.05 in the tuned defaults), so a Rule-flagged
# LiDAR measurement always gets heavily inflated but can never be outright
# rejected the way a confident AI score near 0 can -- this alone turned one
# hard episode (stage4_08 lidar_dropout, ATE 0.31->1.86) into an outlier
# that erased the hybrid's gains everywhere else. Sharpening the floor to
# actually cross S_REJECT fixes that episode outright with no other
# regression found in a full 23-scenario re-check.
SHARP_BAD = 0.02
assert SHARP_BAD < S_REJECT < RULE_S_LOW


def combine(s_ai, s_rule):
    """s_ai, s_rule: arrays shaped (..., len(CHANNELS)) of *raw* (pre-
    smoothing) reliability (s_rule from asr_core.rules.RuleBaseline).
    Returns the per-channel hybrid raw score: AI everywhere except the LiDAR
    channels, where Rule's binary flag is sharpened so a confident "bad" can
    still trigger measurement rejection."""
    s_rule_sharp = np.where(s_rule < 0.5, SHARP_BAD, 1.0)
    return np.where(RULE_MASK, s_rule_sharp, s_ai)
