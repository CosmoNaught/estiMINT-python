"""
Tests for the split-conformal interval correction.
"""

import numpy as np
import pytest

from estimint.v2.training.calibrate import conformal_offset


@pytest.fixture
def y():
    return np.random.default_rng(0).normal(size=500)


def test_offset_shrinks_intervals_that_already_over_cover():
    y = np.linspace(0, 1, 100)
    offset = conformal_offset(y - 1.0, y + 1.0, y, alpha=0.10)
    # every score is negative here, so the correction narrows rather than widens
    assert offset == pytest.approx(-1.0)


def test_offset_restores_target_coverage(y):
    lower, upper = np.zeros_like(y), np.zeros_like(y)  # degenerate point intervals
    offset = conformal_offset(lower, upper, y, alpha=0.10)
    coverage = np.mean((y >= lower - offset) & (y <= upper + offset))
    assert coverage >= 0.90


def test_offset_is_the_score_quantile(y):
    lower, upper = -np.ones_like(y), np.ones_like(y)
    scores = np.maximum(lower - y, y - upper)
    k = int(np.ceil((len(y) + 1) * 0.90))
    assert conformal_offset(lower, upper, y, alpha=0.10) == pytest.approx(np.sort(scores)[k - 1])


def test_wider_intervals_need_a_smaller_offset(y):
    tight = conformal_offset(-0.1 * np.ones_like(y), 0.1 * np.ones_like(y), y)
    loose = conformal_offset(-1.0 * np.ones_like(y), 1.0 * np.ones_like(y), y)
    assert loose < tight


def test_smaller_alpha_gives_a_larger_offset(y):
    lower, upper = np.zeros_like(y), np.zeros_like(y)
    assert conformal_offset(lower, upper, y, alpha=0.01) > conformal_offset(lower, upper, y, alpha=0.20)
