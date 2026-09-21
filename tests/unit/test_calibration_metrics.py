from __future__ import annotations

import numpy as np
import pytest

from mcp_route_reliability.evaluation.calibration import (
    brier_score,
    equal_frequency_ece,
    error_auroc,
)


def test_brier_score_matches_hand_calculation() -> None:
    result = brier_score(np.asarray([1, 0]), np.asarray([0.8, 0.3]))
    assert result == pytest.approx(0.065)


def test_five_equal_frequency_ece_bins_include_boundaries_and_counts() -> None:
    y_correct = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p_correct = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    result = equal_frequency_ece(y_correct, p_correct)

    assert result.ece == pytest.approx(0.17)
    assert len(result.bins) == 5
    assert [bin_.count for bin_ in result.bins] == [2, 2, 2, 2, 2]
    assert result.bins[0].lower_bound == 0.1
    assert result.bins[0].upper_bound == 0.2
    assert result.bins[-1].mean_confidence == pytest.approx(0.95)


def test_error_auroc_uses_routing_error_as_positive_class() -> None:
    result = error_auroc(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.9, 0.2, 0.8]),
    )
    assert result.available is True
    assert result.value == 1.0


def test_error_auroc_single_class_is_explicitly_unavailable() -> None:
    result = error_auroc(np.asarray([0, 0]), np.asarray([0.1, 0.2]))
    assert result.available is False
    assert result.value is None
    assert result.reason == "single_target_class"


@pytest.mark.parametrize(
    ("y", "p"),
    [
        (np.asarray([]), np.asarray([])),
        (np.asarray([1]), np.asarray([0.5, 0.6])),
        (np.asarray([2]), np.asarray([0.5])),
        (np.asarray([1]), np.asarray([1.2])),
    ],
)
def test_probability_metrics_validate_inputs(y: np.ndarray, p: np.ndarray) -> None:
    with pytest.raises(ValueError):
        brier_score(y, p)


def test_ece_requires_at_least_one_sample_per_bin() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        equal_frequency_ece(np.asarray([0, 1]), np.asarray([0.2, 0.8]))
