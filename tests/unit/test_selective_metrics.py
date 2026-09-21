from __future__ import annotations

import numpy as np
import pytest

from mcp_route_reliability.evaluation.selective import risk_coverage_curve


def test_risk_coverage_curve_and_aurc_match_hand_calculation() -> None:
    result = risk_coverage_curve(
        np.asarray([0, 1, 0]),
        np.asarray([0.9, 0.8, 0.2]),
    )

    assert [point.accepted for point in result.points] == [0, 1, 2, 3]
    assert [point.coverage for point in result.points] == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert [point.risk for point in result.points] == pytest.approx([0, 0, 1 / 2, 1 / 3])
    assert result.aurc == pytest.approx(2 / 9)


def test_risk_coverage_sort_is_stable_for_tied_confidence() -> None:
    result = risk_coverage_curve(
        np.asarray([1, 0, 0]),
        np.asarray([0.8, 0.8, 0.7]),
    )
    assert result.points[1].risk == 1.0
    assert result.points[2].risk == 0.5


def test_risk_coverage_validates_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        risk_coverage_curve(np.asarray([]), np.asarray([]))
    with pytest.raises(ValueError, match="same length"):
        risk_coverage_curve(np.asarray([0]), np.asarray([0.5, 0.4]))
    with pytest.raises(ValueError, match="binary"):
        risk_coverage_curve(np.asarray([2]), np.asarray([0.5]))
