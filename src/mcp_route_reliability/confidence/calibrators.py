"""Serializable probability calibrators fitted only on Calibration data."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from mcp_route_reliability.models import ConfidenceFeatures, StrictModel


class ProbabilityCalibrator(Protocol):
    def predict_correctness(
        self, rows: Sequence[ConfidenceFeatures]
    ) -> NDArray[np.float64]: ...

    def to_payload(self) -> dict[str, object]: ...


class UnavailableCalibrator(StrictModel):
    method: Literal["isotonic", "logistic"]
    available: Literal[False] = False
    reason: str
    sample_size: int = Field(ge=0)
    unique_score_count: int | None = Field(default=None, ge=0)

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def _feature_value(row: ConfidenceFeatures, feature_name: str) -> float | None:
    if feature_name not in type(row).model_fields:
        raise ValueError(f"unknown confidence feature {feature_name!r}")
    value = getattr(row, feature_name)
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"feature {feature_name!r} must be finite")
        return number
    raise ValueError(f"feature {feature_name!r} is not numeric")


def _validate_targets(rows: Sequence[ConfidenceFeatures], labels: Sequence[int]) -> None:
    if not rows:
        raise ValueError("calibrator fitting requires nonempty rows")
    if len(rows) != len(labels):
        raise ValueError("calibrator rows and labels must have the same length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("calibrator labels must be binary")


class IsotonicCalibrator(StrictModel):
    method: Literal["isotonic"] = "isotonic"
    available: Literal[True] = True
    feature_name: str
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    sample_size: int = Field(ge=1)
    unique_score_count: int = Field(ge=2)

    def predict_correctness(
        self, rows: Sequence[ConfidenceFeatures]
    ) -> NDArray[np.float64]:
        values: list[float] = []
        for row in rows:
            value = _feature_value(row, self.feature_name)
            if value is None:
                raise ValueError(f"feature {self.feature_name!r} is missing at prediction time")
            values.append(value)
        return np.asarray(
            np.interp(values, self.x_thresholds, self.y_thresholds),
            dtype=np.float64,
        )

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class LogisticConfidenceModel(StrictModel):
    method: Literal["logistic"] = "logistic"
    available: Literal[True] = True
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    sample_size: int = Field(ge=1)
    regularization: Literal["l2"] = "l2"
    c_value: float = Field(gt=0)

    def _matrix(self, rows: Sequence[ConfidenceFeatures]) -> NDArray[np.float64]:
        matrix = np.empty((len(rows), len(self.feature_names)), dtype=np.float64)
        for row_index, row in enumerate(rows):
            for feature_index, feature_name in enumerate(self.feature_names):
                value = _feature_value(row, feature_name)
                matrix[row_index, feature_index] = (
                    self.medians[feature_index] if value is None else value
                )
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        return np.asarray((matrix - means) / scales, dtype=np.float64)

    def predict_correctness(
        self, rows: Sequence[ConfidenceFeatures]
    ) -> NDArray[np.float64]:
        matrix = self._matrix(rows)
        logits = matrix @ np.asarray(self.coefficients) + self.intercept
        logits = np.clip(logits, -709.0, 709.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def fit_isotonic(
    rows: Sequence[ConfidenceFeatures],
    labels: Sequence[int],
    *,
    feature_name: str,
) -> IsotonicCalibrator | UnavailableCalibrator:
    _validate_targets(rows, labels)
    values = [_feature_value(row, feature_name) for row in rows]
    if any(value is None for value in values):
        return UnavailableCalibrator(
            method="isotonic",
            reason=f"feature_missing:{feature_name}",
            sample_size=len(rows),
        )
    numeric = np.asarray([float(value) for value in values if value is not None])
    unique_count = int(np.unique(numeric).size)
    if unique_count < 2:
        return UnavailableCalibrator(
            method="isotonic",
            reason="insufficient_unique_scores",
            sample_size=len(rows),
            unique_score_count=unique_count,
        )
    if len(set(labels)) < 2:
        return UnavailableCalibrator(
            method="isotonic",
            reason="single_target_class",
            sample_size=len(rows),
            unique_score_count=unique_count,
        )

    estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    estimator.fit(numeric, np.asarray(labels, dtype=np.int64))
    return IsotonicCalibrator(
        feature_name=feature_name,
        x_thresholds=tuple(float(value) for value in estimator.X_thresholds_),
        y_thresholds=tuple(float(value) for value in estimator.y_thresholds_),
        sample_size=len(rows),
        unique_score_count=unique_count,
    )


def fit_logistic(
    rows: Sequence[ConfidenceFeatures],
    labels: Sequence[int],
    *,
    feature_names: Sequence[str],
    c_value: float = 1.0,
) -> LogisticConfidenceModel | UnavailableCalibrator:
    if not 4 <= len(feature_names) <= 6:
        raise ValueError("Logistic Confidence Model requires 4 to 6 features")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Logistic feature names must be unique")
    if c_value <= 0 or not math.isfinite(c_value):
        raise ValueError("c_value must be finite and positive")
    _validate_targets(rows, labels)
    if len(set(labels)) < 2:
        return UnavailableCalibrator(
            method="logistic",
            reason="single_target_class",
            sample_size=len(rows),
        )

    matrix = np.empty((len(rows), len(feature_names)), dtype=np.float64)
    medians: list[float] = []
    for feature_index, feature_name in enumerate(feature_names):
        values = [_feature_value(row, feature_name) for row in rows]
        observed = [value for value in values if value is not None]
        if not observed:
            return UnavailableCalibrator(
                method="logistic",
                reason=f"feature_all_missing:{feature_name}",
                sample_size=len(rows),
            )
        median = float(np.median(observed))
        medians.append(median)
        matrix[:, feature_index] = [median if value is None else value for value in values]

    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales[scales == 0] = 1.0
    standardized = (matrix - means) / scales
    estimator = LogisticRegression(
        l1_ratio=0.0,
        C=c_value,
        solver="liblinear",
        random_state=0,
        max_iter=1000,
    )
    estimator.fit(standardized, np.asarray(labels, dtype=np.int64))
    return LogisticConfidenceModel(
        feature_names=tuple(feature_names),
        medians=tuple(float(value) for value in medians),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        sample_size=len(rows),
        c_value=c_value,
    )
