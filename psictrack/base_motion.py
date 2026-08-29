"""TGA box Kalman state, candidate costs, and ordinary Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import BaseTrackerConfig
from .types import Observation
# TGA is a fixed counterfactual, not a per-species tuning surface.
_DEFAULT_PROCESS_VARIANCE = 1.0
_DEFAULT_MEASUREMENT_VARIANCE = 1.0
_IOU_GATE = 0.01
_CENTER_GATE = 4.0
_IOU_EVIDENCE_WEIGHT = 0.65


def _bbox_measurement(bbox_xywh: np.ndarray) -> np.ndarray:
    x, y, width, height = np.asarray(bbox_xywh, dtype=np.float64)
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    return np.asarray([x + width / 2.0, y + height / 2.0, np.log(width), np.log(height)])


@dataclass
class BoxMotionFilter:
    """Linear Kalman filter over center, log-size, and per-frame velocity."""

    mean: np.ndarray
    covariance: np.ndarray
    process_var: float
    measurement_var: float
    last_b0_mean: np.ndarray | None = None

    @classmethod
    def initiate(
        cls,
        bbox_xywh: np.ndarray,
        *,
        process_var: float = _DEFAULT_PROCESS_VARIANCE,
        measurement_var: float = _DEFAULT_MEASUREMENT_VARIANCE,
    ) -> "BoxMotionFilter":
        measurement = _bbox_measurement(bbox_xywh)
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = measurement
        covariance = np.diag([1.0, 1.0, 0.1, 0.1, 10.0, 10.0, 1.0, 1.0])
        return cls(mean, covariance, float(process_var), float(measurement_var))

    @property
    def bbox_xywh(self) -> np.ndarray:
        width = float(np.exp(np.clip(self.mean[2], -20.0, 20.0)))
        height = float(np.exp(np.clip(self.mean[3], -20.0, 20.0)))
        return np.asarray(
            [self.mean[0] - width / 2.0, self.mean[1] - height / 2.0, width, height],
            dtype=np.float64,
        )

    @property
    def b0_bbox_xywh(self) -> np.ndarray:
        mean = self.mean if self.last_b0_mean is None else self.last_b0_mean
        width = float(np.exp(np.clip(mean[2], -20.0, 20.0)))
        height = float(np.exp(np.clip(mean[3], -20.0, 20.0)))
        return np.asarray(
            [mean[0] - width / 2.0, mean[1] - height / 2.0, width, height],
            dtype=np.float64,
        )

    @property
    def center_velocity(self) -> np.ndarray:
        return self.mean[4:6].copy()

    @property
    def normalized_speed(self) -> float:
        bbox = self.bbox_xywh
        body_scale = max(float(np.sqrt(bbox[2] * bbox[3])), 1e-9)
        return float(np.linalg.norm(self.mean[4:6]) / body_scale)

    def predict(
        self,
        delta_frames: int = 1,
        *,
        process_scale: float = 1.0,
        velocity_retention: float = 1.0,
        center_control: np.ndarray | None = None,
        control_covariance: np.ndarray | None = None,
    ) -> np.ndarray:
        if not np.isfinite(process_scale) or process_scale <= 0:
            raise ValueError("process_scale must be positive and finite")
        if not np.isfinite(velocity_retention) or not 0.0 <= velocity_retention <= 1.0:
            raise ValueError("velocity_retention must be finite and in [0, 1]")
        if int(delta_frames) != delta_frames or delta_frames < 1:
            raise ValueError("delta_frames must be a positive integer")
        control = np.zeros(2, dtype=np.float64) if center_control is None else np.asarray(center_control, dtype=np.float64)
        if control.shape != (2,) or not np.all(np.isfinite(control)):
            raise ValueError("center_control must be a finite vector with shape (2,)")
        control_cov = (
            np.zeros((2, 2), dtype=np.float64)
            if control_covariance is None
            else np.asarray(control_covariance, dtype=np.float64)
        )
        if control_cov.shape != (2, 2) or not np.all(np.isfinite(control_cov)):
            raise ValueError("control_covariance must be a finite 2x2 matrix")
        control_cov = (control_cov + control_cov.T) / 2.0
        if np.min(np.linalg.eigvalsh(control_cov)) < -1e-10:
            raise ValueError("control_covariance must be positive semidefinite")
        dt = float(delta_frames)
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4, dtype=np.float64) * dt * velocity_retention
        transition[4:, 4:] = np.eye(4, dtype=np.float64) * velocity_retention
        position_scale = max(1.0, dt * dt)
        velocity_scale = max(1.0, dt)
        process = np.diag(
            [position_scale, position_scale, dt, dt, velocity_scale, velocity_scale, dt, dt]
        ) * self.process_var * float(process_scale)
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.last_b0_mean = self.mean.copy()
        self.mean[:2] += control
        self.covariance[:2, :2] += control_cov
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return self.bbox_xywh

    def innovation(self, bbox_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        measurement = _bbox_measurement(bbox_xywh)
        projection = np.zeros((4, 8), dtype=np.float64)
        projection[:, :4] = np.eye(4, dtype=np.float64)
        residual = measurement - projection @ self.mean
        covariance = projection @ self.covariance @ projection.T
        covariance += np.eye(4, dtype=np.float64) * self.measurement_var
        return residual, covariance

    def update(self, bbox_xywh: np.ndarray, *, strength: float = 1.0) -> None:
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        measurement = _bbox_measurement(bbox_xywh)
        projection = np.zeros((4, 8), dtype=np.float64)
        projection[:, :4] = np.eye(4, dtype=np.float64)
        effective_variance = self.measurement_var / strength
        observation_covariance = np.eye(4, dtype=np.float64) * effective_variance
        residual = measurement - projection @ self.mean
        innovation_covariance = projection @ self.covariance @ projection.T + observation_covariance
        gain = np.linalg.solve(innovation_covariance, projection @ self.covariance).T
        self.mean = self.mean + gain @ residual
        identity = np.eye(8, dtype=np.float64)
        correction = identity - gain @ projection
        self.covariance = (
            correction @ self.covariance @ correction.T + gain @ observation_covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0


@dataclass(frozen=True)
class Assignment:
    matches: tuple[tuple[int, int], ...]
    unmatched_rows: tuple[int, ...]
    unmatched_cols: tuple[int, ...]


def solve_assignment(cost: np.ndarray, max_cost: float) -> Assignment:
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    row_count, col_count = matrix.shape
    if row_count == 0 or col_count == 0:
        return Assignment((), tuple(range(row_count)), tuple(range(col_count)))
    finite = np.isfinite(matrix)
    safe = np.where(finite, matrix, max_cost + 1e6)
    rows, cols = linear_sum_assignment(safe)
    matches = tuple(
        (int(row), int(col))
        for row, col in zip(rows.tolist(), cols.tolist())
        if finite[row, col] and matrix[row, col] <= max_cost
    )
    matched_rows = {row for row, _ in matches}
    matched_cols = {col for _, col in matches}
    return Assignment(
        matches=matches,
        unmatched_rows=tuple(row for row in range(row_count) if row not in matched_rows),
        unmatched_cols=tuple(col for col in range(col_count) if col not in matched_cols),
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_width = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _normalized_center_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_center = left[:2] + left[2:] / 2.0
    right_center = right[:2] + right[2:] / 2.0
    scale = max(1e-6, np.sqrt(0.5 * (left[2] * left[3] + right[2] * right[3])))
    return float(np.linalg.norm(left_center - right_center) / scale)


def box_cost_matrix(
    tracks: Sequence[object],
    observations: Sequence[Observation],
    config: BaseTrackerConfig,
    *,
    center_gate_scale: float | Sequence[float] = 1.0,
) -> np.ndarray:
    cost = np.full((len(tracks), len(observations)), np.inf, dtype=np.float64)
    scales = np.asarray(center_gate_scale, dtype=np.float64)
    if scales.ndim == 0:
        scales = np.full(len(tracks), float(scales), dtype=np.float64)
    if scales.shape != (len(tracks),):
        raise ValueError("center_gate_scale must be scalar or provide one value per track")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("center_gate_scale values must be positive and finite")
    for row, track in enumerate(tracks):
        track_bbox = np.asarray(getattr(track, "bbox_xywh"), dtype=np.float64)
        for col, observation in enumerate(observations):
            iou = _iou(track_bbox, observation.bbox_xywh)
            center = _normalized_center_distance(track_bbox, observation.bbox_xywh)
            center_gate = _CENTER_GATE * scales[row]
            if iou < _IOU_GATE and center > center_gate:
                continue
            center_term = min(1.0, center / center_gate)
            cost[row, col] = (
                _IOU_EVIDENCE_WEIGHT * (1.0 - iou)
                + (1.0 - _IOU_EVIDENCE_WEIGHT) * center_term
            )
    return cost
