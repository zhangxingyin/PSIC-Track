"""Uncertainty-calibrated pose-rhythm association for PSIC-Track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_COVARIANCE_JITTER = 1e-8

class CausalEvidenceCredibility:
    """Causal Beta posterior for whether an evidence stream discriminates IDs."""

    def __init__(self) -> None:
        self._alpha = 1.0
        self._beta = 1.0

    @property
    def strength(self) -> float:
        total = self._alpha + self._beta
        return float(np.clip(max(0.0, self._alpha - self._beta) / total, 0.0, 1.0))

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @property
    def beta(self) -> float:
        return float(self._beta)

    def update(
        self,
        selected_distance: float,
        alternative_distances: np.ndarray,
        *,
        reliability: float,
        ambiguity: float,
    ) -> None:
        if not np.isfinite(selected_distance) or selected_distance < 0.0:
            raise ValueError("selected_distance must be finite and nonnegative")
        if not np.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be finite and in [0, 1]")
        if not np.isfinite(ambiguity) or not 0.0 <= ambiguity <= 1.0:
            raise ValueError("ambiguity must be finite and in [0, 1]")
        alternatives = np.asarray(alternative_distances, dtype=np.float64)
        if alternatives.ndim != 1:
            raise ValueError("alternative_distances must be one-dimensional")
        valid = np.isfinite(alternatives) & (alternatives >= 0.0)
        if not np.any(valid):
            return
        selected_likelihood = float(np.exp(-0.5 * selected_distance))
        alternative_likelihood = float(
            np.mean(np.exp(-0.5 * alternatives[valid]))
        )
        denominator = max(
            selected_likelihood + alternative_likelihood,
            np.finfo(np.float64).tiny,
        )
        success = selected_likelihood / denominator
        weight = (1.0 - ambiguity) * reliability
        self._alpha += weight * success
        self._beta += weight * (1.0 - success)



@dataclass(frozen=True)
class GaussianPose:
    mean: np.ndarray
    covariance: np.ndarray
    valid_mask: np.ndarray
    quality: float
    circular_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        valid = np.asarray(self.valid_mask, dtype=bool).copy()
        if mean.ndim != 1 or covariance.shape != (mean.size, mean.size) or valid.shape != mean.shape:
            raise ValueError("GaussianPose mean, covariance, and valid_mask shapes are inconsistent")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("GaussianPose contains non-finite values")
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValueError("GaussianPose quality must be in [0, 1]")
        circular = (
            np.zeros(mean.size, dtype=bool)
            if self.circular_mask is None
            else np.asarray(self.circular_mask, dtype=bool).copy()
        )
        if circular.shape != mean.shape:
            raise ValueError("circular_mask shape must match mean")
        for array in (mean, covariance, valid, circular):
            array.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "quality", float(self.quality))
        object.__setattr__(self, "circular_mask", circular)


def joint_pose_distance(observed: GaussianPose, predicted: GaussianPose) -> float:
    """Dimension-normalized squared Mahalanobis distance on shared evidence."""

    if observed.mean.shape != predicted.mean.shape:
        raise ValueError("observed and predicted pose dimensions differ")
    valid = observed.valid_mask & predicted.valid_mask
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return float("inf")
    residual = observed.mean[indices] - predicted.mean[indices]
    circular = observed.circular_mask[indices] | predicted.circular_mask[indices]
    residual[circular] = (residual[circular] + np.pi) % (2.0 * np.pi) - np.pi
    covariance = (
        observed.covariance[np.ix_(indices, indices)]
        + predicted.covariance[np.ix_(indices, indices)]
    )
    try:
        solution = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        try:
            solution = np.linalg.solve(
                covariance
                + np.eye(indices.size, dtype=np.float64) * _COVARIANCE_JITTER,
                residual,
            )
        except np.linalg.LinAlgError:
            return float("inf")
    mahalanobis = float(residual @ solution)
    return mahalanobis / float(indices.size)


def pose_reliability(observed: GaussianPose, predicted: GaussianPose) -> float:
    if observed.mean.shape != predicted.mean.shape:
        return 0.0
    common = observed.valid_mask & predicted.valid_mask
    coverage = float(np.mean(common)) if common.size else 0.0
    return float(np.clip(np.sqrt(observed.quality * predicted.quality) * coverage, 0.0, 1.0))


def _posterior_ambiguity(values: np.ndarray) -> float:
    """Return one minus the posterior margin between the best explanations."""

    nonnegative = np.maximum(values, 0.0)
    scale = max(float(np.max(nonnegative)), np.finfo(float).eps)
    likelihood = 1.0 / (
        nonnegative / scale + np.finfo(float).eps
    )
    posterior = np.sort(
        likelihood / max(float(np.sum(likelihood)), np.finfo(float).tiny)
    )[::-1]
    return float(np.clip(1.0 - (posterior[0] - posterior[1]), 0.0, 1.0))


def ambiguity_scores(cost: np.ndarray, feasible_mask: np.ndarray) -> np.ndarray:
    """Infer ambiguity from row/column candidate posteriors without a margin knob."""

    matrix = np.asarray(cost, dtype=np.float64)
    feasible = np.asarray(feasible_mask, dtype=bool)
    if matrix.shape != feasible.shape or matrix.ndim != 2:
        raise ValueError("cost and feasible_mask must be equally shaped matrices")
    ambiguity = np.zeros_like(matrix)
    for row in range(matrix.shape[0]):
        columns = np.flatnonzero(feasible[row] & np.isfinite(matrix[row]))
        if columns.size >= 2:
            score = _posterior_ambiguity(matrix[row, columns])
            ambiguity[row, columns] = np.maximum(ambiguity[row, columns], score)
    for column in range(matrix.shape[1]):
        rows = np.flatnonzero(feasible[:, column] & np.isfinite(matrix[:, column]))
        if rows.size >= 2:
            score = _posterior_ambiguity(matrix[rows, column])
            ambiguity[rows, column] = np.maximum(ambiguity[rows, column], score)
    return ambiguity


def compose_joint_cost(
    box_cost: np.ndarray,
    pose_distance: np.ndarray,
    ambiguity: np.ndarray,
    pose_reliability_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Add PSOI pose evidence only where box candidates genuinely compete."""

    matrices = [
        np.asarray(value, dtype=np.float64)
        for value in (
            box_cost,
            pose_distance,
            ambiguity,
            pose_reliability_matrix,
        )
    ]
    if matrices[0].ndim != 2 or any(value.shape != matrices[0].shape for value in matrices[1:]):
        raise ValueError("all joint association matrices must be equally shaped and two-dimensional")

    def relative_term(distance: np.ndarray, reliability: np.ndarray) -> np.ndarray:
        active = (
            np.isfinite(matrices[0])
            & np.isfinite(distance)
            & (matrices[2] > 0.0)
            & (reliability > 0.0)
        )
        row_relative = np.zeros_like(distance)
        column_relative = np.zeros_like(distance)
        for row in range(distance.shape[0]):
            selected = active[row]
            if np.any(selected):
                row_relative[row, selected] = distance[row, selected] - np.min(distance[row, selected])
        for column in range(distance.shape[1]):
            selected = active[:, column]
            if np.any(selected):
                column_relative[selected, column] = distance[selected, column] - np.min(distance[selected, column])
        relative = np.maximum(0.0, 0.5 * (row_relative + column_relative))
        bounded_negative_log_likelihood = -np.expm1(-0.5 * relative)
        term = np.zeros_like(distance)
        term[active] = (
            matrices[2][active]
            * np.clip(reliability[active], 0.0, 1.0)
            * bounded_negative_log_likelihood[active]
        )
        return term

    pose_term = relative_term(matrices[1], matrices[3])
    return matrices[0] + pose_term, pose_term


def slice_gaussian_pose(value: GaussianPose, start: int, stop: int) -> GaussianPose:
    """Return an uncertainty-aligned contiguous component view."""

    if int(start) != start or int(stop) != stop or start < 0 or stop <= start or stop > value.mean.size:
        raise ValueError("GaussianPose slice bounds are invalid")
    indices = np.arange(int(start), int(stop), dtype=int)
    return GaussianPose(
        mean=value.mean[indices],
        covariance=value.covariance[np.ix_(indices, indices)],
        valid_mask=value.valid_mask[indices],
        quality=value.quality,
        circular_mask=value.circular_mask[indices],
    )
