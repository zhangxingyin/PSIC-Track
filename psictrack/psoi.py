"""PSOI: topology-aware pose state, causal memory, and observation integrity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from .association import GaussianPose
from .config import ObservationIntegrityConfig, PoseConfig
from .types import Observation, UpdateMode
# Numerical safeguards and fixed pose-memory model constants.
_NUMERICAL_EPS = np.finfo(np.float64).eps
_POSE_VARIANCE_FLOOR = 1e-8
_POSE_VARIANCE_CEILING = 25.0
_POSE_MEMORY_PROCESS_VARIANCE = 0.0025
_POSE_MEMORY_SURVIVAL = 0.97


def normalize_skeleton(
    skeleton: Iterable[Iterable[int]],
    keypoint_count: int,
    index_base: int | str = "auto",
) -> tuple[tuple[int, int], ...]:
    edges = tuple(tuple(int(value) for value in edge) for edge in skeleton)
    if any(len(edge) != 2 for edge in edges):
        raise ValueError("every skeleton edge must contain two indices")
    values = [value for edge in edges for value in edge]
    if index_base == "auto":
        base = 0 if 0 in values else 1
    else:
        base = int(index_base)
    normalized = tuple((left - base, right - base) for left, right in edges)
    if any(left < 0 or right < 0 or left >= keypoint_count or right >= keypoint_count for left, right in normalized):
        raise ValueError("skeleton index outside keypoint range")
    if any(left == right for left, right in normalized):
        raise ValueError("skeleton self-edges are not allowed")
    return normalized


@dataclass(frozen=True)
class SpeciesTopology:
    species: str
    keypoints: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...]
    body_axis: tuple[int, int]

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoints)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpeciesTopology":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        keypoints = tuple(str(value) for value in payload["keypoints"])
        count = len(keypoints)
        skeleton = normalize_skeleton(
            payload["skeleton"], count, payload.get("skeleton_index_base", "auto")
        )
        body_axis = tuple(int(value) for value in payload["body_axis"])
        if len(body_axis) != 2 or any(index < 0 or index >= count for index in body_axis):
            raise ValueError("body_axis must contain two valid keypoint indices")
        return cls(
            species=str(payload["species"]),
            keypoints=keypoints,
            skeleton=skeleton,
            body_axis=(body_axis[0], body_axis[1]),
        )


@dataclass(frozen=True)
class PoseEvidence:
    mean: np.ndarray
    covariance: np.ndarray
    valid_mask: np.ndarray
    quality: float
    valid_keypoints: int
    feature_names: tuple[str, ...]


class PoseEncoder:
    def __init__(self, topology: SpeciesTopology, config: PoseConfig):
        self.topology = topology
        self.config = config
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for name in self.topology.keypoints:
            names.extend((f"kp:{name}:x", f"kp:{name}:y"))
        for left, right in self.topology.skeleton:
            edge_name = f"edge:{self.topology.keypoints[left]}->{self.topology.keypoints[right]}"
            names.extend((f"{edge_name}:dx", f"{edge_name}:dy"))
        names.extend(("body_axis:x", "body_axis:y"))
        return tuple(names)

    def encode(self, observation: Observation) -> PoseEvidence:
        if observation.keypoints is None or observation.keypoint_scores is None:
            raise ValueError("PoseEncoder requires keypoints and keypoint_scores")
        if observation.keypoints.shape[0] != self.topology.keypoint_count:
            raise ValueError(
                f"expected {self.topology.keypoint_count} keypoints for {self.topology.species}, "
                f"got {observation.keypoints.shape[0]}"
            )
        bbox = observation.bbox_xywh
        center = bbox[:2] + bbox[2:] / 2.0
        scale = float(np.sqrt(bbox[2] * bbox[3]))
        points = (observation.keypoints - center) / max(scale, 1e-9)
        scores = observation.keypoint_scores
        valid_points = np.isfinite(scores) & (scores > 0.0)
        bbox_lower = bbox[:2]
        bbox_upper = bbox[:2] + bbox[2:]
        outside = np.maximum(
            np.maximum(bbox_lower - observation.keypoints, observation.keypoints - bbox_upper),
            0.0,
        )
        normalized_outside = outside / np.maximum(bbox[2:], _NUMERICAL_EPS)
        consistency_scale = max(
            0.02, np.sqrt(_NUMERICAL_EPS)
        )
        inside_probability = np.exp(
            -0.5 * np.sum(np.square(normalized_outside / consistency_scale), axis=1)
        )
        point_count = self.topology.keypoint_count
        edge_count = len(self.topology.skeleton)
        dimension = 2 * point_count + 2 * edge_count + 2
        mean = np.zeros(dimension, dtype=np.float64)
        valid_mask = np.zeros(dimension, dtype=bool)
        diagonal = np.full(dimension, _POSE_VARIANCE_CEILING, dtype=np.float64)

        point_variance = np.full(
            point_count,
            0.02 ** 2,
            dtype=np.float64,
        ) / np.maximum(scores, _NUMERICAL_EPS)
        point_variance = np.clip(
            point_variance, _POSE_VARIANCE_FLOOR, _POSE_VARIANCE_CEILING
        )

        for index in range(point_count):
            feature_slice = slice(2 * index, 2 * index + 2)
            mean[feature_slice] = points[index]
            if valid_points[index]:
                valid_mask[feature_slice] = True
                diagonal[feature_slice] = point_variance[index]

        edge_offset = 2 * point_count
        for edge_index, (left, right) in enumerate(self.topology.skeleton):
            feature_slice = slice(edge_offset + 2 * edge_index, edge_offset + 2 * edge_index + 2)
            mean[feature_slice] = points[right] - points[left]
            if valid_points[left] and valid_points[right]:
                valid_mask[feature_slice] = True
                diagonal[feature_slice] = point_variance[left] + point_variance[right]

        axis_offset = edge_offset + 2 * edge_count
        axis_left, axis_right = self.topology.body_axis
        axis = points[axis_right] - points[axis_left]
        axis_norm = float(np.linalg.norm(axis))
        if valid_points[axis_left] and valid_points[axis_right] and axis_norm > 1e-9:
            mean[axis_offset : axis_offset + 2] = axis / axis_norm
            valid_mask[axis_offset : axis_offset + 2] = True
            diagonal[axis_offset : axis_offset + 2] = (
                point_variance[axis_left] + point_variance[axis_right]
            ) / (axis_norm * axis_norm)

        covariance = np.diag(
            np.clip(diagonal, _POSE_VARIANCE_FLOOR, _POSE_VARIANCE_CEILING)
        )
        valid_stds = np.sqrt(point_variance[valid_points])
        if valid_stds.size:
            common_std = 0.1 * float(np.mean(valid_stds))
            x_vector = np.zeros(dimension, dtype=np.float64)
            y_vector = np.zeros(dimension, dtype=np.float64)
            for index in np.flatnonzero(valid_points):
                x_vector[2 * index] = common_std
                y_vector[2 * index + 1] = common_std
            covariance += np.outer(x_vector, x_vector) + np.outer(y_vector, y_vector)
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
        covariance = (
            eigenvectors
            * np.clip(eigenvalues, _POSE_VARIANCE_FLOOR, _POSE_VARIANCE_CEILING)
        ) @ eigenvectors.T

        valid_fraction = float(np.mean(valid_points))
        confidence_quality = (
            float(np.mean(np.clip(scores[valid_points], 0.0, 1.0)))
            if np.any(valid_points)
            else 0.0
        )
        inside_quality = (
            float(np.mean(inside_probability[valid_points]))
            if np.any(valid_points)
            else 0.0
        )
        quality = float(
            np.clip(valid_fraction * confidence_quality * inside_quality, 0.0, 1.0)
        )
        if int(np.count_nonzero(valid_points)) < 2:
            valid_mask[:] = False
            quality = 0.0

        for array in (mean, covariance, valid_mask):
            array.setflags(write=False)
        return PoseEvidence(
            mean=mean,
            covariance=covariance,
            valid_mask=valid_mask,
            quality=quality,
            valid_keypoints=int(np.count_nonzero(valid_points)),
            feature_names=self._feature_names,
        )


SPECTRAL_SCALES = (0.25, 0.5, 1.0, 2.0)


GRAPH_POSE_NAMES = (
    "graph_log_energy",
    "graph_heat_0.25",
    "graph_heat_0.5",
    "graph_heat_1.0",
    "graph_heat_2.0",
    "graph_dirichlet_fraction",
    "body_axis_x",
    "body_axis_y",
)


def _readonly(value: object, dtype=np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GraphPoseEvidence:
    mean: np.ndarray
    covariance: np.ndarray
    valid_mask: np.ndarray
    quality: float
    names: tuple[str, ...] = GRAPH_POSE_NAMES

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if mean.shape != (8,) or covariance.shape != (8, 8) or valid.shape != (8,):
            raise ValueError("graph pose must have fixed 8-dimensional state")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("graph pose contains non-finite values")
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValueError("graph pose quality must be in [0, 1]")
        if len(self.names) != 8:
            raise ValueError("graph pose must provide eight names")
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "covariance", _readonly(covariance))
        object.__setattr__(self, "valid_mask", _readonly(valid, bool))
        object.__setattr__(self, "quality", float(self.quality))


def spectral_energy_features(
    eigenvalues: np.ndarray,
    basis: np.ndarray,
    centered_points: np.ndarray,
    scales: tuple[float, ...] = SPECTRAL_SCALES,
) -> np.ndarray:
    """Return basis-invariant graph-filter energies for one valid pose graph."""

    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(basis, dtype=np.float64)
    points = np.asarray(centered_points, dtype=np.float64)
    if values.ndim != 1 or vectors.shape != (values.size, values.size):
        raise ValueError("eigendecomposition shapes are inconsistent")
    if points.shape != (values.size, 2):
        raise ValueError("centered_points must have shape (N, 2)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(points)):
        raise ValueError("spectral inputs must be finite")
    coefficients = vectors.T @ points
    modal_energy = np.sum(np.square(coefficients), axis=1)
    total = max(float(np.sum(modal_energy)), 1e-12)
    heat = [
        float(np.sum(np.exp(-2.0 * scale * values) * modal_energy) / total)
        for scale in scales
    ]
    dirichlet = float(np.sum(values * modal_energy) / (2.0 * total))
    return np.asarray([np.log1p(total), *heat, np.clip(dirichlet, 0.0, 1.0)])


class GraphPoseEncoder:
    """Encode a supplied body topology into one shared graph-spectral state."""

    def __init__(self, topology: SpeciesTopology, config: PoseConfig):
        self.topology = topology
        self.config = config

    def _valid_laplacian(self, valid_indices: np.ndarray) -> tuple[np.ndarray, int]:
        lookup = {int(original): local for local, original in enumerate(valid_indices.tolist())}
        adjacency = np.zeros((len(valid_indices), len(valid_indices)), dtype=np.float64)
        edge_count = 0
        for left, right in self.topology.skeleton:
            if left in lookup and right in lookup:
                a, b = lookup[left], lookup[right]
                adjacency[a, b] = adjacency[b, a] = 1.0
                edge_count += 1
        degree = np.sum(adjacency, axis=1)
        inverse = np.zeros_like(degree)
        positive = degree > 0
        inverse[positive] = 1.0 / np.sqrt(degree[positive])
        laplacian = np.diag(positive.astype(np.float64)) - (
            inverse[:, None] * adjacency * inverse[None, :]
        )
        return (laplacian + laplacian.T) / 2.0, edge_count

    def encode(self, observation: Observation) -> GraphPoseEvidence:
        if observation.keypoints is None or observation.keypoint_scores is None:
            raise ValueError("graph pose encoding requires keypoints and scores")
        if observation.keypoints.shape[0] != self.topology.keypoint_count:
            raise ValueError("keypoint count does not match species topology")
        bbox = observation.bbox_xywh
        center = bbox[:2] + bbox[2:] / 2.0
        scale = max(float(np.sqrt(bbox[2] * bbox[3])), 1e-9)
        points = (observation.keypoints - center) / scale
        scores = observation.keypoint_scores
        valid = np.isfinite(scores) & (scores > 0.0)
        lower = bbox[:2]
        upper = bbox[:2] + bbox[2:]
        outside = np.maximum(
            np.maximum(lower - observation.keypoints, observation.keypoints - upper),
            0.0,
        )
        normalized_outside = outside / np.maximum(bbox[2:], _NUMERICAL_EPS)
        consistency_scale = max(
            0.02, np.sqrt(_NUMERICAL_EPS)
        )
        inside_probability = np.exp(
            -0.5 * np.sum(np.square(normalized_outside / consistency_scale), axis=1)
        )
        indices = np.flatnonzero(valid)

        mean = np.zeros(8, dtype=np.float64)
        valid_mask = np.zeros(8, dtype=bool)
        diagonal = np.full(8, _POSE_VARIANCE_CEILING, dtype=np.float64)
        edge_count = 0
        if indices.size >= 2:
            valid_points = points[indices]
            weights = np.clip(scores[indices], _NUMERICAL_EPS, 1.0)
            centroid = np.average(
                valid_points, axis=0, weights=weights
            )
            centered = valid_points - centroid
            laplacian, edge_count = self._valid_laplacian(indices)
            if edge_count > 0:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                mean[:6] = spectral_energy_features(eigenvalues, eigenvectors, centered)
                valid_mask[:6] = True
                point_uncertainty = float(
                    np.mean(
                        (0.02 ** 2)
                        / np.maximum(scores[indices], _NUMERICAL_EPS)
                    )
                )
                diagonal[:6] = np.clip(
                    point_uncertainty,
                    _POSE_VARIANCE_FLOOR,
                    _POSE_VARIANCE_CEILING,
                )

        axis_left, axis_right = self.topology.body_axis
        axis = points[axis_right] - points[axis_left]
        axis_length = float(np.linalg.norm(axis))
        if valid[axis_left] and valid[axis_right] and axis_length > 1e-9:
            mean[6:8] = axis / axis_length
            valid_mask[6:8] = True
            axis_indices = np.asarray([axis_left, axis_right], dtype=int)
            axis_uncertainty = float(
                np.mean(
                    (0.02 ** 2)
                    / np.maximum(scores[axis_indices], _NUMERICAL_EPS)
                )
            )
            diagonal[6:8] = np.clip(
                axis_uncertainty / max(axis_length * axis_length, _NUMERICAL_EPS),
                _POSE_VARIANCE_FLOOR,
                _POSE_VARIANCE_CEILING,
            )

        coverage = float(indices.size / max(self.topology.keypoint_count, 1))
        confidence = (
            float(np.mean(np.clip(scores[indices], 0.0, 1.0)))
            if indices.size
            else 0.0
        )
        inside_quality = (
            float(np.mean(inside_probability[indices])) if indices.size else 0.0
        )
        connectivity = float(
            np.clip(2.0 * edge_count / max(indices.size, 1), 0.0, 1.0)
        )
        quality = coverage * confidence * inside_quality * connectivity
        if indices.size < 2 or edge_count == 0:
            valid_mask[:] = False
            quality = 0.0
        covariance = np.diag(diagonal)
        return GraphPoseEvidence(mean, covariance, valid_mask, float(np.clip(quality, 0.0, 1.0)))


def graph_pose_distance(left: GraphPoseEvidence, right: GraphPoseEvidence) -> float:
    common = left.valid_mask & right.valid_mask
    indices = np.flatnonzero(common)
    if indices.size == 0:
        return float("inf")
    residual = left.mean[indices] - right.mean[indices]
    covariance = left.covariance[np.ix_(indices, indices)] + right.covariance[np.ix_(indices, indices)]
    covariance += np.eye(indices.size) * 1e-8
    try:
        value = float(residual @ np.linalg.solve(covariance, residual))
    except np.linalg.LinAlgError:
        return float("inf")
    return value / float(indices.size)


class PoseMemory:
    def __init__(
        self,
        evidence: PoseEvidence,
        topology: SpeciesTopology,
        pose_config: PoseConfig,
    ):
        self.topology = topology
        self.pose_config = pose_config
        self.pose_mean = evidence.mean.copy()
        self.pose_covariance = evidence.covariance.copy()
        self.pose_valid = evidence.valid_mask.copy()
        self.quality = evidence.quality
        self._predicted = False

    def _assemble(
        self,
        pose_mean: np.ndarray,
        pose_covariance: np.ndarray,
        pose_valid: np.ndarray,
        quality: float,
    ) -> GaussianPose | None:
        if not self.pose_config.enabled:
            return None
        return GaussianPose(
            mean=pose_mean,
            covariance=pose_covariance,
            valid_mask=pose_valid,
            quality=float(np.clip(quality, 0.0, 1.0)),
            circular_mask=np.zeros(pose_mean.size, dtype=bool),
        )

    def observation(self, evidence: PoseEvidence) -> GaussianPose | None:
        return self._assemble(
            evidence.mean,
            evidence.covariance,
            evidence.valid_mask,
            evidence.quality,
        )

    def predict(self, delta_frames: int) -> GaussianPose | None:
        if self._predicted:
            raise RuntimeError("pose memory must be updated or missed before another prediction")
        dt = int(delta_frames)
        self.pose_covariance = (
            self.pose_covariance
            + np.eye(self.pose_mean.size) * _POSE_MEMORY_PROCESS_VARIANCE * dt
        )
        self.quality *= _POSE_MEMORY_SURVIVAL**dt
        self._predicted = True
        return self._assemble(
            self.pose_mean,
            self.pose_covariance,
            self.pose_valid,
            self.quality,
        )

    def update(self, evidence: PoseEvidence, mode: UpdateMode) -> None:
        if not self._predicted:
            raise RuntimeError("pose memory predict must precede update")
        if mode is UpdateMode.FROZEN:
            self.miss()
            return
        strength = 1.0 if mode is UpdateMode.NORMAL else 0.25
        common = self.pose_valid & evidence.valid_mask
        self.pose_mean[common] = (
            (1.0 - strength) * self.pose_mean[common] + strength * evidence.mean[common]
        )
        self.pose_covariance[np.ix_(common, common)] = (
            (1.0 - strength) * self.pose_covariance[np.ix_(common, common)]
            + strength * evidence.covariance[np.ix_(common, common)]
        )
        new_pose = (~self.pose_valid) & evidence.valid_mask
        self.pose_mean[new_pose] = evidence.mean[new_pose]
        self.pose_valid |= evidence.valid_mask
        self.quality = (1.0 - strength) * self.quality + strength * evidence.quality
        self._predicted = False

    def miss(self) -> None:
        if not self._predicted:
            return
        self._predicted = False


@dataclass(frozen=True)
class ObservationIntegrity:
    singleton_probability: float
    duplicate_probability: float
    merge_probability: float
    clutter_probability: float
    duplicate_group: int | None = None


def _intersection(left: np.ndarray, right: np.ndarray) -> float:
    width = max(0.0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1]))
    return width * height


def _overlap_min_area(left: np.ndarray, right: np.ndarray) -> float:
    denominator = min(float(left[2] * left[3]), float(right[2] * right[3]))
    return _intersection(left, right) / denominator if denominator > 0 else 0.0


def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_center = left[:2] + left[2:] / 2.0
    right_center = right[:2] + right[2:] / 2.0
    scale = max(np.sqrt(0.5 * (left[2] * left[3] + right[2] * right[3])), 1e-9)
    return float(np.linalg.norm(left_center - right_center) / scale)


def _keypoint_occupancy_agreement(
    left: Observation, right: Observation
) -> float | None:
    """Confidence-calibrated agreement in image coordinates.

    Normalized graph pose can change when two detector crops cover different
    fractions of the same animal.  Image-coordinate occupancy supplies the
    complementary question: did both top-down pose predictions land on the
    same physical body?  Its scale and reliability are derived from the boxes
    and keypoint scores, so it introduces no species-specific constant.
    """

    if left.keypoints is None or right.keypoints is None:
        return None
    if left.keypoints.shape != right.keypoints.shape:
        return None
    weights = np.sqrt(left.keypoint_scores * right.keypoint_scores)
    support = float(np.mean(weights)) if weights.size else 0.0
    weight_sum = float(np.sum(weights))
    if weight_sum <= _NUMERICAL_EPS:
        return 0.0
    squared_distance = np.sum((left.keypoints - right.keypoints) ** 2, axis=1)
    body_scale_squared = max(
        0.5
        * (
            float(left.bbox_xywh[2] * left.bbox_xywh[3])
            + float(right.bbox_xywh[2] * right.bbox_xywh[3])
        ),
        _NUMERICAL_EPS,
    )
    normalized_distance = float(
        np.sum(weights * squared_distance)
        / (weight_sum * body_scale_squared)
    )
    return float(np.clip(support * np.exp(-0.5 * normalized_distance), 0.0, 1.0))


class ObservationIntegrityModel:
    def __init__(self, config: ObservationIntegrityConfig):
        config.validate()
        self.config = config

    def duplicate_affinity(
        self,
        observations: Sequence[Observation],
        graph_poses: Sequence[GraphPoseEvidence | None],
    ) -> np.ndarray:
        """Continuous probability that two detections explain the same body."""

        if len(observations) != len(graph_poses):
            raise ValueError("one graph pose entry is required per observation")
        count = len(observations)
        affinity = np.zeros((count, count), dtype=np.float64)
        if not self.config.enabled:
            return affinity
        for left in range(count):
            for right in range(left + 1, count):
                overlap = _overlap_min_area(
                    observations[left].bbox_xywh, observations[right].bbox_xywh
                )
                center = _center_distance(
                    observations[left].bbox_xywh, observations[right].bbox_xywh
                )
                left_pose, right_pose = graph_poses[left], graph_poses[right]
                if (
                    left_pose is not None
                    and right_pose is not None
                    and min(left_pose.quality, right_pose.quality) > 0.0
                ):
                    distance = graph_pose_distance(left_pose, right_pose)
                    pose_agreement = (
                        0.0
                        if not np.isfinite(distance)
                        else float(np.exp(-0.5 * distance))
                    )
                else:
                    pose_agreement = 0.5
                occupancy_agreement = _keypoint_occupancy_agreement(
                    observations[left], observations[right]
                )
                if occupancy_agreement is not None:
                    pose_agreement = max(pose_agreement, occupancy_agreement)
                center_likelihood = float(np.exp(-0.5 * center * center))
                strength = float(
                    np.clip(
                        overlap * center_likelihood * pose_agreement,
                        0.0,
                        1.0,
                    )
                )
                affinity[left, right] = strength
                affinity[right, left] = strength
        return affinity

    def assess(
        self,
        observations: Sequence[Observation],
        graph_poses: Sequence[GraphPoseEvidence | None],
        predicted_boxes: Sequence[np.ndarray],
    ) -> tuple[ObservationIntegrity, ...]:
        if len(observations) != len(graph_poses):
            raise ValueError("one graph pose entry is required per observation")
        count = len(observations)
        affinity = self.duplicate_affinity(observations, graph_poses)
        duplicates = np.zeros(count, dtype=np.float64)
        parent = list(range(count))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        if self.config.enabled:
            for left in range(count):
                for right in range(left + 1, count):
                    left_pose, right_pose = graph_poses[left], graph_poses[right]
                    strength = float(affinity[left, right])
                    left_quality = observations[left].score * (
                        0.5 + 0.5 * (0.0 if left_pose is None else left_pose.quality)
                    )
                    right_quality = observations[right].score * (
                        0.5 + 0.5 * (0.0 if right_pose is None else right_pose.quality)
                    )
                    lower = left if left_quality <= right_quality else right
                    duplicates[lower] = max(duplicates[lower], strength)
                    if strength >= 0.5:
                        union(left, right)

        merges = np.zeros(count, dtype=np.float64)
        if self.config.enabled and predicted_boxes:
            for index, observation in enumerate(observations):
                box = observation.bbox_xywh
                covered = 0
                for predicted in predicted_boxes:
                    candidate = np.asarray(predicted, dtype=np.float64)
                    center = candidate[:2] + candidate[2:] / 2.0
                    inside = float(
                        box[0] <= center[0] <= box[0] + box[2]
                        and box[1] <= center[1] <= box[1] + box[3]
                    )
                    containment = _intersection(box, candidate) / max(
                        candidate[2] * candidate[3], _NUMERICAL_EPS
                    )
                    covered += inside * containment
                merges[index] = 1.0 - float(
                    np.exp(-max(float(covered) - 1.0, 0.0))
                )

        group_roots = [find(index) for index in range(count)]
        group_sizes = {root: group_roots.count(root) for root in set(group_roots)}
        groups = {root: group for group, root in enumerate(sorted(root for root, size in group_sizes.items() if size > 1))}
        result = []
        for index, observation in enumerate(observations):
            duplicate = float(np.clip(duplicates[index], 0.0, 1.0))
            merge = float(np.clip(merges[index], 0.0, 1.0))
            singleton = float(
                np.clip((1.0 - duplicate) * (1.0 - merge), 0.0, 1.0)
            )
            clutter = duplicate
            result.append(
                ObservationIntegrity(
                    singleton_probability=singleton,
                    duplicate_probability=duplicate,
                    merge_probability=merge,
                    clutter_probability=clutter,
                    duplicate_group=groups.get(group_roots[index]),
                )
            )
        return tuple(result)
