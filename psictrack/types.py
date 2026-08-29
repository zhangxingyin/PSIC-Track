"""Typed boundary objects shared by all PSIC-Track modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ObservationSource(str, Enum):
    PREDICTION = "prediction"
    PROPAGATED = "propagated"


class Lifecycle(str, Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    DORMANT = "dormant"
    RECOVERED = "recovered"
    REMOVED = "removed"


class UpdateMode(str, Enum):
    NORMAL = "normal"
    WEAK = "weak"
    FROZEN = "frozen"


class MotionRegime(str, Enum):
    UNKNOWN = "unknown"
    QUIESCENT = "quiescent"
    PERIODIC = "periodic"
    APERIODIC = "aperiodic"
    CHANGE_POINT = "change_point"


def _readonly_float_array(
    value: object, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Observation:
    frame: int
    detection_id: int
    bbox_xywh: np.ndarray
    score: float
    keypoints: np.ndarray | None = None
    keypoint_scores: np.ndarray | None = None
    source: ObservationSource = ObservationSource.PREDICTION
    pose_source: ObservationSource | None = None

    def __post_init__(self) -> None:
        if int(self.frame) != self.frame or self.frame < 1:
            raise ValueError("frame must be a positive integer")
        if int(self.detection_id) != self.detection_id or self.detection_id < 0:
            raise ValueError("detection_id must be a non-negative integer")
        bbox = _readonly_float_array(self.bbox_xywh, name="bbox_xywh", shape=(4,))
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("bbox width and height must be positive")
        if not np.isfinite(self.score) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        source = self.source if isinstance(self.source, ObservationSource) else ObservationSource(self.source)
        keypoints = None
        keypoint_scores = None
        if self.keypoints is not None:
            keypoints = _readonly_float_array(self.keypoints, name="keypoints")
            if keypoints.ndim != 2 or keypoints.shape[1] != 2:
                raise ValueError(f"keypoints must have shape (K, 2), got {keypoints.shape}")
            if self.keypoint_scores is None:
                keypoint_scores = np.ones(keypoints.shape[0], dtype=np.float64)
                keypoint_scores.setflags(write=False)
            else:
                keypoint_scores = _readonly_float_array(
                    self.keypoint_scores, name="keypoint_scores"
                )
                if keypoint_scores.shape != (keypoints.shape[0],):
                    raise ValueError(
                        "keypoint_scores must have shape (K,) matching keypoints; "
                        f"got {keypoint_scores.shape} for K={keypoints.shape[0]}"
                    )
                if np.any((keypoint_scores < 0.0) | (keypoint_scores > 1.0)):
                    raise ValueError("keypoint_scores must be in [0, 1]")
        elif self.keypoint_scores is not None:
            raise ValueError("keypoint_scores cannot be provided without keypoints")
        pose_source = self.pose_source
        if keypoints is None:
            if pose_source is not None:
                raise ValueError("pose_source cannot be provided without keypoints")
        else:
            pose_source = source if pose_source is None else ObservationSource(pose_source)
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "detection_id", int(self.detection_id))
        object.__setattr__(self, "bbox_xywh", bbox)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "keypoint_scores", keypoint_scores)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "pose_source", pose_source)


@dataclass(frozen=True)
class TrackOutput:
    frame: int
    track_id: int
    bbox_xywh: np.ndarray
    score: float
    lifecycle: Lifecycle
    observed: bool
    observation_source: ObservationSource
    emit_mot: bool | None = None
    association_quality: float = 0.0
    update_mode: UpdateMode = UpdateMode.FROZEN
    motion_regime: MotionRegime = MotionRegime.UNKNOWN
    motion_expert: str = "cv"
    motion_control_weight: float = 0.0
    motion_prediction_variance: float = 0.0
    keypoints: np.ndarray | None = None
    keypoint_scores: np.ndarray | None = None
    pose_source: ObservationSource | None = None

    def __post_init__(self) -> None:
        bbox = _readonly_float_array(self.bbox_xywh, name="bbox_xywh", shape=(4,))
        object.__setattr__(self, "bbox_xywh", bbox)
        object.__setattr__(
            self,
            "emit_mot",
            bool(self.observed) if self.emit_mot is None else bool(self.emit_mot),
        )
        if not 0.0 <= float(self.motion_control_weight) <= 1.0:
            raise ValueError("motion_control_weight must be in [0, 1]")
        if not np.isfinite(self.motion_prediction_variance) or self.motion_prediction_variance < 0.0:
            raise ValueError("motion_prediction_variance must be finite and nonnegative")
        object.__setattr__(self, "motion_expert", str(self.motion_expert))
        object.__setattr__(self, "motion_control_weight", float(self.motion_control_weight))
        object.__setattr__(self, "motion_prediction_variance", float(self.motion_prediction_variance))
        if self.keypoints is not None:
            object.__setattr__(
                self, "keypoints", _readonly_float_array(self.keypoints, name="keypoints")
            )
        if self.keypoint_scores is not None:
            object.__setattr__(
                self,
                "keypoint_scores",
                _readonly_float_array(self.keypoint_scores, name="keypoint_scores"),
            )
