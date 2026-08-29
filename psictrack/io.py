"""Prediction-input adapters for strictly online PSIC-Track."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Observation, ObservationSource


@dataclass(frozen=True)
class MotRow:
    frame: int
    row_index: int
    input_id: int
    bbox_xywh: np.ndarray
    score: float


@dataclass(frozen=True)
class PoseRow:
    frame: int
    pose_id: int
    detection_id: int | None
    bbox_xywh: np.ndarray
    keypoints: np.ndarray
    keypoint_scores: np.ndarray


def _array(value: object, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("array contains non-finite values")
    result.setflags(write=False)
    return result


def read_mot(path: str | Path, min_score: float = 0.0) -> dict[int, list[MotRow]]:
    """Read detector output in MOTChallenge ``frame,id,x,y,w,h,score,...`` format.

    The input ID is accepted for interoperability but is not used as a tracking
    identity.  Row order within each frame defines the zero-based ``detection_id``
    used to bind pose records.
    """

    frames: dict[int, list[MotRow]] = {}
    per_frame_index: dict[int, int] = {}
    source = Path(path)
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < 7:
            raise ValueError(f"{source}:{line_number}: expected at least 7 MOT columns")
        frame = int(float(parts[0]))
        input_id = int(float(parts[1]))
        bbox = _array([float(value) for value in parts[2:6]], (4,))
        score = float(parts[6])
        if frame < 1 or bbox[2] <= 0 or bbox[3] <= 0 or not np.isfinite(score):
            raise ValueError(f"{source}:{line_number}: invalid frame, bbox, or score")
        if score < min_score:
            continue
        row_index = per_frame_index.get(frame, 0)
        per_frame_index[frame] = row_index + 1
        frames.setdefault(frame, []).append(MotRow(frame, row_index, input_id, bbox, score))
    return frames


def read_pose_json(path: str | Path) -> dict[int, list[PoseRow]]:
    """Read predicted pose records from the documented JSON interchange schema."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"{source}: 'annotations' must be a JSON list")

    frames: dict[int, list[PoseRow]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"{source}: each pose annotation must be a JSON object")
        frame = int(annotation.get("frame_id", annotation.get("mot_frame_id", 0)))
        if frame < 1:
            raise ValueError(f"pose annotation {annotation.get('id')} has invalid frame")
        flat = np.asarray(annotation["keypoints"], dtype=np.float64)
        if flat.ndim != 1 or flat.size % 3:
            raise ValueError(f"pose annotation {annotation.get('id')} has invalid keypoints")
        triplets = flat.reshape(-1, 3)
        visible = triplets[:, 2] > 0
        confidence_values = annotation.get("keypoint_confidences")
        if confidence_values is None:
            scores = visible.astype(np.float64)
        else:
            scores = np.asarray(confidence_values, dtype=np.float64)
            if scores.shape != visible.shape:
                raise ValueError(f"pose annotation {annotation.get('id')} confidence shape mismatch")
            scores = np.where(visible, scores, 0.0)
        bbox = _array(annotation["bbox"], (4,))
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"pose annotation {annotation.get('id')} has invalid bbox")
        raw_detection_id = annotation.get("detection_id")
        detection_id = None if raw_detection_id is None else int(raw_detection_id)
        if detection_id is not None and detection_id < 0:
            raise ValueError(f"pose annotation {annotation.get('id')} has invalid detection_id")
        frames.setdefault(frame, []).append(
            PoseRow(
                frame=frame,
                pose_id=int(annotation.get("id", len(frames.get(frame, [])))),
                detection_id=detection_id,
                bbox_xywh=bbox,
                keypoints=_array(triplets[:, :2]),
                keypoint_scores=_array(scores),
            )
        )
    return frames


def bbox_iou_xywh(left: np.ndarray, right: np.ndarray) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    iw = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    ih = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = iw * ih
    union = lw * lh + rw * rh - intersection
    return float(intersection / union) if union > 0 else 0.0


def merge_box_pose_frames(
    boxes: dict[int, list[MotRow]],
    poses: dict[int, list[PoseRow]] | None = None,
    *,
    iou_threshold: float = 0.99,
    sequence_length: int | None = None,
) -> Iterator[tuple[int, list[Observation]]]:
    """Attach pose records to detections by ID, or by IoU when no ID is given."""

    poses = poses or {}
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    inferred_length = max([0, *boxes.keys(), *poses.keys()])
    if sequence_length is not None:
        if int(sequence_length) != sequence_length or sequence_length < inferred_length:
            raise ValueError("sequence_length must be a positive integer no smaller than the last input frame")
        maximum = int(sequence_length)
    else:
        maximum = inferred_length
    if maximum < 1:
        raise ValueError("at least one input frame is required when sequence_length is omitted")

    for frame in range(1, maximum + 1):
        frame_boxes = boxes.get(frame, [])
        frame_poses = poses.get(frame, [])
        pose_for_box: dict[int, PoseRow] = {}
        geometric_poses: list[PoseRow] = []
        for pose in frame_poses:
            if pose.detection_id is None:
                geometric_poses.append(pose)
                continue
            if pose.detection_id >= len(frame_boxes):
                raise ValueError(f"frame {frame}: pose detection_id {pose.detection_id} outside detection range")
            if pose.detection_id in pose_for_box:
                raise ValueError(f"frame {frame}: duplicate pose detection_id {pose.detection_id}")
            pose_for_box[pose.detection_id] = pose

        available_boxes = [index for index in range(len(frame_boxes)) if index not in pose_for_box]
        if available_boxes and geometric_poses:
            ious = np.asarray(
                [
                    [bbox_iou_xywh(frame_boxes[box_index].bbox_xywh, pose.bbox_xywh) for pose in geometric_poses]
                    for box_index in available_boxes
                ],
                dtype=np.float64,
            )
            rows, cols = linear_sum_assignment(1.0 - ious)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if ious[row, col] >= iou_threshold:
                    pose_for_box[available_boxes[row]] = geometric_poses[col]

        observations: list[Observation] = []
        for detection_id, box in enumerate(frame_boxes):
            pose = pose_for_box.get(detection_id)
            observations.append(
                Observation(
                    frame=frame,
                    detection_id=detection_id,
                    bbox_xywh=box.bbox_xywh,
                    score=min(1.0, max(0.0, box.score)),
                    keypoints=None if pose is None else pose.keypoints,
                    keypoint_scores=None if pose is None else pose.keypoint_scores,
                    source=ObservationSource.PREDICTION,
                    pose_source=None if pose is None else ObservationSource.PREDICTION,
                )
            )
        yield frame, observations


def load_prediction_observations(
    detections_path: str | Path,
    poses_path: str | Path | None = None,
    *,
    sequence_length: int | None = None,
    iou_threshold: float = 0.99,
) -> Iterator[tuple[int, list[Observation]]]:
    """Construct an online observation stream directly from prediction files."""

    boxes = read_mot(detections_path)
    poses = {} if poses_path is None else read_pose_json(poses_path)
    return merge_box_pose_frames(
        boxes, poses, iou_threshold=iou_threshold, sequence_length=sequence_length
    )


def _observation_dict(observation: Observation) -> dict:
    return {
        "detection_id": observation.detection_id,
        "bbox_xywh": observation.bbox_xywh.tolist(),
        "score": observation.score,
        "keypoints": None if observation.keypoints is None else observation.keypoints.tolist(),
        "keypoint_scores": None if observation.keypoint_scores is None else observation.keypoint_scores.tolist(),
        "source": observation.source.value,
        "pose_source": None if observation.pose_source is None else observation.pose_source.value,
    }


def write_observation_stream(
    path: str | Path,
    frames: Iterable[tuple[int, Sequence[Observation]]],
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for frame, observations in frames:
            payload = {"frame": int(frame), "observations": [_observation_dict(item) for item in observations]}
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_observation_stream(path: str | Path) -> Iterator[tuple[int, list[Observation]]]:
    previous_frame = 0
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        frame = int(payload["frame"])
        if frame <= previous_frame:
            raise ValueError(f"{path}:{line_number}: frames must be strictly increasing")
        previous_frame = frame
        observations = []
        for item in payload.get("observations", []):
            observations.append(
                Observation(
                    frame=frame,
                    detection_id=int(item["detection_id"]),
                    bbox_xywh=np.asarray(item["bbox_xywh"], dtype=np.float64),
                    score=float(item["score"]),
                    keypoints=None if item.get("keypoints") is None else np.asarray(item["keypoints"], dtype=np.float64),
                    keypoint_scores=None
                    if item.get("keypoint_scores") is None
                    else np.asarray(item["keypoint_scores"], dtype=np.float64),
                    source=ObservationSource(item["source"]),
                    pose_source=None
                    if item.get("pose_source") is None
                    else ObservationSource(item["pose_source"]),
                )
            )
        yield frame, observations
