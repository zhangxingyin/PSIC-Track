"""Identity-stable trajectory metrics over interpolated bounding-box references."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Box:
    """An axis-aligned bounding box stored as top-left x/y and width/height."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Segment:
    """A maximal contiguous set of reference frames for one identity."""

    track_id: int
    frames: tuple[int, ...]


def load_mot_boxes(path: str | Path) -> dict[int, dict[int, Box]]:
    """Read valid `frame,id,x,y,w,h` rows from a MOT-style text file."""
    frames: dict[int, dict[int, Box]] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number} has fewer than six MOT fields")
        try:
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            box = Box(*(float(field) for field in fields[2:6]))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} contains a non-numeric MOT field") from error
        if frame_id <= 0 or track_id <= 0:
            continue
        if not all(isfinite(value) for value in (box.x, box.y, box.w, box.h)) or box.w <= 0.0 or box.h <= 0.0:
            continue
        frame = frames.setdefault(frame_id, {})
        if track_id in frame:
            raise ValueError(f"{path}:{line_number} duplicates frame {frame_id}, identity {track_id}")
        frame[track_id] = box
    return frames


def valid_segments(reference: dict[int, dict[int, Box]]) -> list[Segment]:
    """Split each identity wherever its interpolated reference is absent."""
    identity_frames: dict[int, list[int]] = {}
    for frame_id, identities in reference.items():
        for track_id in identities:
            identity_frames.setdefault(track_id, []).append(frame_id)

    segments: list[Segment] = []
    for track_id in sorted(identity_frames):
        ordered_frames = sorted(identity_frames[track_id])
        current: list[int] = []
        for frame_id in ordered_frames:
            if current and frame_id != current[-1] + 1:
                segments.append(Segment(track_id=track_id, frames=tuple(current)))
                current = []
            current.append(frame_id)
        if current:
            segments.append(Segment(track_id=track_id, frames=tuple(current)))
    return segments


@dataclass(frozen=True)
class SegmentMetrics:
    """Four trajectory-input reliability statistics for one reference segment."""

    coverage: float
    motion_distance_mae: float
    activity_fraction_absolute_error: float | None
    nearest_neighbor_relative_distance_mae: float | None
    frame_count: int
    activity_transition_count: int
    reference_active_frame_count: int
    predicted_active_frame_count: int
    reference_motion_distance: float
    predicted_motion_distance: float
    nearest_neighbor_frame_count: int


def box_center(box: Box) -> np.ndarray:
    """Return the x/y center of a box."""
    return np.asarray((box.x + box.w / 2.0, box.y + box.h / 2.0), dtype=np.float64)


def _distance(left: Box, right: Box) -> float:
    return float(np.linalg.norm(box_center(left) - box_center(right)))


def _reference_scene_diagonal(reference: dict[int, dict[int, Box]]) -> float:
    """Fallback spatial extent when sequence metadata is unavailable."""
    boxes = [box for frame in reference.values() for box in frame.values()]
    if not boxes:
        raise ValueError("reference contains no boxes")
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.w for box in boxes)
    bottom = max(box.y + box.h for box in boxes)
    return max(float(np.hypot(right - left, bottom - top)), 1e-6)


def _activity_flags(
    reference_boxes: list[Box],
    prediction_boxes: list[Box | None],
    activity_threshold: float,
) -> tuple[list[bool], list[bool]]:
    reference_active: list[bool] = []
    prediction_active: list[bool] = []
    for previous_index in range(len(reference_boxes) - 1):
        current_index = previous_index + 1
        reference_current = reference_boxes[current_index]
        scale = float(np.sqrt(reference_current.w * reference_current.h))
        reference_speed = _distance(reference_boxes[previous_index], reference_current) / scale
        reference_active.append(reference_speed > activity_threshold)

        prediction_previous = prediction_boxes[previous_index]
        prediction_current = prediction_boxes[current_index]
        if prediction_previous is None or prediction_current is None:
            prediction_active.append(False)
            continue
        prediction_speed = _distance(prediction_previous, prediction_current) / scale
        prediction_active.append(prediction_speed > activity_threshold)
    return reference_active, prediction_active


def evaluate_segment(
    reference: dict[int, dict[int, Box]],
    prediction: dict[int, dict[int, Box]],
    mapping: dict[int, int],
    segment: Segment,
    *,
    activity_threshold: float,
    iou_threshold: float = 0.5,
    scene_diagonal: float | None = None,
) -> SegmentMetrics:
    """Evaluate one reference identity without remapping predicted IDs per frame."""
    if activity_threshold < 0.0:
        raise ValueError("activity_threshold must be non-negative")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if scene_diagonal is None:
        scene_diagonal = _reference_scene_diagonal(reference)
    if not isfinite(scene_diagonal) or scene_diagonal <= 0.0:
        raise ValueError("scene_diagonal must be finite and positive")

    reference_id = segment.track_id
    prediction_id = mapping.get(reference_id)
    reference_boxes = [reference[frame_id][reference_id] for frame_id in segment.frames]
    prediction_boxes = [
        prediction.get(frame_id, {}).get(prediction_id) if prediction_id is not None else None
        for frame_id in segment.frames
    ]

    matched = [
        predicted_box is not None and box_iou(reference_box, predicted_box) >= iou_threshold
        for reference_box, predicted_box in zip(reference_boxes, prediction_boxes, strict=True)
    ]
    coverage = float(np.mean(matched))

    reference_motion = sum(
        _distance(reference_boxes[index - 1], reference_boxes[index])
        for index in range(1, len(reference_boxes))
    )
    predicted_motion = sum(
        _distance(prediction_boxes[index - 1], prediction_boxes[index])
        if prediction_boxes[index - 1] is not None and prediction_boxes[index] is not None
        else 0.0
        for index in range(1, len(prediction_boxes))
    )
    motion_error = abs(predicted_motion - reference_motion)

    reference_active, prediction_active = _activity_flags(
        reference_boxes, prediction_boxes, activity_threshold
    )
    if reference_active:
        reference_active_count = sum(reference_active)
        prediction_active_count = sum(prediction_active)
        activity_fraction_error: float | None = abs(
            float(np.mean(prediction_active)) - float(np.mean(reference_active))
        )
    else:
        activity_fraction_error = None
        reference_active_count = 0
        prediction_active_count = 0

    neighbor_errors: list[float] = []
    for index, (frame_id, reference_box, predicted_box) in enumerate(zip(
        segment.frames, reference_boxes, prediction_boxes, strict=True
    )):
        reference_frame = reference[frame_id]
        reference_neighbors = [
            (other_id, other_box)
            for other_id, other_box in reference_frame.items()
            if other_id != reference_id
        ]
        if not reference_neighbors:
            continue
        scale = float(np.sqrt(reference_box.w * reference_box.h))
        nearest_reference_id, nearest_reference_box = min(
            reference_neighbors,
            key=lambda neighbor: _distance(reference_box, neighbor[1]),
        )
        reference_nearest = _distance(reference_box, nearest_reference_box)
        nearest_prediction_id = mapping.get(nearest_reference_id)
        nearest_prediction_box = (
            prediction.get(frame_id, {}).get(nearest_prediction_id)
            if nearest_prediction_id is not None
            else None
        )
        if (
            not matched[index]
            or nearest_prediction_box is None
            or box_iou(nearest_reference_box, nearest_prediction_box) < iou_threshold
        ):
            maximum_error = max(reference_nearest, max(0.0, scene_diagonal - reference_nearest))
            neighbor_errors.append(maximum_error / scale)
            continue
        predicted_nearest = _distance(predicted_box, nearest_prediction_box)
        neighbor_errors.append(abs(predicted_nearest - reference_nearest) / scale)

    return SegmentMetrics(
        coverage=coverage,
        motion_distance_mae=motion_error,
        activity_fraction_absolute_error=activity_fraction_error,
        nearest_neighbor_relative_distance_mae=float(np.mean(neighbor_errors)) if neighbor_errors else None,
        frame_count=len(segment.frames),
        activity_transition_count=len(reference_active),
        reference_active_frame_count=reference_active_count,
        predicted_active_frame_count=prediction_active_count,
        reference_motion_distance=reference_motion,
        predicted_motion_distance=predicted_motion,
        nearest_neighbor_frame_count=len(neighbor_errors),
    )


def box_iou(left: Box, right: Box) -> float:
    """Return intersection-over-union for two valid axis-aligned boxes."""
    intersection_left = max(left.x, right.x)
    intersection_top = max(left.y, right.y)
    intersection_right = min(left.x + left.w, right.x + right.w)
    intersection_bottom = min(left.y + left.h, right.y + right.h)
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection = intersection_width * intersection_height
    union = left.w * left.h + right.w * right.h - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _interpolate_box(left: Box, right: Box, alpha: float) -> Box:
    """Interpolate a box using the reference builder's center/log-size rule."""
    left_center = box_center(left)
    right_center = box_center(right)
    center = (1.0 - alpha) * left_center + alpha * right_center
    width = exp((1.0 - alpha) * log(left.w) + alpha * log(right.w))
    height = exp((1.0 - alpha) * log(left.h) + alpha * log(right.h))
    return Box(
        x=float(center[0] - width / 2.0),
        y=float(center[1] - height / 2.0),
        w=width,
        h=height,
    )


def reconstruct_anchor_interpolated_prediction(
    reference: dict[int, dict[int, Box]],
    prediction: dict[int, dict[int, Box]],
    mapping: dict[int, int],
    manual_frame_ids: set[int],
    *,
    iou_threshold: float = 0.5,
) -> dict[int, dict[int, Box]]:
    """Build measurement boxes from prediction anchors without repairing gaps.

    Prediction boxes at official manual frames are retained. Intermediate
    boxes are reconstructed with the same interpolation rule as the reference
    only when the fixed mapped identity remains IoU-valid throughout the whole
    anchor interval. Missing or invalid observations therefore remain missing.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    reconstructed: dict[int, dict[int, Box]] = {}
    segments_by_identity: dict[int, list[Segment]] = {}
    for segment in valid_segments(reference):
        segments_by_identity.setdefault(segment.track_id, []).append(segment)

    for reference_id, prediction_id in mapping.items():
        for segment in segments_by_identity.get(reference_id, []):
            segment_frames = set(segment.frames)
            anchors = [frame_id for frame_id in segment.frames if frame_id in manual_frame_ids]
            for frame_id in anchors:
                predicted_box = prediction.get(frame_id, {}).get(prediction_id)
                if (
                    predicted_box is not None
                    and box_iou(reference[frame_id][reference_id], predicted_box) >= iou_threshold
                ):
                    reconstructed.setdefault(frame_id, {})[prediction_id] = predicted_box

            for left_frame, right_frame in zip(anchors, anchors[1:]):
                interval = tuple(range(left_frame, right_frame + 1))
                if any(frame_id not in segment_frames for frame_id in interval):
                    continue
                if any(
                    prediction_id not in prediction.get(frame_id, {})
                    or box_iou(
                        reference[frame_id][reference_id],
                        prediction[frame_id][prediction_id],
                    ) < iou_threshold
                    for frame_id in interval
                ):
                    continue
                left_box = prediction[left_frame][prediction_id]
                right_box = prediction[right_frame][prediction_id]
                gap = right_frame - left_frame
                for frame_id in range(left_frame + 1, right_frame):
                    alpha = (frame_id - left_frame) / gap
                    reconstructed.setdefault(frame_id, {})[prediction_id] = _interpolate_box(
                        left_box, right_box, alpha
                    )
    return reconstructed


def global_identity_mapping(
    reference: dict[int, dict[int, Box]],
    prediction: dict[int, dict[int, Box]],
    iou_threshold: float = 0.5,
) -> dict[int, int]:
    """Map each reference identity to one dominant predicted identity per sequence.

    The first Hungarian pass makes valid one-to-one matches within each frame;
    valid match counts are accumulated, with IoU used only as a tie-break. A
    second pass fixes a one-to-one identity correspondence for the whole
    sequence, so later tracker ID changes are not concealed through framewise
    rematching.
    """
    reference_ids = sorted({track_id for frame in reference.values() for track_id in frame})
    prediction_ids = sorted({track_id for frame in prediction.values() for track_id in frame})
    if not reference_ids or not prediction_ids:
        return {}

    reference_index = {track_id: index for index, track_id in enumerate(reference_ids)}
    prediction_index = {track_id: index for index, track_id in enumerate(prediction_ids)}
    match_count = np.zeros((len(reference_ids), len(prediction_ids)), dtype=np.float64)
    iou_sum = np.zeros_like(match_count)

    for frame_id in sorted(set(reference).intersection(prediction)):
        reference_frame = reference[frame_id]
        prediction_frame = prediction[frame_id]
        if not reference_frame or not prediction_frame:
            continue
        frame_reference_ids = sorted(reference_frame)
        frame_prediction_ids = sorted(prediction_frame)
        overlaps = np.asarray(
            [
                [box_iou(reference_frame[reference_id], prediction_frame[prediction_id])
                 for prediction_id in frame_prediction_ids]
                for reference_id in frame_reference_ids
            ],
            dtype=np.float64,
        )
        feasible = overlaps >= iou_threshold
        gated_costs = np.where(feasible, 1.0 - overlaps, 1e6)
        rows, columns = linear_sum_assignment(gated_costs)
        for row, column in zip(rows, columns, strict=True):
            if not feasible[row, column]:
                continue
            overlap = float(overlaps[row, column])
            reference_id = frame_reference_ids[row]
            prediction_id = frame_prediction_ids[column]
            row_index = reference_index[reference_id]
            column_index = prediction_index[prediction_id]
            match_count[row_index, column_index] += 1.0
            iou_sum[row_index, column_index] += overlap

    normalized_iou = iou_sum / max(float(iou_sum.max()), 1.0)
    score = match_count + 1e-6 * normalized_iou
    rows, columns = linear_sum_assignment(-score)
    mapping: dict[int, int] = {}
    for row, column in zip(rows, columns, strict=True):
        if match_count[row, column] > 0.0:
            mapping[reference_ids[row]] = prediction_ids[column]
    return mapping
