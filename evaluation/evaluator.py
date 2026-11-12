"""Shared configuration and summary helpers for trajectory-measurement evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from configparser import ConfigParser
from pathlib import Path
from typing import Iterable

import numpy as np

from .trajectory_metrics import (
    Box,
    SegmentMetrics,
    box_center,
    evaluate_segment,
    global_identity_mapping,
    reconstruct_anchor_interpolated_prediction,
    valid_segments,
)


@dataclass(frozen=True)
class Method:
    """A manuscript comparison method and its result-directory key."""

    label: str
    directory: str
    family: str


PAPER_METHODS: tuple[Method, ...] = (
    Method("SORT", "sort", "TbD"),
    Method("AdapTrack", "adaptrack", "TbD"),
    Method("TOPICTrack", "topictrack", "TbD"),
    Method("DeepSORT", "deepsort", "TbD"),
    Method("StrongSORT", "strongsort", "TbD"),
    Method("FastTracker", "fasttracker", "TbD"),
    Method("C-BIoU", "cbiou", "TbD"),
    Method("ConfTrack", "conftrack", "TbD"),
    Method("UCMCTrack", "ucmctrack", "TbD"),
    Method("ByteTrack", "bytetrack", "TbD"),
    Method("BoT-SORT", "botsort", "TbD"),
    Method("OC-SORT", "ocsort", "TbD"),
    Method("SparseTrack", "sparsetrack", "TbD"),
    Method("BoostTrack", "boosttrack", "TbD"),
    Method("PoseFlow", "poseflow", "Pose/keypoint"),
    Method("PoseBYTE", "posebyte", "Pose/keypoint"),
    Method("SLEAP-Tracker", "sleap", "Pose/keypoint"),
    Method("LightTrack-SGCN", "lighttrack_sgcn", "Pose/keypoint"),
    Method("PSIC-Track", "psictrack_ablation_tga_plus_psoi_ncic", "Ours"),
)


def read_scene_diagonal(sequence_root: str | Path) -> float | None:
    """Read the fixed image diagonal from MOT sequence metadata when available."""
    metadata_path = Path(sequence_root) / "seqinfo.ini"
    if not metadata_path.is_file():
        return None
    parser = ConfigParser()
    parser.read(metadata_path, encoding="utf-8")
    try:
        width = parser.getfloat("Sequence", "imWidth")
        height = parser.getfloat("Sequence", "imHeight")
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid image dimensions in {metadata_path}") from error
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"image dimensions must be positive in {metadata_path}")
    return float(np.hypot(width, height))


def load_manual_frame_ids(path: str | Path) -> set[int]:
    """Read positive official annotation-frame indices for measurement anchors."""
    frame_ids: set[int] = set()
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            frame_id = int(line)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} is not an integer frame ID") from error
        if frame_id <= 0:
            raise ValueError(f"{path}:{line_number} must be positive")
        if frame_id in frame_ids:
            raise ValueError(f"{path}:{line_number} duplicates frame {frame_id}")
        frame_ids.add(frame_id)
    if not frame_ids:
        raise ValueError(f"{path} contains no manual frame IDs")
    return frame_ids


def derive_activity_threshold(
    references: Iterable[dict[int, dict[int, Box]]],
    *,
    quantile: float = 0.75,
) -> float:
    """Freeze an activity threshold from reference-only normalized displacements."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    normalized_speeds: list[float] = []
    for reference in references:
        for segment in valid_segments(reference):
            for previous_frame, current_frame in zip(segment.frames, segment.frames[1:]):
                previous_box = reference[previous_frame][segment.track_id]
                current_box = reference[current_frame][segment.track_id]
                displacement = float(np.linalg.norm(box_center(current_box) - box_center(previous_box)))
                scale = float(np.sqrt(current_box.w * current_box.h))
                normalized_speeds.append(displacement / scale)
    if not normalized_speeds:
        raise ValueError("at least one reference transition is required to derive activity threshold")
    return float(np.quantile(np.asarray(normalized_speeds, dtype=np.float64), quantile))


@dataclass(frozen=True)
class SequenceMetrics:
    """A sequence-level summary assembled from individual continuous segments."""

    coverage: float
    motion_distance_mae: float
    activity_fraction_absolute_error: float | None
    nearest_neighbor_relative_distance_mae: float | None
    segment_count: int
    reference_frame_count: int
    activity_segment_count: int
    activity_transition_count: int
    nearest_neighbor_segment_count: int
    nearest_neighbor_frame_count: int
    trajectory_metrics: tuple[SegmentMetrics, ...]


def _mean_defined(values: Iterable[float | None]) -> tuple[float | None, int]:
    observed = [float(value) for value in values if value is not None]
    return (float(np.mean(observed)) if observed else None, len(observed))


def evaluate_sequence(
    reference: dict[int, dict[int, Box]],
    prediction: dict[int, dict[int, Box]],
    *,
    activity_threshold: float,
    iou_threshold: float = 0.5,
    scene_diagonal: float | None = None,
    manual_frame_ids: set[int] | None = None,
) -> SequenceMetrics:
    """Summarize reference segments under one fixed sequence-global identity mapping."""
    mapping = global_identity_mapping(reference, prediction, iou_threshold=iou_threshold)
    measurement_prediction = (
        reconstruct_anchor_interpolated_prediction(
            reference,
            prediction,
            mapping,
            manual_frame_ids,
            iou_threshold=iou_threshold,
        )
        if manual_frame_ids is not None
        else prediction
    )
    segment_metrics: list[SegmentMetrics] = []
    for segment in valid_segments(reference):
        measurement_metrics = evaluate_segment(
            reference,
            measurement_prediction,
            mapping,
            segment,
            activity_threshold=activity_threshold,
            iou_threshold=iou_threshold,
            scene_diagonal=scene_diagonal,
        )
        if manual_frame_ids is not None:
            raw_metrics = evaluate_segment(
                reference,
                prediction,
                mapping,
                segment,
                activity_threshold=activity_threshold,
                iou_threshold=iou_threshold,
                scene_diagonal=scene_diagonal,
            )
            measurement_metrics = replace(measurement_metrics, coverage=raw_metrics.coverage)
        segment_metrics.append(measurement_metrics)
    if not segment_metrics:
        raise ValueError("reference contains no valid continuous segments")

    reference_frame_count = sum(metrics.frame_count for metrics in segment_metrics)
    coverage = sum(metrics.coverage * metrics.frame_count for metrics in segment_metrics) / reference_frame_count
    reference_motion = sum(metrics.reference_motion_distance for metrics in segment_metrics)
    predicted_motion = sum(metrics.predicted_motion_distance for metrics in segment_metrics)
    motion_error = abs(predicted_motion - reference_motion)
    activity_transition_count = sum(metrics.activity_transition_count for metrics in segment_metrics)
    activity_segment_count = sum(metrics.activity_transition_count > 0 for metrics in segment_metrics)
    if activity_transition_count:
        activity_fraction_error = abs(
            sum(metrics.predicted_active_frame_count for metrics in segment_metrics) / activity_transition_count
            - sum(metrics.reference_active_frame_count for metrics in segment_metrics) / activity_transition_count
        )
    else:
        activity_fraction_error = None
    nearest_neighbor_frame_count = sum(
        metrics.nearest_neighbor_frame_count for metrics in segment_metrics
    )
    nearest_neighbor_segment_count = sum(
        metrics.nearest_neighbor_frame_count > 0 for metrics in segment_metrics
    )
    nearest_neighbor_error = (
        sum(
            metrics.nearest_neighbor_relative_distance_mae * metrics.nearest_neighbor_frame_count
            for metrics in segment_metrics
            if metrics.nearest_neighbor_relative_distance_mae is not None
        ) / nearest_neighbor_frame_count
        if nearest_neighbor_frame_count
        else None
    )
    return SequenceMetrics(
        coverage=coverage,
        motion_distance_mae=motion_error,
        activity_fraction_absolute_error=activity_fraction_error,
        nearest_neighbor_relative_distance_mae=nearest_neighbor_error,
        segment_count=len(segment_metrics),
        reference_frame_count=reference_frame_count,
        activity_segment_count=activity_segment_count,
        activity_transition_count=activity_transition_count,
        nearest_neighbor_segment_count=nearest_neighbor_segment_count,
        nearest_neighbor_frame_count=nearest_neighbor_frame_count,
        trajectory_metrics=tuple(segment_metrics),
    )



METRIC_FIELDS: tuple[str, ...] = (
    "coverage",
    "motion_distance_mae",
    "activity_fraction_absolute_error",
    "nearest_neighbor_relative_distance_mae",
)


def _summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    observed = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not observed.size:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(observed)),
        "std": float(np.std(observed, ddof=0)),
        "n": int(observed.size),
    }


def _weighted_summary(
    values_and_weights: Iterable[tuple[float, int]],
) -> dict[str, float | int | None]:
    observed = tuple((float(value), int(weight)) for value, weight in values_and_weights if weight > 0)
    if not observed:
        return {"mean": None, "std": None, "n": 0}
    total_weight = sum(weight for _, weight in observed)
    mean = sum(value * weight for value, weight in observed) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in observed) / total_weight
    return {"mean": mean, "std": float(np.sqrt(variance)), "n": len(observed)}


def _summarize_method_sequences(sequences: Iterable[SequenceMetrics]) -> dict[str, object]:
    sequence_metrics = tuple(sequences)
    if not sequence_metrics:
        raise ValueError("at least one test sequence is required")
    trajectory_metrics = tuple(
        metrics
        for sequence in sequence_metrics
        for metrics in sequence.trajectory_metrics
    )
    summary: dict[str, object] = {
        "coverage": _weighted_summary(
            (metrics.coverage, metrics.frame_count) for metrics in trajectory_metrics
        )
    }
    summary.update(
        {
            metric_name: _summary(
                getattr(metrics, metric_name) for metrics in trajectory_metrics
            )
            for metric_name in METRIC_FIELDS
            if metric_name != "coverage"
        }
    )
    summary["sequence_count"] = len(sequence_metrics)
    summary["segment_count"] = sum(metrics.segment_count for metrics in sequence_metrics)
    summary["reference_frame_count"] = sum(metrics.reference_frame_count for metrics in sequence_metrics)
    summary["activity_segment_count"] = sum(metrics.activity_segment_count for metrics in sequence_metrics)
    summary["activity_transition_count"] = sum(
        metrics.activity_transition_count for metrics in sequence_metrics
    )
    summary["nearest_neighbor_segment_count"] = sum(
        metrics.nearest_neighbor_segment_count for metrics in sequence_metrics
    )
    summary["nearest_neighbor_frame_count"] = sum(
        metrics.nearest_neighbor_frame_count for metrics in sequence_metrics
    )
    return summary


def run_evaluation(
    dataset_root: str | Path,
    track_results_root: str | Path,
    *,
    methods: Iterable[Method] = PAPER_METHODS,
    activity_quantile: float = 0.75,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate only manifest val sequences for each supplied comparison method."""
    import json
    from pathlib import Path

    from .trajectory_metrics import load_mot_boxes

    dataset_root = Path(dataset_root)
    track_results_root = Path(track_results_root)
    manifest = json.loads((dataset_root / "split_manifest.json").read_text(encoding="utf-8"))
    selected_methods = tuple(methods)
    if not selected_methods:
        raise ValueError("at least one comparison method is required")

    species_payload: dict[str, object] = {}
    for species, split in manifest["species"].items():
        train_sequences = tuple(split["train"])
        test_sequences = tuple(split["val"])
        if not train_sequences or not test_sequences:
            raise ValueError(f"{species} requires non-empty train and val sequences")
        train_references = [
            load_mot_boxes(dataset_root / species / "train" / sequence / "gt" / "gt.txt")
            for sequence in train_sequences
        ]
        activity_threshold = derive_activity_threshold(
            train_references, quantile=activity_quantile
        )
        test_sequence_roots = {
            sequence: dataset_root / species / "val" / sequence for sequence in test_sequences
        }
        test_references = {
            sequence: load_mot_boxes(test_sequence_roots[sequence] / "gt" / "gt.txt")
            for sequence in test_sequences
        }
        scene_diagonals = {
            sequence: read_scene_diagonal(test_sequence_roots[sequence])
            for sequence in test_sequences
        }
        manual_frame_ids = {
            sequence: load_manual_frame_ids(
                test_sequence_roots[sequence] / "gt" / "manual_frame_ids.txt"
            )
            for sequence in test_sequences
        }

        method_payload: dict[str, object] = {}
        for method in selected_methods:
            sequence_metrics: list[SequenceMetrics] = []
            for sequence in test_sequences:
                prediction_path = track_results_root / species / method.directory / f"{sequence}.txt"
                if not prediction_path.is_file():
                    raise FileNotFoundError(
                        f"missing result for {method.label}, {species}, {sequence}: {prediction_path}"
                    )
                prediction = load_mot_boxes(prediction_path)
                sequence_metrics.append(
                    evaluate_sequence(
                        test_references[sequence],
                        prediction,
                        activity_threshold=activity_threshold,
                        iou_threshold=iou_threshold,
                        scene_diagonal=scene_diagonals[sequence],
                        manual_frame_ids=manual_frame_ids[sequence],
                    )
                )
            method_summary = _summarize_method_sequences(sequence_metrics)
            method_summary["family"] = method.family
            method_payload[method.label] = method_summary

        species_payload[species] = {
            "activity_threshold": activity_threshold,
            "test_sequences": list(test_sequences),
            "methods": method_payload,
        }

    return {
        "protocol": "interpolated_reference_trajectory_input_v2",
        "reference": "人工锚点边界框插值参考GT",
        "split": "val",
        "identity_iou_threshold": iou_threshold,
        "activity_threshold_quantile": activity_quantile,
        "prediction_reconstruction": "manual-anchor conditional interpolation",
        "measurement_aggregation": "identity-trajectory macro average",
        "species": species_payload,
    }
