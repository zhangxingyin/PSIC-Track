"""Command-line interface for prediction-driven PSIC-Track."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import yaml

from .config import TrackerConfig
from .io import load_prediction_observations
from .psoi import SpeciesTopology
from .tracker import PSICTracker


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strictly online PSIC-Track from prepared detector and pose predictions."
    )
    parser.add_argument("--detections", required=True, type=Path, help="MOT-format detection file")
    parser.add_argument("--poses", type=Path, help="pose JSON file; required when PSOI is enabled")
    parser.add_argument("--topology", type=Path, help="YAML body-topology file; required when PSOI is enabled")
    parser.add_argument("--config", required=True, type=Path, help="tracker configuration YAML")
    parser.add_argument("--sequence-name", help="optional identifier recorded in the output manifest")
    parser.add_argument("--sequence-length", type=_positive_int, help="include trailing empty frames through this index")
    parser.add_argument(
        "--frame-size",
        type=_positive_int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="image width and height; required when NCIC is enabled",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_config(path: Path) -> TrackerConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("tracker config must be a YAML mapping")
    return TrackerConfig.from_mapping(payload)


def _validate_inputs(args: argparse.Namespace, config: TrackerConfig) -> None:
    if not args.detections.is_file():
        raise FileNotFoundError(args.detections)
    if args.poses is not None and not args.poses.is_file():
        raise FileNotFoundError(args.poses)
    if args.topology is not None and not args.topology.is_file():
        raise FileNotFoundError(args.topology)
    if config.closed_world.enabled and args.frame_size is None:
        raise ValueError("--frame-size WIDTH HEIGHT is required when NCIC is enabled")
    if config.pose.enabled or config.observation_integrity.enabled:
        if args.poses is None:
            raise ValueError("--poses is required when a pose-structure module is enabled")
        if args.topology is None:
            raise ValueError("--topology is required when a pose-structure module is enabled")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":")) + "\n")


def _mot_line(track) -> str:
    x, y, width, height = track.bbox_xywh
    return (
        f"{track.frame},{track.track_id},{x:.6f},{y:.6f},{width:.6f},{height:.6f},"
        f"{track.score:.6f},-1,-1,-1"
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(args.config)
    _validate_inputs(args, config)
    topology = None if args.topology is None else SpeciesTopology.from_yaml(args.topology)
    frame_size = None if args.frame_size is None else tuple(args.frame_size)
    tracker = PSICTracker(config, topology=topology, frame_size=frame_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    mot_lines: list[str] = []
    diagnostics: list[dict[str, object]] = []
    frame_count = 0
    try:
        frames = load_prediction_observations(
            args.detections, args.poses, sequence_length=args.sequence_length
        )
        for frame, observations in frames:
            frame_count += 1
            outputs = tracker.update(frame, observations)
            mot_lines.extend(_mot_line(track) for track in outputs if track.emit_mot)
            diagnostics.append(
                {
                    "frame": frame,
                    "module_switches": {
                        "tga": True,
                        "psoi": config.pose.enabled,
                        "ncic": config.closed_world.enabled,
                    },
                    "observation_sources": sorted({item.source.value for item in observations}),
                    "pose_sources": sorted(
                        {item.pose_source.value for item in observations if item.pose_source is not None}
                    ),
                    "associations": tracker.last_diagnostics,
                }
            )

        (temporary / "tracks.txt").write_text(
            "\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8"
        )
        _write_jsonl(temporary / "diagnostics.jsonl", diagnostics)
        manifest = {
            "schema_version": 2,
            "sequence": args.sequence_name or args.detections.stem,
            "topology": None if topology is None else topology.species,
            "frame_size": frame_size,
            "sequence_length": args.sequence_length,
            "time_unit": "frame",
            "speed_units": ["pixel/frame", "body-length/frame"],
            "mot_output_policy": "detector observations plus qualified short-gap propagation",
            "frames_processed": frame_count,
            "config": asdict(config),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0


def main() -> None:
    raise SystemExit(run_cli())
