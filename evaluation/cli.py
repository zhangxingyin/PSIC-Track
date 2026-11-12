"""Portable entry for the frozen paper metrics and manual-frame MOT inputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import scipy

# Resolve only this shipped package, never the original workspace or cwd.
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from evaluation.evaluator import Method, load_manual_frame_ids, run_evaluation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value, field):
    if not isinstance(value, str) or not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field} must be one non-empty path component")
    return value


def unused_output(path):
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    return path


def methods_from_file(path, results_root):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("methods file must be a non-empty JSON list")
    methods, roots = [], {}
    labels, directories = set(), set()
    for entry in payload:
        if not isinstance(entry, dict) or set(entry) - {"label", "directory", "family", "root"}:
            raise ValueError("method entries allow only label/directory/family/root")
        label = entry["label"]
        directory = safe_name(entry["directory"], "method directory")
        if not isinstance(label, str) or not label or label in labels or directory in directories:
            raise ValueError("method labels and directory names must be non-empty and unique")
        family = entry["family"]
        if not isinstance(family, str) or not family:
            raise ValueError("method family must be non-empty")
        root = Path(entry.get("root", "."))
        roots[directory] = (results_root / root).resolve()
        labels.add(label)
        directories.add(directory)
        methods.append(Method(label, directory, family))
    return methods, roots


def measure(args):
    output = unused_output(args.output_dir)
    dataset = args.dataset_root.resolve()
    methods, roots = methods_from_file(args.methods_file, args.results_root.resolve())
    manifest_path = dataset / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {manifest_path, args.methods_file.resolve()}
    predictions = []
    for species, split in manifest["species"].items():
        safe_name(species, "species")
        for split_name in ("train", "val"):
            names = split[split_name]
            if len(names) != len(set(names)):
                raise ValueError(f"{species}/{split_name}: duplicate sequence name")
            for sequence in names:
                safe_name(sequence, "sequence")
                sequence_root = dataset / species / split_name / sequence
                inputs.add(sequence_root / "gt" / "gt.txt")
                if split_name == "val":
                    inputs.add(sequence_root / "gt" / "manual_frame_ids.txt")
                    inputs.add(sequence_root / "seqinfo.ini")
                    for method in methods:
                        source = roots[method.directory] / species / method.directory / f"{sequence}.txt"
                        inputs.add(source)
                        predictions.append((species, method.directory, sequence, source))
    before = {str(path): sha256(path) for path in sorted(inputs)}
    # A small temporary view adapts explicitly supplied result roots to the
    # original evaluator's directory contract; it never changes metric logic.
    with tempfile.TemporaryDirectory(prefix="psictrack-evaluation-") as temporary:
        view = Path(temporary)
        for species, directory, sequence, source in predictions:
            target = view / species / directory / f"{sequence}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        payload = run_evaluation(dataset, view, methods=methods)
    checks = None
    if args.expected_json is not None:
        expected = json.loads(args.expected_json.read_text(encoding="utf-8"))
        checks = []
        for species, actual in payload["species"].items():
            saved = expected["species"][species]
            for field in ("activity_threshold", "test_sequences"):
                if actual[field] != saved[field]:
                    raise ValueError(f"saved-result mismatch: {species}/{field}")
            for method in methods:
                if actual["methods"][method.label] != saved["methods"][method.label]:
                    raise ValueError(f"saved-result mismatch: {species}/{method.label}")
                checks.append(dict(species=species, method=method.label, exact_match=True))
    after = {str(path): sha256(path) for path in sorted(inputs)}
    if before != after:
        raise RuntimeError("evaluation input changed while the evaluator was running")
    provenance = dict(
        python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
        core_sha256={p.name: sha256(p) for p in (PACKAGE / "evaluation").glob("*.py")},
        input_files=before, inputs_unchanged=True, expected_result_checks=checks,
        output_source="calculated from supplied reference and prediction files",
        expected_json_sha256=sha256(args.expected_json) if args.expected_json else None,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "method", "metric", "mean", "std", "n"])
        for species, details in payload["species"].items():
            for method, values in details["methods"].items():
                for metric, summary in values.items():
                    if isinstance(summary, dict) and "mean" in summary:
                        writer.writerow([species, method, metric, summary["mean"], summary["std"], summary["n"]])
    print(f"calculated metrics: {output / 'metrics.json'}")
    if checks is not None:
        print(f"exact saved-result checks passed: {len(checks)} species/method combinations")


def filtered_mot(path, manual, length):
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) < 7:
            raise ValueError(f"{path}:{number}: fewer than seven MOT fields")
        values = list(map(float, fields[:7]))
        frame, identity, _, _, width, height, _ = values
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: non-finite MOT field")
        if not frame.is_integer() or not 1 <= frame <= length or not identity.is_integer() or identity < 1:
            raise ValueError(f"{path}:{number}: invalid frame or identity")
        if width <= 0 or height <= 0 or (int(frame), int(identity)) in seen:
            raise ValueError(f"{path}:{number}: invalid box or duplicate identity")
        seen.add((int(frame), int(identity)))
        if int(frame) in manual:
            rows.append(line)
    return "\n".join(rows) + ("\n" if rows else "")


def manual_mot(args):
    output = unused_output(args.output_dir)
    manual = load_manual_frame_ids(args.manual_frame_ids)
    if args.sequence_length < 1 or max(manual) > args.sequence_length:
        raise ValueError("manual frame IDs must be within 1..sequence_length")
    gt = filtered_mot(args.gt, manual, args.sequence_length)
    prediction = filtered_mot(args.prediction, manual, args.sequence_length)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gt.txt").write_text(gt, encoding="utf-8")
    (output / "prediction.txt").write_text(prediction, encoding="utf-8")
    (output / "manual_frame_ids.txt").write_text("".join(f"{frame}\n" for frame in sorted(manual)), encoding="utf-8")
    print(f"manual-frame rows prepared without frame renumbering: {output}")


def build_reference(args):
    from evaluation.reference_builder import build_reference as build

    count = build(args.source_root.resolve(), args.manifest.resolve(), unused_output(args.output_dir))
    print(f"built {count} reference sequences from original annotations: {args.output_dir}")


def mot(args):
    from evaluation.mot_runner import run_mot

    run_mot(args, safe_name=safe_name, unused_output=unused_output,
            methods_from_file=methods_from_file, filtered_mot=filtered_mot,
            load_manual_frame_ids=load_manual_frame_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measurement = commands.add_parser("measure", help="calculate the four paper trajectory metrics")
    measurement.add_argument("--dataset-root", type=Path, required=True)
    measurement.add_argument("--results-root", type=Path, required=True)
    measurement.add_argument("--methods-file", type=Path, required=True)
    measurement.add_argument("--expected-json", type=Path)
    measurement.add_argument("--output-dir", type=Path, required=True)
    measurement.set_defaults(run=measure)
    manual = commands.add_parser("manual-mot", help="filter both MOT inputs to original manual frames")
    manual.add_argument("--manual-frame-ids", type=Path, required=True)
    manual.add_argument("--gt", type=Path, required=True)
    manual.add_argument("--prediction", type=Path, required=True)
    manual.add_argument("--sequence-length", type=int, required=True)
    manual.add_argument("--output-dir", type=Path, required=True)
    manual.set_defaults(run=manual_mot)
    reference = commands.add_parser("build-reference", help="rebuild references from official annotations and an explicit manifest")
    reference.add_argument("--source-root", type=Path, required=True)
    reference.add_argument("--manifest", type=Path, required=True)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.set_defaults(run=build_reference)
    tracking = commands.add_parser('mot', help='calculate manual-frame MOT metrics with external TrackEval')
    tracking.add_argument('--dataset-root', type=Path, required=True)
    tracking.add_argument('--results-root', type=Path, required=True)
    tracking.add_argument('--manifest', type=Path, required=True)
    tracking.add_argument('--methods-file', type=Path, required=True)
    tracking.add_argument('--species', required=True)
    tracking.add_argument('--trackeval-root', type=Path, required=True)
    tracking.add_argument('--output-dir', type=Path, required=True)
    tracking.set_defaults(run=mot)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
