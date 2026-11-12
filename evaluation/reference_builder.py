"""Build measurement references from official COCO annotations and an explicit split manifest.

Interpolation and six-decimal MOT formatting retain the original reference
builder's operations. Images are indexed for sequence length, never copied or
decoded; the measurement evaluator needs only reference boxes and metadata.
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re


SPECS = {
    "celegans": ("C. elegans", "worm", "worm_id"),
    "drosophila": ("Drosophila", "drosophila", "fly_id"),
    "zebrafish": ("Zebrafish", "zebrafish", "fish_id"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def physical_index(file_name):
    return int(Path(file_name).stem.rsplit("_", 1)[-1])


def interpolate_bbox(start, end, alpha, image_width, image_height):
    x0, y0, w0, h0 = start
    x1, y1, w1, h1 = end
    cx0, cy0 = x0 + w0 / 2.0, y0 + h0 / 2.0
    cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
    cx = (1.0 - alpha) * cx0 + alpha * cx1
    cy = (1.0 - alpha) * cy0 + alpha * cy1
    width = math.exp((1.0 - alpha) * math.log(w0) + alpha * math.log(w1))
    height = math.exp((1.0 - alpha) * math.log(h0) + alpha * math.log(h1))
    x = cx - width / 2.0
    y = cy - height / 2.0
    x = min(max(0.0, x), max(0.0, image_width - 1e-6))
    y = min(max(0.0, y), max(0.0, image_height - 1e-6))
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Interpolation produced a non-positive bounding box")
    return x, y, width, height


def interpolate_tracks(manual_frames, manual, width, height):
    interpolated = defaultdict(dict)
    metadata = {}
    for left_frame, right_frame in zip(manual_frames, manual_frames[1:]):
        gap = right_frame - left_frame
        if gap <= 1:
            continue
        common_ids = sorted(set(manual[left_frame]) & set(manual[right_frame]))
        for frame in range(left_frame + 1, right_frame):
            alpha = (frame - left_frame) / gap
            for identity in common_ids:
                interpolated[frame][identity] = interpolate_bbox(
                    manual[left_frame][identity], manual[right_frame][identity], alpha, width, height)
                metadata[(frame, identity)] = (left_frame, right_frame, alpha)
    return interpolated, metadata


def mot_line(frame, identity, bbox):
    x, y, width, height = bbox
    return f"{frame},{identity},{x:.6f},{y:.6f},{width:.6f},{height:.6f},1,1,1"


def load_source(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = defaultdict(list)
    annotations = defaultdict(list)
    for image in payload["images"]:
        images[image["file_name"].split("/", 1)[0]].append(image)
    for annotation in payload["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    return images, annotations


def prepare_sequence(source_root, species, sequence, grouped, annotations, key):
    match = re.fullmatch(re.escape(species) + r"_(\d+)_(train|val)", sequence)
    if match is None:
        raise ValueError(f"invalid original sequence name: {sequence}")
    source_id = match.group(1)
    images = sorted(grouped[source_id], key=lambda image: physical_index(image["file_name"]))
    if not images:
        raise ValueError(f"{sequence}: no official annotation images")
    width, height = int(images[0]["width"]), int(images[0]["height"])
    image_dir = source_root / SPECS[species][0] / "data" / source_id
    physical = sorted((physical_index(path.name), path.suffix.lower()) for path in image_dir.iterdir()
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not physical or [index for index, _ in physical] != list(range(physical[-1][0] + 1)):
        raise ValueError(f"{sequence}: physical image sequence is missing or non-contiguous")
    manual_frames = sorted(physical_index(image["file_name"]) + 1 for image in images)
    if len(set(manual_frames)) != len(manual_frames) or max(manual_frames) > len(physical):
        raise ValueError(f"{sequence}: duplicate or out-of-range manual frame")
    manual = defaultdict(dict)
    annotation_ids = {}
    for image in images:
        if (int(image["width"]), int(image["height"])) != (width, height):
            raise ValueError(f"{sequence}: inconsistent image dimensions")
        frame = physical_index(image["file_name"]) + 1
        # Include explicitly empty manual images as interpolation boundaries.
        manual[frame]
        for annotation in annotations[int(image["id"])]:
            identity = annotation[key]
            if (isinstance(identity, bool) or not isinstance(identity, (int, float))
                    or not math.isfinite(identity) or identity <= 0 or int(identity) != identity):
                raise ValueError(f"{sequence}: identity must be a finite positive integer at frame {frame}")
            identity = int(identity)
            box = tuple(map(float, annotation["bbox"]))
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{sequence}: invalid source box at frame {frame}, identity {identity}")
            if identity in manual[frame]:
                raise ValueError(f"{sequence}: duplicate identity at frame {frame}")
            manual[frame][identity] = box
            annotation_ids[(frame, identity)] = int(annotation["id"])
    interpolated, metadata = interpolate_tracks(manual_frames, manual, width, height)
    rows = [(frame, identity, box, "manual") for frame, tracks in manual.items()
            for identity, box in tracks.items()]
    rows += [(frame, identity, box, "interpolated") for frame, tracks in interpolated.items()
             for identity, box in tracks.items()]
    rows.sort(key=lambda row: (row[0], row[1]))
    return dict(name=sequence, width=width, height=height, length=len(physical),
                extension=physical[0][1], manual_frames=manual_frames,
                annotation_ids=annotation_ids, interpolation_metadata=metadata, rows=rows)


def write_sequence(root, details):
    gt = root / "gt"
    gt.mkdir(parents=True)
    rows = details["rows"]
    for name, subset in (("gt.txt", rows), ("gt_manual.txt", [row for row in rows if row[3] == "manual"])):
        (gt / name).write_text("\n".join(mot_line(frame, identity, box)
                                        for frame, identity, box, _ in subset) + "\n", encoding="utf-8")
    (gt / "manual_frame_ids.txt").write_text("\n".join(map(str, details["manual_frames"])) + "\n", encoding="utf-8")
    with (gt / "interpolation_mask.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "source", "left_manual_frame", "right_manual_frame",
                         "alpha", "source_annotation_id"])
        for frame, identity, _, source in rows:
            if source == "manual":
                writer.writerow([frame, identity, source, frame, frame, "0.000000", details["annotation_ids"][(frame, identity)]])
            else:
                left, right, alpha = details["interpolation_metadata"][(frame, identity)]
                writer.writerow([frame, identity, source, left, right, f"{alpha:.6f}", ""])
    (root / "seqinfo.ini").write_text("\n".join([
        "[Sequence]", f"name = {details['name']}", "imDir = img1", "frameRate = 1",
        f"seqLength = {details['length']}", f"imWidth = {details['width']}",
        f"imHeight = {details['height']}", f"imExt = {details['extension']}", "",
    ]), encoding="utf-8")


def build_reference(source_root: Path, manifest_path: Path, output_root: Path):
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty reference directory: {output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared, source_cache, input_hashes = [], {}, {}
    for species, splits in manifest["species"].items():
        if species not in SPECS:
            raise ValueError(f"unsupported species: {species}")
        source_name, prefix, identity_key = SPECS[species]
        for physical_split in ("train", "train_val", "val"):
            names = splits.get(physical_split, [])
            if len(names) != len(set(names)):
                raise ValueError(f"{species}/{physical_split}: duplicate sequence name")
            for name in names:
                match = re.fullmatch(re.escape(species) + r"_(\d+)_(train|val)", name)
                if match is None:
                    raise ValueError(f"invalid sequence name: {name}")
                annotation_path = source_root / source_name / "annotations" / f"{prefix}_{match.group(2)}.json"
                if annotation_path not in source_cache:
                    source_cache[annotation_path] = load_source(annotation_path)
                    input_hashes[str(annotation_path)] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
                grouped, annotations = source_cache[annotation_path]
                details = prepare_sequence(source_root, species, name, grouped, annotations, identity_key)
                prepared.append((species, physical_split, details))
    if not prepared:
        raise ValueError("manifest selects no reference sequences")
    # All requested sequences have been validated before creating output files.
    output_root.mkdir(parents=True, exist_ok=True)
    for species, split, details in prepared:
        write_sequence(output_root / species / split / details["name"], details)
    (output_root / "split_manifest.json").write_bytes(manifest_path.read_bytes())
    (output_root / "reference_provenance.json").write_text(json.dumps(dict(
        sequences=len(prepared), input_annotation_sha256=input_hashes,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        image_pixels_copied=False, no_implicit_sequence_exclusions=True,
    ), indent=2) + "\n", encoding="utf-8")
    return len(prepared)
