# PSIC-Track

PSIC-Track is a strictly online multi-animal tracker for maintaining identity-consistent trajectories in near-closed monitoring domains. It combines temporal geometric association (TGA), pose-structured observation integrity (PSOI), and near-closed identity conservation (NCIC).

## Applicability

PSIC-Track is intended for continuous animal monitoring with a fixed view, a finite population and a bounded activity domain. It is particularly useful when visually similar individuals experience occlusion, overlap, temporary visibility loss, duplicate detections or merged detections. PSOI assesses whether a candidate provides a structurally credible observation of one body, while NCIC uses bounded-domain visibility, identity persistence and soft population capacity to resolve competing explanations online.

The SpaceAnimal space-experiment videos of *C. elegans*, *Drosophila* and zebrafish provide one such near-closed monitoring setting.

## Repository layout

```text
PSIC-Track/
├── run_tracker.py                    # command-line tracking entry point
├── psictrack/                        # strictly online tracking implementation
│   ├── __init__.py                   # package interface and shared exports
│   ├── types.py                      # observations, track outputs and lifecycle states
│   ├── config.py                     # YAML configuration parsing and validation
│   ├── association.py                # association evidence and joint pair-cost utilities
│   ├── base_motion.py                # TGA: box-motion prediction, gating and assignment
│   ├── psoi.py                       # PSOI: pose encoding, structural integrity and memory
│   ├── ncic.py                       # NCIC: visibility, identity existence and soft capacity
│   ├── tracker.py                    # causal tracking loop and identity-state updates
│   ├── io.py                         # detection/pose input parsing and observation binding
│   └── cli.py                        # tracking arguments, input checks and result writing
├── configs/                          # tracking settings, body topologies and method labels
│   ├── tga.yaml                      # TGA-only configuration
│   ├── paper_methods.json            # evaluation labels and prediction-directory mappings
│   ├── ablations/                    # configurations for PSOI/NCIC combinations
│   │   ├── tga_plus_psoi.yaml         # TGA + PSOI
│   │   ├── tga_plus_ncic.yaml         # TGA + NCIC
│   │   └── tga_plus_psoi_ncic.yaml    # full PSIC-Track
│   └── species/                      # keypoint order and anatomical topology definitions
│       ├── README.md                 # topology fields and adaptation guide
│       ├── celegans.yaml             # C. elegans body topology
│       ├── drosophila.yaml           # Drosophila body topology
│       └── zebrafish.yaml            # zebrafish body topology
├── evaluation/                       # reference construction and quantitative evaluation
│   ├── __init__.py                   # evaluation package declaration
│   ├── cli.py                        # build-reference, measure and manual-frame MOT commands
│   ├── reference_builder.py          # COCO annotations to manual/interpolated MOT references
│   ├── trajectory_metrics.py         # identity mapping and segment-level trajectory metrics
│   ├── evaluator.py                  # activity thresholds, sequence evaluation and summaries
│   └── mot_runner.py                 # manual-frame tracking metrics through external TrackEval
├── requirements.txt                  # dependencies for prepared-input tracking and evaluation
├── .gitignore                        # exclusions for generated outputs, weights and local files
└── README.md                         # method overview, input formats and execution commands
```

Start with `psictrack/tracker.py` to follow the tracking logic, then read
`base_motion.py`, `psoi.py` and `ncic.py` for the three framework modules.
For evaluation, start with `evaluation/cli.py` and follow its calls to reference
construction, trajectory measurements or TrackEval. Run commands are given below.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required.

## Quick start: full PSIC-Track

```bash
python run_tracker.py \
  --detections /path/to/detections.txt \
  --poses /path/to/poses.json \
  --topology configs/species/drosophila.yaml \
  --config configs/ablations/tga_plus_psoi_ncic.yaml \
  --frame-size 1920 1080 \
  --sequence-length 1200 \
  --output outputs/example
```

The full configuration requires predicted detection boxes, predicted poses, a body-topology definition, and image dimensions. Frames are processed causally; no future observations are read.

This repository takes detection and pose predictions as inputs; detector and pose-model inference are outside the scope of this package.

## Input files

### Detection file

Pass `--detections` a comma-separated MOT-style file with at least these columns:

```text
frame,input_id,x,y,width,height,score
```

Frame indices are one-based. `input_id` is accepted for compatibility with detector exports but is not used as a tracking identity. The row order within each frame defines the zero-based `detection_id` used to attach poses.

### Pose file

Pass `--poses` a JSON file containing an `annotations` list. Each annotation provides:

```json
{
  "frame_id": 1,
  "detection_id": 0,
  "bbox": [x, y, width, height],
  "keypoints": [x1, y1, v1, x2, y2, v2],
  "keypoint_confidences": [c1, c2]
}
```

`detection_id` binds a pose to a detection in the same frame and should normally be supplied. It is the zero-based row position in that frame's detection file, not the detector's `input_id`. If omitted, a pose is bound only when its `bbox` has IoU at least `0.99` with an otherwise unbound detection. `keypoint_confidences` is optional; when absent, the third value of each keypoint triplet is treated as a visibility flag. Pose input is mandatory whenever PSOI is enabled.

### Topology file

`--topology` is the YAML body-topology file whose keypoint order matches the pose-estimator output. The repository provides:

- `configs/species/celegans.yaml`
- `configs/species/drosophila.yaml`
- `configs/species/zebrafish.yaml`

Use one of these paths directly when its topology matches your pose output, or create a YAML for another species with `keypoints`, `skeleton`, `body_axis`, and `pose_consistency_scale`. See [`configs/species/README.md`](configs/species/README.md) for the field definitions.

### Optional sequence arguments

- `--sequence-name` records an optional sequence identifier in `run_manifest.json`.
- `--sequence-length` processes trailing empty frames through the specified index; otherwise, the final frame is inferred from the input files.
- `--frame-size WIDTH HEIGHT` is required when NCIC is enabled because the arena-visibility state uses image geometry.

All detection boxes, pose boxes, and keypoints must use the same pixel coordinate system.

## Configurations

| Configuration | Enabled modules | Required inputs |
| --- | --- | --- |
| `configs/tga.yaml` | TGA | detections |
| `configs/ablations/tga_plus_psoi.yaml` | TGA + PSOI | detections, poses, topology |
| `configs/ablations/tga_plus_ncic.yaml` | TGA + NCIC | detections, frame size |
| `configs/ablations/tga_plus_psoi_ncic.yaml` | TGA + PSOI + NCIC | detections, poses, topology, frame size |

## Outputs

`--output` creates:

- `tracks.txt`: online MOT-format trajectory output;
- `diagnostics.jsonl`: frame-level association diagnostics;
- `run_manifest.json`: run configuration and output metadata.

## Evaluation

All commands are run from the repository root with Python 3.10 or newer. Numerical evaluation uses NumPy and SciPy; tracking metrics additionally require an external [TrackEval](https://github.com/JonathonLuiten/TrackEval) checkout. The paper used TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`.

### Inputs

```text
SpaceAnimal_MOT_interpolate/
  split_manifest.json
  SPECIES/train/SEQUENCE/gt/gt.txt
  SPECIES/val/SEQUENCE/gt/gt.txt
  SPECIES/val/SEQUENCE/gt/manual_frame_ids.txt
  SPECIES/val/SEQUENCE/seqinfo.ini
track_results/
  SPECIES/METHOD/SEQUENCE.txt
```

The manifest explicitly lists the training and evaluation sequences for each species. The directory key `val` denotes the evaluation partition in these files, not an automatically inferred internal-validation split. `configs/paper_methods.json` maps method labels to result directories; its optional `root` selects a result subdirectory. Supply the actual manifest and prediction files rather than inferring a split from directory contents.

### Build measurement references

```bash
python evaluation/cli.py build-reference \
  --source-root /data/SpaceAnimal \
  --manifest /data/split_manifest.json \
  --output-dir outputs/reference
```

The source directory contains `C. elegans`, `Drosophila` and `Zebrafish`, each with `data` and `annotations`. The command reads original COCO coordinates at their supplied precision, retains original frame indices, and exports manual and interpolated MOT boxes to six decimal places, together with manual-frame masks and source records. Reference construction does not copy images or silently discard invalid requested sequences.

### Trajectory measurements

```bash
python evaluation/cli.py measure \
  --dataset-root /data/SpaceAnimal_MOT_interpolate \
  --results-root /data/track_results \
  --methods-file configs/paper_methods.json \
  --output-dir outputs/measurements
```

The four outputs are identity-consistent trajectory coverage (ITC), cumulative travel distance error (CTD-MAE), active-frame ratio error (AFR-MAE) and nearest-neighbor relative distance error (NNRD-MAE). Outputs contain JSON/CSV statistics and input/source checksums. An optional `--expected-json` checks newly computed statistics against an independently supplied result; it never replaces computation with stored scores. Existing nonempty output directories are not overwritten.

Reference centers are linearly interpolated and box widths/heights log-linearly interpolated only between adjacent manual anchors of the same identity, using original frame intervals. Reference identity gaps and sequence endpoints are not bridged. ITC uses raw framewise output; the other measurements use valid prediction anchors, reconstructing intermediate measurement boxes only when the fixed predicted identity remains present and IoU-valid throughout the interval. This is offline evaluation and is not fed to the tracker.

Displacements are evaluated only across consecutive original frames within a reference segment. Missing predicted endpoints contribute no travel or active transition, while the reference transition count remains the activity-ratio denominator. CTD is in pixels, AFR is a frame-transition fraction, and no placeholder frame rate is interpreted as physical seconds. NNRD uses the same reference-neighbor identity pair and assigns the paper's image-domain error penalty to missing or invalid pairs. Emitted propagation positions receive the same identity/IoU checks; an internal identity without an emitted position is missing.

### Manual-frame tracking metrics

```bash
python evaluation/cli.py mot \
  --dataset-root /data/SpaceAnimal_MOT_full \
  --manifest /data/split_manifest.json \
  --results-root /data/track_results \
  --methods-file configs/paper_methods.json \
  --species celegans \
  --trackeval-root /opt/TrackEval \
  --output-dir outputs/mot_celegans
```
This command evaluates only original manually annotated frames, retains frame numbers and explicit empty annotated frames, and never supplies interpolated references to TrackEval. If no explicit manual-frame mask is supplied, the GT must already be sparse manual GT; arbitrary coordinates alone cannot establish annotation provenance. HOTA uses the official threshold average, CLEAR/Identity use IoU 0.50, and reported IDs is `IDSW`. Species-level results come from TrackEval's combined-sequence output, not a mean of sequence scores.

Predictions, trained weights, datasets and generated results are not bundled.
