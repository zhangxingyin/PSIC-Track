# PSIC-Track

PSIC-Track is a strictly online multi-animal tracker for maintaining identity-consistent trajectories in near-closed monitoring domains. It combines temporal geometric association (TGA), pose-structured observation integrity (PSOI), and near-closed identity conservation (NCIC).

## Applicability

PSIC-Track is intended for continuous animal monitoring with a fixed view, a finite population and a bounded activity domain. It is particularly useful when visually similar individuals experience occlusion, overlap, temporary visibility loss, duplicate detections or merged detections. PSOI assesses whether a candidate provides a structurally credible observation of one body, while NCIC uses bounded-domain visibility, identity persistence and soft population capacity to resolve competing explanations online.

The SpaceAnimal videos of *C. elegans*, *Drosophila* and zebrafish aboard the China Space Station provide one such near-closed monitoring setting.

## Repository layout

```text
PSIC-Track/
├── run_tracker.py          # command-line entry point
├── psictrack/              # strictly online tracking implementation
│   ├── __init__.py         # package interface
│   ├── types.py            # shared tracking data structures
│   ├── config.py           # YAML configuration parsing and validation
│   ├── association.py      # one-to-one assignment utilities
│   ├── base_motion.py      # temporal geometric association (TGA)
│   ├── psoi.py             # pose-structured observation integrity (PSOI)
│   ├── ncic.py             # near-closed identity conservation (NCIC)
│   ├── tracker.py          # unified causal tracking loop
│   ├── io.py               # detection and pose input adapters
│   └── cli.py              # command-line validation and output writing
├── configs/
│   ├── tga.yaml            # TGA-only configuration
│   ├── ablations/          # configurations for PSOI/NCIC combinations
│   │   ├── tga_plus_psoi.yaml
│   │   ├── tga_plus_ncic.yaml
│   │   └── tga_plus_psoi_ncic.yaml
│   └── species/            # example body-topology definitions
│       ├── README.md       # topology fields and adaptation guide
│       ├── celegans.yaml
│       ├── drosophila.yaml
│       └── zebrafish.yaml
├── requirements.txt        # Python dependencies
├── .gitignore              # generated-output exclusions
└── README.md               # repository guide
```

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

Use one of these paths directly when its topology matches your pose output, or create a YAML for another species with `keypoints`, `skeleton`, and `body_axis`. See [`configs/species/README.md`](configs/species/README.md) for the field definitions.

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
