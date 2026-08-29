# Species topology configuration

Each YAML file in this directory is a static, species-level topology specification used by PSOI to interpret the ordered keypoints predicted for one detection. It is not a per-frame annotation file and it does not contain ground-truth tracks or poses.

Use the YAML whose keypoint order exactly matches the pose estimator output. The examples correspond to the three SpaceAnimal model-animal topologies.

## Fields

- `species`: species identifier used by the runner.
- `skeleton_index_base`: index convention for `skeleton`; the provided files use zero-based indices.
- `keypoints`: semantic names and ordered output channels of the pose estimator.
- `skeleton`: pairs of keypoint indices defining the body graph. PSOI uses these edges to construct pose-structure features and connectivity evidence.
- `body_axis`: two keypoint indices defining the directed body axis used for orientation features.

## Topology provenance

For the provided examples, `keypoints` and `skeleton` follow the released pose-annotation schema and the corresponding ViTPose output order. `body_axis` selects an anatomically meaningful directed axis, such as head to tail.

## Creating a topology for another species

1. Obtain the exact keypoint names and output order from the pose-estimation annotation schema.
2. Define anatomically valid skeletal edges and a directed body axis.
3. Verify that the number and order of `keypoints` match every pose record passed to the tracker.
