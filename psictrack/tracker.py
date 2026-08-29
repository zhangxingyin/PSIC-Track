"""Single online PSIC-Track framework: TGA box motion plus PSOI and NCIC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .association import (
    CausalEvidenceCredibility,
    GaussianPose,
    ambiguity_scores,
    compose_joint_cost,
)
from .base_motion import BoxMotionFilter, box_cost_matrix, solve_assignment
from .config import TrackerConfig
from .psoi import (
    GraphPoseEncoder,
    GraphPoseEvidence,
    ObservationIntegrity,
    ObservationIntegrityModel,
    PoseEncoder,
    PoseEvidence,
    PoseMemory,
    SpeciesTopology,
    graph_pose_distance,
)
from .ncic import (
    ArenaVisibilityField,
    DiscreteCapacityBelief,
    IdentityAssignment,
    IdentityBelief,
    bbox_center,
    solve_identity_conserving_assignment,
)
from .types import (
    Lifecycle,
    MotionRegime,
    Observation,
    ObservationSource,
    TrackOutput,
    UpdateMode,
)

def _body_scale(bbox: np.ndarray) -> float:
    return max(float(np.sqrt(bbox[2] * bbox[3])), 1e-9)

def _closed_world_transport_energy(
    left_bbox: np.ndarray, right_bbox: np.ndarray
) -> float:
    """Dimensionless transport energy in body-length units."""
    scale = np.sqrt(0.5 * (_body_scale(left_bbox) ** 2 + _body_scale(right_bbox) ** 2))
    return float(np.linalg.norm(bbox_center(left_bbox) - bbox_center(right_bbox)) / scale)


def _normalized_uncertainty(motion: BoxMotionFilter, bbox: np.ndarray) -> float:
    variance = max(float(np.trace(motion.covariance[:2, :2])), 0.0)
    return float(np.sqrt(variance) / _body_scale(bbox))


def _graph_reliability(left: GraphPoseEvidence, right: GraphPoseEvidence) -> float:
    common = left.valid_mask & right.valid_mask
    coverage = float(np.mean(common)) if common.size else 0.0
    return float(np.clip(np.sqrt(left.quality * right.quality) * coverage, 0.0, 1.0))


def _identity_conditioned_multiplicity_energy(
    pair_cost: np.ndarray,
    observation_energy: np.ndarray,
    observation_probability: np.ndarray,
) -> np.ndarray:
    """Condition observation multiplicity on reciprocal identity support.

    A duplicate hypothesis is meaningful only relative to a primary
    observation. Reciprocal MAP support proposes that primary, while the
    detector probability supplies independent evidence that it exists, and a
    valid PSOI pose graph certifies that reciprocal support is structurally
    observable. Their product discounts duplicate energy only on a mutually
    supported, pose-qualified continuation; competing, pose-missing, and
    low-confidence edges retain their integrity energy.

    This is a parameter-free reciprocal posterior; it does not use species
    labels or a tuned distance threshold.
    """

    costs = np.asarray(pair_cost, dtype=np.float64)
    energies = np.asarray(observation_energy, dtype=np.float64)
    probabilities = np.asarray(observation_probability, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("pair_cost must be two-dimensional")
    if energies.shape != (costs.shape[1],):
        raise ValueError("observation_energy must match pair_cost columns")
    if probabilities.shape != (costs.shape[1],):
        raise ValueError("observation_probability must match pair_cost columns")
    if np.any(~np.isfinite(energies)) or np.any(energies < 0.0):
        raise ValueError("observation_energy must be finite and nonnegative")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("observation_probability must lie in [0, 1]")

    conditioned = np.broadcast_to(energies[None, :], costs.shape).copy()
    if not costs.size:
        return conditioned
    finite = np.isfinite(costs)
    safe = np.where(finite, costs, np.inf)
    row_best = np.argmin(safe, axis=1)
    column_best = np.argmin(safe, axis=0)
    for row, column in enumerate(row_best.tolist()):
        if finite[row, column] and column_best[column] == row:
            conditioned[row, column] *= 1.0 - probabilities[column]
    return conditioned


@dataclass(frozen=True)
class _PredictedBox:
    bbox_xywh: np.ndarray


@dataclass
class _BirthProposal:
    bbox_xywh: np.ndarray
    graph_pose: GraphPoseEvidence | None
    existence: float
    hits: int
    last_frame: int
    detection_id: int


class _GraphMemory:
    """Causal Gaussian memory for anatomical pose transition.

    Pose is treated as a kinematic state rather than a persistent appearance
    descriptor.  The latent state contains graph pose and graph-pose velocity;
    velocity is initialized from the first pair of reliable observations and
    all later predictions use the real frame interval.
    """

    _DIMENSION = 8
    _STATE_DIMENSION = 16

    def __init__(self, evidence: GraphPoseEvidence):
        self._static_mean = evidence.mean.copy()
        self._static_covariance = evidence.covariance.copy()
        self._state = np.zeros(self._STATE_DIMENSION, dtype=np.float64)
        self._state[: self._DIMENSION] = evidence.mean
        self._state_covariance = np.zeros(
            (self._STATE_DIMENSION, self._STATE_DIMENSION), dtype=np.float64
        )
        self._state_covariance[: self._DIMENSION, : self._DIMENSION] = (
            evidence.covariance
        )
        self._state_covariance[self._DIMENSION :, self._DIMENSION :] = (
            evidence.covariance
        )
        self._valid = evidence.valid_mask.copy()
        self._velocity_valid = np.zeros(self._DIMENSION, dtype=bool)
        self._process_covariance = evidence.covariance.copy()
        self._process_mass = 0.0
        self._last_observation_mean = evidence.mean.copy()
        self._last_observation_covariance = evidence.covariance.copy()
        self._last_observation_valid = evidence.valid_mask.copy()
        self._frames_since_observation = 0
        self._transition_probability = 0.5
        self._quality = float(evidence.quality)
        self._predicted = False

    @property
    def transition_probability(self) -> float:
        return float(self._transition_probability)

    @property
    def anatomical_identity(self) -> GraphPoseEvidence:
        """Return the non-extrapolated anatomical posterior used by PSOI.

        PSOI answers whether a candidate is structurally compatible with the
        identity. The dynamic graph state is retained only to propagate its
        uncertainty; it must not form a second identity-motion predictor during
        association.
        """

        return GraphPoseEvidence(
            self._static_mean,
            self._static_covariance,
            self._valid,
            self._quality,
        )

    @property
    def current(self) -> GraphPoseEvidence:
        probability = self._transition_probability
        transition_mean = self._state[: self._DIMENSION]
        transition_covariance = self._state_covariance[
            : self._DIMENSION, : self._DIMENSION
        ]
        mean = (
            (1.0 - probability) * self._static_mean
            + probability * transition_mean
        )
        static_residual = self._static_mean - mean
        transition_residual = transition_mean - mean
        covariance = (
            (1.0 - probability)
            * (
                self._static_covariance
                + np.outer(static_residual, static_residual)
            )
            + probability
            * (
                transition_covariance
                + np.outer(transition_residual, transition_residual)
            )
        )
        axis = mean[6:8]
        if np.all(self._valid[6:8]):
            norm = float(np.linalg.norm(axis))
            if norm > np.finfo(np.float64).eps:
                mean[6:8] = axis / norm
        return GraphPoseEvidence(mean, covariance, self._valid, self._quality)

    def distance_to(self, evidence: GraphPoseEvidence) -> float:
        """Exact two-hypothesis anatomical continuation distance.

        Association marginalizes the persistence and transition likelihoods
        instead of replacing them with one moment-matched Gaussian.  The
        latter is retained by ``current`` for continuous downstream state but
        must not create a high-likelihood posture between two discrete modes.
        """

        static = GraphPoseEvidence(
            self._static_mean,
            self._static_covariance,
            self._valid,
            self._quality,
        )
        transition = GraphPoseEvidence(
            self._state[: self._DIMENSION],
            self._state_covariance[: self._DIMENSION, : self._DIMENSION],
            self._valid,
            self._quality,
        )
        static_distance = graph_pose_distance(static, evidence)
        transition_distance = graph_pose_distance(transition, evidence)
        probability = float(np.clip(self._transition_probability, 0.0, 1.0))
        if probability <= 0.0:
            return static_distance
        if probability >= 1.0:
            return transition_distance
        log_terms = np.asarray(
            [
                np.log1p(-probability) - 0.5 * static_distance,
                np.log(probability) - 0.5 * transition_distance,
            ],
            dtype=np.float64,
        )
        maximum = float(np.max(log_terms))
        if not np.isfinite(maximum):
            return float("inf")
        log_likelihood = maximum + float(
            np.log(np.sum(np.exp(log_terms - maximum)))
        )
        return float(max(0.0, -2.0 * log_likelihood))

    @staticmethod
    def _predictive_log_likelihood(
        mean: np.ndarray,
        covariance: np.ndarray,
        evidence: GraphPoseEvidence,
        valid: np.ndarray,
    ) -> float | None:
        indices = np.flatnonzero(valid & evidence.valid_mask)
        if indices.size == 0:
            return None
        block = np.ix_(indices, indices)
        innovation = covariance[block] + evidence.covariance[block]
        residual = evidence.mean[indices] - mean[indices]
        try:
            solved = np.linalg.solve(innovation, residual)
            sign, log_determinant = np.linalg.slogdet(innovation)
        except np.linalg.LinAlgError:
            return None
        if sign <= 0.0 or not np.isfinite(log_determinant):
            return None
        dimension = float(indices.size)
        return float(
            -0.5
            * (
                residual @ solved
                + log_determinant
                + dimension * np.log(2.0 * np.pi)
            )
            / dimension
        )

    def _update_model_probability(
        self, evidence: GraphPoseEvidence, strength: float
    ) -> None:
        static_log_likelihood = self._predictive_log_likelihood(
            self._static_mean,
            self._static_covariance,
            evidence,
            self._valid,
        )
        transition_log_likelihood = self._predictive_log_likelihood(
            self._state[: self._DIMENSION],
            self._state_covariance[: self._DIMENSION, : self._DIMENSION],
            evidence,
            self._valid,
        )
        if static_log_likelihood is None or transition_log_likelihood is None:
            return
        tiny = np.finfo(np.float64).tiny
        transition_log_posterior = np.log(
            max(self._transition_probability, tiny)
        ) + strength * transition_log_likelihood
        static_log_posterior = np.log(
            max(1.0 - self._transition_probability, tiny)
        ) + strength * static_log_likelihood
        maximum = max(transition_log_posterior, static_log_posterior)
        transition_mass = np.exp(transition_log_posterior - maximum)
        static_mass = np.exp(static_log_posterior - maximum)
        self._transition_probability = float(
            transition_mass / (transition_mass + static_mass)
        )

    def _condition(
        self,
        state_indices: np.ndarray,
        values: np.ndarray,
        observation_covariance: np.ndarray,
    ) -> None:
        """Condition selected latent coordinates with a Gaussian observation."""

        if state_indices.size == 0:
            return
        observation = np.zeros(
            (state_indices.size, self._STATE_DIMENSION), dtype=np.float64
        )
        observation[np.arange(state_indices.size), state_indices] = 1.0
        residual = values - observation @ self._state
        innovation = (
            observation @ self._state_covariance @ observation.T
            + observation_covariance
        )
        cross = self._state_covariance @ observation.T
        try:
            gain = np.linalg.solve(innovation, cross.T).T
        except np.linalg.LinAlgError:
            gain = cross @ np.linalg.pinv(innovation)
        self._state += gain @ residual
        identity = np.eye(self._STATE_DIMENSION, dtype=np.float64)
        correction = identity - gain @ observation
        self._state_covariance = (
            correction @ self._state_covariance @ correction.T
            + gain @ observation_covariance @ gain.T
        )
        self._state_covariance = (
            self._state_covariance + self._state_covariance.T
        ) / 2.0

    def _normalize_body_axis(self) -> None:
        axis_indices = np.asarray([6, 7], dtype=int)
        if not np.all(self._valid[axis_indices]):
            return
        axis = self._state[axis_indices]
        norm = float(np.linalg.norm(axis))
        if norm <= np.finfo(np.float64).eps:
            self._valid[axis_indices] = False
            self._velocity_valid[axis_indices] = False
            return
        axis /= norm
        self._state[axis_indices] = axis
        if np.all(self._velocity_valid[axis_indices]):
            velocity = self._state[self._DIMENSION + axis_indices]
            self._state[self._DIMENSION + axis_indices] = (
                velocity - axis * float(axis @ velocity)
            )

    def predict(self, delta_frames: int) -> GraphPoseEvidence:
        if self._predicted:
            raise RuntimeError("graph memory must be updated or missed before another prediction")
        delta = int(delta_frames)
        if delta <= 0:
            raise ValueError("delta_frames must be positive")
        transition = np.eye(self._STATE_DIMENSION, dtype=np.float64)
        velocity_indices = np.flatnonzero(self._velocity_valid)
        transition[
            velocity_indices, self._DIMENSION + velocity_indices
        ] = float(delta)
        self._state = transition @ self._state
        self._state_covariance = (
            transition @ self._state_covariance @ transition.T
        )

        acceleration_map = np.zeros(
            (self._STATE_DIMENSION, self._DIMENSION), dtype=np.float64
        )
        acceleration_map[velocity_indices, velocity_indices] = 0.5 * delta * delta
        acceleration_map[
            self._DIMENSION + velocity_indices, velocity_indices
        ] = float(delta)
        self._state_covariance += (
            acceleration_map @ self._process_covariance @ acceleration_map.T
        )
        position_only = np.flatnonzero(self._valid & ~self._velocity_valid)
        if position_only.size:
            block = np.ix_(position_only, position_only)
            self._state_covariance[block] += (
                float(delta) * self._process_covariance[block]
            )
        self._state_covariance = (
            self._state_covariance + self._state_covariance.T
        ) / 2.0
        self._static_covariance += (
            np.eye(self._DIMENSION, dtype=np.float64) * 0.0025 * delta
        )
        self._frames_since_observation += delta
        self._quality *= float(np.exp(-0.03 * delta))
        self._normalize_body_axis()
        self._predicted = True
        return self.current

    def update(self, evidence: GraphPoseEvidence, mode: UpdateMode) -> None:
        if not self._predicted:
            raise RuntimeError("graph memory predict must precede update")
        if mode is UpdateMode.FROZEN:
            self.miss()
            return
        strength = 1.0 if mode is UpdateMode.NORMAL else 0.25
        self._update_model_probability(evidence, strength)
        elapsed = max(self._frames_since_observation, 1)
        common = self._valid & evidence.valid_mask
        indices = np.flatnonzero(common)
        if indices.size:
            block = np.ix_(indices, indices)
            self._static_mean[indices] = (
                (1.0 - strength) * self._static_mean[indices]
                + strength * evidence.mean[indices]
            )
            self._static_covariance[block] = (
                (1.0 - strength) * self._static_covariance[block]
                + strength * evidence.covariance[block]
            )
            self._condition(
                indices,
                evidence.mean[indices],
                evidence.covariance[block] / strength,
            )
        new = (~self._valid) & evidence.valid_mask
        new_indices = np.flatnonzero(new)
        if new_indices.size:
            self._static_mean[new_indices] = evidence.mean[new_indices]
            self._state[new_indices] = evidence.mean[new_indices]
            block = np.ix_(new_indices, new_indices)
            self._static_covariance[block] = evidence.covariance[block]
            self._state_covariance[block] = evidence.covariance[block] / strength

        velocity_observed = self._last_observation_valid & evidence.valid_mask
        velocity_indices = np.flatnonzero(velocity_observed)
        if velocity_indices.size:
            velocity_block = np.ix_(velocity_indices, velocity_indices)
            measured_velocity = (
                evidence.mean[velocity_indices]
                - self._last_observation_mean[velocity_indices]
            ) / float(elapsed)
            measured_covariance = (
                evidence.covariance[velocity_block]
                + self._last_observation_covariance[velocity_block]
            ) / float(elapsed * elapsed * strength)

            established = velocity_observed & self._velocity_valid
            established_indices = np.flatnonzero(established)
            if established_indices.size:
                local = np.searchsorted(velocity_indices, established_indices)
                process_residual = (
                    measured_velocity[local]
                    - self._state[self._DIMENSION + established_indices]
                )
                process_sample = np.outer(process_residual, process_residual)
                process_sample += measured_covariance[np.ix_(local, local)]
                old_mass = self._process_mass
                new_mass = old_mass + strength
                process_block = np.ix_(established_indices, established_indices)
                self._process_covariance[process_block] = (
                    old_mass * self._process_covariance[process_block]
                    + strength * process_sample
                ) / new_mass
                self._process_mass = new_mass
                self._condition(
                    self._DIMENSION + established_indices,
                    measured_velocity[local],
                    measured_covariance[np.ix_(local, local)],
                )

            initialized = velocity_observed & ~self._velocity_valid
            initialized_indices = np.flatnonzero(initialized)
            if initialized_indices.size:
                local = np.searchsorted(velocity_indices, initialized_indices)
                latent_indices = self._DIMENSION + initialized_indices
                self._state[latent_indices] = strength * measured_velocity[local]
                self._state_covariance[latent_indices, :] = 0.0
                self._state_covariance[:, latent_indices] = 0.0
                self._state_covariance[np.ix_(latent_indices, latent_indices)] = (
                    measured_covariance[np.ix_(local, local)]
                )
                self._velocity_valid[initialized_indices] = True

        self._valid |= evidence.valid_mask
        self._last_observation_mean = evidence.mean.copy()
        self._last_observation_covariance = evidence.covariance.copy()
        self._last_observation_valid = evidence.valid_mask.copy()
        self._frames_since_observation = 0
        self._quality = (1.0 - strength) * self._quality + strength * evidence.quality
        self._normalize_body_axis()
        self._predicted = False

    def miss(self) -> None:
        self._predicted = False


@dataclass
class _IdentityTrack:
    track_id: int
    b0: BoxMotionFilter
    observed_bbox: np.ndarray
    predicted_bbox: np.ndarray
    score: float
    hits: int
    time_since_update: int
    lifecycle: Lifecycle
    source: ObservationSource
    keypoints: np.ndarray | None
    keypoint_scores: np.ndarray | None
    pose_source: ObservationSource | None
    pose_memory: PoseMemory | None
    graph_memory: _GraphMemory | None
    pose_credibility: CausalEvidenceCredibility
    belief: IdentityBelief | None
    pose_prediction: GaussianPose | None = None
    graph_prediction: GraphPoseEvidence | None = None
    association_quality: float = 1.0
    update_mode: UpdateMode = UpdateMode.FROZEN

    @property
    def bbox_xywh(self) -> np.ndarray:
        return self.predicted_bbox


class _NearClosedBase:
    """Strictly causal TbD tracker for a fixed, approximately closed arena."""

    def __init__(
        self,
        config: TrackerConfig,
        topology: SpeciesTopology | None = None,
        frame_size: tuple[int, int] | None = None,
    ):
        self.config = config
        self.topology = topology
        uses_pose = bool(config.pose.enabled or config.observation_integrity.enabled)
        if uses_pose and topology is None:
            raise ValueError("species topology is required for pose-integrity tracking")
        if config.closed_world.enabled and frame_size is None:
            raise ValueError("frame_size is required for the near-closed arena model")

        self.pose_encoder = PoseEncoder(topology, config.pose) if uses_pose else None
        self.graph_encoder = GraphPoseEncoder(topology, config.pose) if uses_pose else None
        self.integrity_model = ObservationIntegrityModel(config.observation_integrity)
        self.arena = (
            ArenaVisibilityField(frame_size, config.closed_world)
            if config.closed_world.enabled and frame_size is not None
            else None
        )
        self.capacity = DiscreteCapacityBelief(config.closed_world)
        self._birth_proposals: list[_BirthProposal] = []
        self._capacity_initialized = False
        self._tracks: list[_IdentityTrack] = []
        self._next_id = 1
        self._last_frame = 0
        self.last_diagnostics: list[dict[str, object]] = []

    def _encode(
        self, observations: Sequence[Observation]
    ) -> tuple[list[PoseEvidence | None], list[GraphPoseEvidence | None]]:
        raw: list[PoseEvidence | None] = []
        graph: list[GraphPoseEvidence | None] = []
        for observation in observations:
            if self.pose_encoder is None or observation.keypoints is None:
                raw.append(None)
                graph.append(None)
            else:
                raw.append(self.pose_encoder.encode(observation))
                graph.append(self.graph_encoder.encode(observation))
        return raw, graph

    def _new_track(
        self,
        observation: Observation,
        raw: PoseEvidence | None,
        graph: GraphPoseEvidence | None,
        *,
        bootstrap: bool,
    ) -> _IdentityTrack:
        pose_memory = (
            PoseMemory(raw, self.topology, self.config.pose)
            if raw is not None and self.topology is not None
            else None
        )
        graph_memory = _GraphMemory(graph) if graph is not None and graph.quality > 0.0 else None
        belief = None
        if self.config.closed_world.enabled:
            belief = (
                IdentityBelief.bootstrap(self.config.closed_world)
                if bootstrap
                else IdentityBelief.tentative(self.config.closed_world)
            )
            lifecycle = belief.lifecycle
        else:
            lifecycle = (
                Lifecycle.ACTIVE
                if self.config.base.min_hits <= 1
                else Lifecycle.TENTATIVE
            )
        track = _IdentityTrack(
            track_id=self._next_id,
            b0=BoxMotionFilter.initiate(observation.bbox_xywh),
            observed_bbox=observation.bbox_xywh.copy(),
            predicted_bbox=observation.bbox_xywh.copy(),
            score=observation.score,
            hits=1,
            time_since_update=0,
            lifecycle=lifecycle,
            source=observation.source,
            keypoints=observation.keypoints,
            keypoint_scores=observation.keypoint_scores,
            pose_source=observation.pose_source,
            pose_memory=pose_memory,
            graph_memory=graph_memory,
            pose_credibility=CausalEvidenceCredibility(),
            belief=belief,
            update_mode=UpdateMode.NORMAL if raw is not None else UpdateMode.FROZEN,
        )
        self._next_id += 1
        return track

    def _predict_tracks(self, frame: int, delta: int) -> None:
        if self.arena is not None:
            self.arena.advance(frame)
        for track in self._tracks:
            if track.lifecycle is Lifecycle.RECOVERED:
                track.lifecycle = Lifecycle.ACTIVE
                if track.belief is not None:
                    track.belief.lifecycle = Lifecycle.ACTIVE
            track.b0.predict(delta)
            b0_bbox = track.b0.bbox_xywh.copy()
            track.graph_prediction = (
                None if track.graph_memory is None else track.graph_memory.predict(delta)
            )
            track.pose_prediction = (
                None if track.pose_memory is None else track.pose_memory.predict(delta)
            )
            track.predicted_bbox = b0_bbox
            track.time_since_update += delta

    def _association_matrices(
        self,
        observations: Sequence[Observation],
        raw: Sequence[PoseEvidence | None],
        graph: Sequence[GraphPoseEvidence | None],
    ) -> dict[str, np.ndarray]:
        rows, columns = len(self._tracks), len(observations)
        b0_cost = box_cost_matrix(
            [_PredictedBox(track.b0.bbox_xywh) for track in self._tracks],
            observations,
            self.config.base,
        )
        closed_reacquisition = np.zeros((rows, columns), dtype=bool)
        spatial = b0_cost.copy()

        persistent_columns: set[int] = set()
        if self.config.closed_world.enabled and self._birth_proposals and columns:
            proposal_assignment = solve_assignment(
                self._proposal_cost_matrix(tuple(range(columns)), observations, graph),
                self.config.base.max_assignment_cost,
            )
            persistent_columns = {column for _, column in proposal_assignment.matches}
        if self.config.closed_world.enabled:
            for row, track in enumerate(self._tracks):
                for column, observation in enumerate(observations):
                    dormant_recovery = track.lifecycle in {
                        Lifecycle.DORMANT,
                        Lifecycle.LOST,
                    }
                    if (
                        np.isfinite(spatial[row, column])
                        or (
                            not dormant_recovery
                            and column not in persistent_columns
                        )
                    ):
                        continue
                    spatial[row, column] = _closed_world_transport_energy(
                        track.b0.bbox_xywh, observation.bbox_xywh
                    )
                    closed_reacquisition[row, column] = True

        pose_distance = np.full_like(spatial, np.inf)
        pose_support = np.zeros_like(spatial)
        for row, track in enumerate(self._tracks):
            for column in range(columns):
                if track.graph_prediction is not None and graph[column] is not None:
                    identity_pose = (
                        track.graph_memory.anatomical_identity
                        if track.graph_memory is not None
                        else track.graph_prediction
                    )
                    pose_distance[row, column] = graph_pose_distance(
                        identity_pose, graph[column]
                    )
                    pose_support[row, column] = _graph_reliability(
                        track.graph_prediction, graph[column]
                    )

        pose_credibility = np.broadcast_to(
            np.asarray(
                [track.pose_credibility.strength for track in self._tracks],
                dtype=np.float64,
            )[:, None],
            spatial.shape,
        ).copy()
        feasible = np.isfinite(spatial)
        ambiguity = ambiguity_scores(spatial, feasible)
        if self.config.association.enabled:
            combined, pose_term = compose_joint_cost(
                spatial,
                pose_distance,
                ambiguity,
                pose_support * pose_credibility,
            )
        else:
            combined = spatial.copy()
            pose_term = np.zeros_like(spatial)

        if np.any(closed_reacquisition):
            energy = np.maximum(combined[closed_reacquisition], 0.0)
            combined[closed_reacquisition] = (
                self.config.base.max_assignment_cost * energy / (1.0 + energy)
            )
        return {
            "tga": b0_cost,
            "spatial": spatial,
            "pose": pose_distance,
            "pose_support": pose_support,
            "pose_credibility": pose_credibility,
            "ambiguity": ambiguity,
            "pose_term": pose_term,
            "combined": combined,
            "closed_reacquisition": closed_reacquisition,
        }

    @staticmethod
    def _counterfactual_distances(
        distance: np.ndarray,
        support: np.ndarray,
        row: int,
        column: int,
    ) -> np.ndarray:
        row_mask = np.isfinite(distance[row]) & (support[row] > 0.0)
        row_mask[column] = False
        column_mask = np.isfinite(distance[:, column]) & (support[:, column] > 0.0)
        column_mask[row] = False
        return np.concatenate(
            [distance[row, row_mask], distance[column_mask, column]]
        )

    def _update_evidence_credibility(
        self,
        matches: Sequence[tuple[int, int]],
        matrices: dict[str, np.ndarray],
    ) -> None:
        for row, column in matches:
            track = self._tracks[row]
            streams = (
                (track.pose_credibility, "pose", "pose_support"),
            )
            ambiguity = float(matrices["ambiguity"][row, column])
            for belief, distance_key, support_key in streams:
                selected = float(matrices[distance_key][row, column])
                reliability = float(matrices[support_key][row, column])
                if not np.isfinite(selected) or reliability <= 0.0:
                    continue
                alternatives = self._counterfactual_distances(
                    matrices[distance_key],
                    matrices[support_key],
                    row,
                    column,
                )
                belief.update(
                    selected,
                    alternatives,
                    reliability=reliability,
                    ambiguity=ambiguity,
                )

    def _closed_assignment(
        self,
        matrices: dict[str, np.ndarray],
        observations: Sequence[Observation],
        graph: Sequence[GraphPoseEvidence | None],
        integrity: Sequence[ObservationIntegrity],
        duplicate_affinity: np.ndarray,
    ) -> IdentityAssignment:
        current_mass = sum(
            track.belief.existence
            for track in self._tracks
            if track.belief is not None
        )
        pressure = self.capacity.birth_pressure(current_mass)
        # Observation multiplicity is relational rather than an unconditional
        # property of one detection. A reciprocal identity posterior estimates
        # how likely each edge is the primary observation; multiplicity energy
        # remains strongest on competing, non-reciprocal explanations.
        probability_floor = np.finfo(np.float64).eps
        identity_observation_energy = np.asarray(
            [
                -np.log(
                    max(1.0 - state.duplicate_probability, probability_floor)
                )
                for state in integrity
            ],
            dtype=np.float64,
        )
        conditioned_multiplicity = _identity_conditioned_multiplicity_energy(
            matrices["combined"],
            identity_observation_energy,
            np.asarray(
                [
                    observation.score
                    if evidence is not None and evidence.quality > 0.0
                    else 0.0
                    for observation, evidence in zip(observations, graph)
                ],
                dtype=np.float64,
            ),
        )
        pair = matrices["combined"] + conditioned_multiplicity
        matrices["identity_pair"] = pair
        matrices["identity_observation_energy"] = identity_observation_energy
        matrices["identity_conditioned_multiplicity"] = conditioned_multiplicity

        miss_cost = np.zeros(len(self._tracks), dtype=np.float64)
        for row, track in enumerate(self._tracks):
            center = bbox_center(track.predicted_bbox)
            visibility = 0.5 if self.arena is None else self.arena.visibility(center)
            existence = 0.5 if track.belief is None else track.belief.existence
            miss_probability = 1.0 - existence * visibility
            miss_cost[row] = -np.log(max(miss_probability, probability_floor))

        birth_cost = np.zeros(len(observations), dtype=np.float64)
        clutter_cost = np.zeros(len(observations), dtype=np.float64)
        for column, (observation, state) in enumerate(zip(observations, integrity)):
            support = (
                0.5
                if self.arena is None
                else self.arena.support(bbox_center(observation.bbox_xywh))
            )
            birth_probability = (
                observation.score
                * state.singleton_probability * support * (1.0 - pressure)
            )
            clutter_probability = max(
                state.clutter_probability,
                state.duplicate_probability,
                state.merge_probability,
            )
            birth_cost[column] = -np.log(
                max(birth_probability, probability_floor)
            )
            clutter_cost[column] = -np.log(
                max(clutter_probability, probability_floor)
            )
        primary_rows = np.asarray(
            [
                track.lifecycle
                not in {Lifecycle.DORMANT, Lifecycle.LOST, Lifecycle.REMOVED}
                for track in self._tracks
            ],
            dtype=bool,
        )
        primary_support = (
            ~matrices["closed_reacquisition"] & primary_rows[:, None]
        )
        assignment = solve_identity_conserving_assignment(
            pair,
            miss_cost=miss_cost,
            birth_cost=birth_cost,
            clutter_cost=clutter_cost,
            max_pair_cost=self.config.base.max_assignment_cost,
            local_support_mask=primary_support,
            duplicate_affinity=duplicate_affinity,
        )
        if duplicate_affinity.shape != (len(observations), len(observations)):
            raise ValueError("duplicate affinity must be square over observations")
        if not self._capacity_initialized and not self._tracks:
            # At sequence start, one representative is retained for each hard
            # duplicate component; soft overlaps remain independent proposals.
            components: dict[tuple[str, int], list[int]] = {}
            for column, state in enumerate(integrity):
                key = (
                    ("observation", column)
                    if state.duplicate_group is None
                    else ("duplicate_group", state.duplicate_group)
                )
                components.setdefault(key, []).append(column)
            births = tuple(
                sorted(
                    max(
                        members,
                        key=lambda column: (
                            observations[column].score
                            * integrity[column].singleton_probability,
                            observations[column].score,
                            -column,
                        ),
                    )
                    for members in components.values()
                )
            )
            birth_set = set(births)
            clutters = tuple(
                column
                for column in range(len(observations))
                if column not in birth_set
            )
            return IdentityAssignment((), (), births, clutters, 0.0)

        matched_columns = tuple(column for _, column in assignment.matches)
        residual_columns = tuple(
            sorted((*assignment.births, *assignment.clutters))
        )
        adjusted_birth_cost = birth_cost.copy()
        adjusted_clutter_cost = clutter_cost.copy()
        for column in residual_columns:
            explained = (
                max(
                    float(duplicate_affinity[column, matched])
                    for matched in matched_columns
                )
                if matched_columns
                else 0.0
            )
            unexplained = float(np.clip(1.0 - explained, 0.0, 1.0))
            base_clutter = float(np.exp(-clutter_cost[column]))
            residual_clutter_probability = 1.0 - (1.0 - base_clutter) * (
                1.0 - explained
            )
            adjusted_birth_cost[column] = birth_cost[column] - np.log(
                max(unexplained, probability_floor)
            )
            adjusted_clutter_cost[column] = -np.log(
                max(residual_clutter_probability, probability_floor)
            )

        births = tuple(
            column
            for column in residual_columns
            if adjusted_birth_cost[column] <= adjusted_clutter_cost[column]
        )
        clutters = tuple(
            column
            for column in residual_columns
            if adjusted_birth_cost[column] > adjusted_clutter_cost[column]
        )
        total_cost = float(
            sum(pair[row, column] for row, column in assignment.matches)
            + sum(miss_cost[row] for row in assignment.misses)
            + sum(adjusted_birth_cost[column] for column in births)
            + sum(adjusted_clutter_cost[column] for column in clutters)
        )
        closed = IdentityAssignment(
            matches=assignment.matches,
            misses=assignment.misses,
            births=births,
            clutters=clutters,
            total_cost=total_cost,
        )
        return closed

    def _proposal_cost_matrix(
        self,
        columns: Sequence[int],
        observations: Sequence[Observation],
        graph: Sequence[GraphPoseEvidence | None],
    ) -> np.ndarray:
        selected = [observations[column] for column in columns]
        cost = box_cost_matrix(
            [_PredictedBox(proposal.bbox_xywh) for proposal in self._birth_proposals],
            selected,
            self.config.base,
        )
        for row, proposal in enumerate(self._birth_proposals):
            if proposal.graph_pose is None:
                continue
            for local_column, column in enumerate(columns):
                candidate = graph[column]
                if candidate is None or not np.isfinite(cost[row, local_column]):
                    continue
                distance = graph_pose_distance(proposal.graph_pose, candidate)
                if not np.isfinite(distance):
                    continue
                reliability = _graph_reliability(proposal.graph_pose, candidate)
                cost[row, local_column] += reliability * (-np.expm1(-0.5 * distance))
        return cost

    def _update_birth_proposals(
        self,
        frame: int,
        columns: Sequence[int],
        new_proposal_columns: Sequence[int],
        observations: Sequence[Observation],
        graph: Sequence[GraphPoseEvidence | None],
        integrity: Sequence[ObservationIntegrity],
        *,
        bootstrap: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...], list[dict[str, object]]]:
        """Accumulate causal evidence before assigning a new persistent ID.

        New proposals are created only from current birth evidence. Existing
        proposals may, however, consume any causally compatible residual
        observation. This separates conservative identity creation from
        continuity-preserving proposal evolution.
        """
        def observation_evidence(column: int) -> float:
            state = integrity[column]
            return float(
                np.clip(
                    observations[column].score * state.singleton_probability,
                    0.0,
                    1.0,
                )
            )


        diagnostics: list[dict[str, object]] = []
        promoted: list[int] = []
        deferred: list[int] = []
        new_proposals = set(new_proposal_columns)
        candidate_columns: list[int] = []
        for column in columns:
            existence = observation_evidence(column)
            if bootstrap and existence >= 0.5:
                promoted.append(column)
                diagnostics.append(
                    {
                        "frame": frame,
                        "track_id": None,
                        "detection_id": observations[column].detection_id,
                        "interpretation": "proposal_promoted",
                        "proposal_existence": existence,
                        "proposal_hits": 1,
                        "capacity": self.capacity.value,
                    }
                )
            else:
                candidate_columns.append(column)

        assignment = solve_assignment(
            self._proposal_cost_matrix(candidate_columns, observations, graph),
            self.config.base.max_assignment_cost,
        )
        retained: list[_BirthProposal] = []
        for row, local_column in assignment.matches:
            proposal = self._birth_proposals[row]
            column = candidate_columns[local_column]
            observation_existence = observation_evidence(column)
            proposal.existence = float(
                1.0
                - (1.0 - proposal.existence) * (1.0 - observation_existence)
            )
            proposal.hits += 1
            proposal.last_frame = frame
            proposal.bbox_xywh = observations[column].bbox_xywh.copy()
            proposal.graph_pose = graph[column]
            proposal.detection_id = observations[column].detection_id
            decision_due = bool(
                proposal.hits >= self.config.closed_world.birth_confirmation
            )
            is_promoted = bool(decision_due and proposal.existence >= 0.5)
            diagnostics.append(
                {
                    "frame": frame,
                    "track_id": None,
                    "detection_id": observations[column].detection_id,
                    "interpretation": (
                        "proposal_promoted"
                        if is_promoted
                        else "proposal_rejected"
                        if decision_due
                        else "proposal"
                    ),
                    "proposal_existence": proposal.existence,
                    "proposal_hits": proposal.hits,
                    "capacity": self.capacity.value,
                }
            )
            if is_promoted:
                promoted.append(column)
            elif decision_due:
                deferred.append(column)
            else:
                retained.append(proposal)
                deferred.append(column)

        for row in assignment.unmatched_rows:
            proposal = self._birth_proposals[row]
            if frame - proposal.last_frame >= self.config.closed_world.birth_confirmation:
                diagnostics.append(
                    {
                        "frame": frame,
                        "track_id": None,
                        "detection_id": proposal.detection_id,
                        "interpretation": "proposal_expired",
                        "proposal_existence": proposal.existence,
                        "proposal_hits": proposal.hits,
                        "capacity": self.capacity.value,
                    }
                )
            else:
                retained.append(proposal)

        for local_column in assignment.unmatched_cols:
            column = candidate_columns[local_column]
            if column not in new_proposals:
                continue
            existence = observation_evidence(column)
            if (
                self.config.closed_world.birth_confirmation <= 1
                and existence >= 0.5
            ):
                promoted.append(column)
                interpretation = "proposal_promoted"
            else:
                retained.append(
                    _BirthProposal(
                        bbox_xywh=observations[column].bbox_xywh.copy(),
                        graph_pose=graph[column],
                        existence=existence,
                        hits=1,
                        last_frame=frame,
                        detection_id=observations[column].detection_id,
                    )
                )
                interpretation = "proposal"
                deferred.append(column)
            diagnostics.append(
                {
                    "frame": frame,
                    "track_id": None,
                    "detection_id": observations[column].detection_id,
                    "interpretation": interpretation,
                    "proposal_existence": existence,
                    "proposal_hits": 1,
                    "capacity": self.capacity.value,
                }
            )

        self._birth_proposals = retained
        return (
            tuple(sorted(set(promoted))),
            tuple(sorted(set(deferred))),
            diagnostics,
        )

    def _select_update_mode(
        self,
        evidence: PoseEvidence | None,
        state: ObservationIntegrity,
        reliability: float,
        ambiguity: float,
        quality: float,
    ) -> UpdateMode:
        if evidence is None or evidence.quality <= 0.0:
            return UpdateMode.FROZEN
        support = (
            evidence.quality
            * state.singleton_probability
            * quality
            * (1.0 - ambiguity)
        )
        competing = max(
            1.0 - support,
            state.duplicate_probability,
            state.merge_probability,
            ambiguity,
        )
        if support >= competing:
            return UpdateMode.NORMAL
        if support > max(state.duplicate_probability, state.merge_probability):
            return UpdateMode.WEAK
        return UpdateMode.FROZEN

    def _match(
        self,
        frame: int,
        row: int,
        column: int,
        observations: Sequence[Observation],
        raw: Sequence[PoseEvidence | None],
        graph: Sequence[GraphPoseEvidence | None],
        integrity: Sequence[ObservationIntegrity],
        matrices: dict[str, np.ndarray],
    ) -> None:
        track = self._tracks[row]
        observation = observations[column]
        state = integrity[column]
        was_dormant = track.lifecycle in {Lifecycle.DORMANT, Lifecycle.LOST}
        match_cost = float(matrices["combined"][row, column])
        quality = float(
            np.clip(1.0 - match_cost / self.config.base.max_assignment_cost, 0.0, 1.0)
        )
        reliability = float(matrices["pose_support"][row, column])
        mode = self._select_update_mode(
            raw[column], state, reliability, float(matrices["ambiguity"][row, column]), quality
        )

        # TGA remains a detector-only, full-strength counterfactual reference.
        track.b0.update(observation.bbox_xywh)
        track.observed_bbox = observation.bbox_xywh.copy()
        track.predicted_bbox = observation.bbox_xywh.copy()
        track.score = observation.score
        track.hits += 1
        track.time_since_update = 0
        track.source = observation.source
        track.keypoints = observation.keypoints
        track.keypoint_scores = observation.keypoint_scores
        track.pose_source = observation.pose_source
        track.association_quality = quality
        track.update_mode = mode

        if track.graph_memory is None and graph[column] is not None and graph[column].quality > 0.0:
            track.graph_memory = _GraphMemory(graph[column])
        elif track.graph_memory is not None:
            if graph[column] is None:
                track.graph_memory.miss()
            else:
                track.graph_memory.update(graph[column], mode)
        if track.pose_memory is None and raw[column] is not None and self.topology is not None:
            track.pose_memory = PoseMemory(raw[column], self.topology, self.config.pose)
        elif track.pose_memory is not None:
            if raw[column] is None:
                track.pose_memory.miss()
            else:
                track.pose_memory.update(raw[column], mode)

        if track.belief is not None:
            track.belief.match(quality)
            track.lifecycle = track.belief.lifecycle
        else:
            track.lifecycle = (
                Lifecycle.RECOVERED
                if was_dormant
                else Lifecycle.ACTIVE
                if track.hits >= self.config.base.min_hits
                else Lifecycle.TENTATIVE
            )
        if was_dormant:
            track.lifecycle = Lifecycle.RECOVERED
            if track.belief is not None:
                track.belief.lifecycle = Lifecycle.RECOVERED
        if self.arena is not None:
            self.arena.record_hit(bbox_center(observation.bbox_xywh), weight=max(quality, 0.1))

        self.last_diagnostics.append(
            {
                "frame": frame,
                "track_id": track.track_id,
                "detection_id": observation.detection_id,
                "interpretation": "recovery" if was_dormant else "match",
                "singleton_probability": state.singleton_probability,
                "duplicate_probability": state.duplicate_probability,
                "merge_probability": state.merge_probability,
                "clutter_probability": state.clutter_probability,
                "b0_cost": float(matrices["tga"][row, column]),
                "spatial_cost": float(matrices["spatial"][row, column]),
                "pose_distance": float(matrices["pose"][row, column]),
                "pose_cost_term": float(matrices["pose_term"][row, column]),
                "pose_credibility": float(matrices["pose_credibility"][row, column]),
                "ambiguity": float(matrices["ambiguity"][row, column]),
                "combined_cost": match_cost,
                "identity_pair_cost": float(
                    matrices.get("identity_pair", matrices["combined"])[row, column]
                ),
                "duplicate_occupancy_energy": float(
                    matrices.get(
                        "identity_observation_energy",
                        np.zeros(len(observations), dtype=np.float64),
                    )[column]
                ),
                "conditioned_multiplicity_energy": float(
                    matrices.get(
                        "identity_conditioned_multiplicity",
                        np.zeros_like(matrices["combined"]),
                    )[row, column]
                ),
                "association_quality": quality,
                "update_mode": mode.value,
                "closed_world_reacquisition": bool(matrices["closed_reacquisition"][row, column]),
                "identity_existence": None if track.belief is None else track.belief.existence,
                "capacity": self.capacity.value if self.config.closed_world.enabled else None,
            }
        )

    def _birth(
        self,
        frame: int,
        column: int,
        observations: Sequence[Observation],
        raw: Sequence[PoseEvidence | None],
        graph: Sequence[GraphPoseEvidence | None],
        integrity: Sequence[ObservationIntegrity],
        *,
        bootstrap: bool,
    ) -> None:
        track = self._new_track(
            observations[column], raw[column], graph[column], bootstrap=bootstrap
        )
        self._tracks.append(track)
        if self.arena is not None:
            self.arena.record_hit(bbox_center(observations[column].bbox_xywh))
        state = integrity[column]
        self.last_diagnostics.append(
            {
                "frame": frame,
                "track_id": track.track_id,
                "detection_id": observations[column].detection_id,
                "interpretation": "birth",
                "singleton_probability": state.singleton_probability,
                "duplicate_probability": state.duplicate_probability,
                "merge_probability": state.merge_probability,
                "clutter_probability": state.clutter_probability,
                "update_mode": track.update_mode.value,
                "identity_existence": None if track.belief is None else track.belief.existence,
                "capacity": self.capacity.value if self.config.closed_world.enabled else None,
            }
        )

    def _miss(self, frame: int, track: _IdentityTrack) -> bool:
        if track.pose_memory is not None:
            track.pose_memory.miss()
        if track.graph_memory is not None:
            track.graph_memory.miss()
        center = bbox_center(track.predicted_bbox)
        visibility = 0.5 if self.arena is None else self.arena.visibility(center)
        if track.belief is not None:
            track.belief.miss(visibility=visibility)
            track.lifecycle = track.belief.lifecycle
        else:
            track.lifecycle = Lifecycle.LOST
        track.update_mode = UpdateMode.FROZEN
        track.keypoints = None
        track.keypoint_scores = None
        track.pose_source = None
        track.association_quality = 0.0

        removable = (
            track.time_since_update > self.config.base.max_age
            if track.belief is None
            else track.belief.removable
            or track.time_since_update > 4 * max(self.config.base.max_age, 1)
        )
        if removable:
            track.lifecycle = Lifecycle.REMOVED
            return False
        self.last_diagnostics.append(
            {
                "frame": frame,
                "track_id": track.track_id,
                "detection_id": None,
                "interpretation": "miss",
                "arena_visibility": visibility,
                "arena_support": 0.5 if self.arena is None else self.arena.support(center),
                "identity_existence": None if track.belief is None else track.belief.existence,
                "capacity": self.capacity.value if self.config.closed_world.enabled else None,
                "update_mode": UpdateMode.FROZEN.value,
            }
        )
        return True

    def _output(self, frame: int, track: _IdentityTrack, observed: bool) -> TrackOutput:
        propagated = False
        if not observed and track.belief is not None:
            center = bbox_center(track.predicted_bbox)
            visibility = 0.5 if self.arena is None else self.arena.visibility(center)
            propagated = track.belief.propagation_eligible(
                visibility=visibility,
                normalized_uncertainty=_normalized_uncertainty(track.b0, track.predicted_bbox),
            )
            if propagated:
                for item in reversed(self.last_diagnostics):
                    if item.get("track_id") == track.track_id:
                        item["interpretation"] = "propagation"
                        break
        if not observed and self.arena is not None:
            # Commit the miss only after the current-frame propagation
            # decision.  The arena posterior entering frame t may gate frame
            # t; evidence generated at t is available from t+1 onward.
            # Learning uses the detector-only TGA location, independent of
            # the visibility field it will update for later frames.
            reference_center = bbox_center(track.b0.bbox_xywh)
            reference_visibility = self.arena.visibility(reference_center)
            self.arena.record_expected_miss(
                reference_center,
                weight=0.5 + 0.5 * reference_visibility,
            )
        return TrackOutput(
            frame=frame,
            track_id=track.track_id,
            bbox_xywh=track.observed_bbox if observed else track.predicted_bbox,
            score=track.score,
            lifecycle=track.lifecycle,
            observed=observed,
            emit_mot=observed or propagated,
            observation_source=track.source if observed else ObservationSource.PROPAGATED,
            association_quality=track.association_quality if observed else 0.0,
            update_mode=track.update_mode if observed else UpdateMode.FROZEN,
            motion_regime=MotionRegime.UNKNOWN,
            motion_expert="cv",
            motion_control_weight=0.0,
            motion_prediction_variance=float(np.trace(track.b0.covariance[:2, :2])),
            keypoints=track.keypoints if observed else None,
            keypoint_scores=track.keypoint_scores if observed else None,
            pose_source=track.pose_source if observed else None,
        )

    def update(self, frame: int, observations: Sequence[Observation]) -> list[TrackOutput]:
        if frame <= self._last_frame:
            raise ValueError("frames must be strictly increasing")
        if any(observation.frame != frame for observation in observations):
            raise ValueError("observation frame must match tracker frame")
        delta = frame - self._last_frame if self._last_frame else 1
        accepted = [
            observation
            for observation in observations
            if observation.score >= self.config.base.min_score
        ]
        self._predict_tracks(frame, delta)
        raw, graph = self._encode(accepted)
        integrity = self.integrity_model.assess(
            accepted, graph, [track.predicted_bbox for track in self._tracks]
        )
        duplicate_affinity = self.integrity_model.duplicate_affinity(
            accepted, graph
        )
        matrices = self._association_matrices(accepted, raw, graph)
        if self.config.closed_world.enabled:
            assignment = self._closed_assignment(
                matrices, accepted, graph, integrity, duplicate_affinity
            )
            matches = assignment.matches
            missed = assignment.misses
            births = assignment.births
            clutters = assignment.clutters
        else:
            standard = solve_assignment(
                matrices["combined"], self.config.base.max_assignment_cost
            )
            matches = standard.matches
            missed = standard.unmatched_rows
            births = standard.unmatched_cols
            clutters = ()

        self.last_diagnostics = []
        observed_ids: set[int] = set()
        for row, column in matches:
            self._match(frame, row, column, accepted, raw, graph, integrity, matrices)
            observed_ids.add(self._tracks[row].track_id)
        self._update_evidence_credibility(matches, matrices)

        bootstrap = not self._capacity_initialized and not self._tracks
        if self.config.closed_world.enabled:
            birth_candidates = tuple(births)
            residual_columns = tuple(sorted((*births, *clutters)))
            proposal_columns = (
                residual_columns if self._birth_proposals else birth_candidates
            )
            births, deferred, proposal_diagnostics = self._update_birth_proposals(
                frame,
                proposal_columns,
                birth_candidates,
                accepted,
                graph,
                integrity,
                bootstrap=bootstrap,
            )
            resolved = set(births) | set(deferred)
            clutters = tuple(
                column for column in residual_columns if column not in resolved
            )
            self.last_diagnostics.extend(proposal_diagnostics)
        for column in births:
            self._birth(
                frame,
                column,
                accepted,
                raw,
                graph,
                integrity,
                # A promoted proposal already satisfies birth confirmation.
                bootstrap=(self.config.closed_world.enabled or bootstrap),
            )
            observed_ids.add(self._tracks[-1].track_id)

        for column in clutters:
            state = integrity[column]
            self.last_diagnostics.append(
                {
                    "frame": frame,
                    "track_id": None,
                    "detection_id": accepted[column].detection_id,
                    "interpretation": "clutter",
                    "singleton_probability": state.singleton_probability,
                    "duplicate_probability": state.duplicate_probability,
                    "merge_probability": state.merge_probability,
                    "clutter_probability": state.clutter_probability,
                    "capacity": self.capacity.value,
                }
            )

        retained: list[_IdentityTrack] = []
        missed_indices = set(missed)
        for index, track in enumerate(self._tracks):
            # Births were appended after assignment and are therefore never misses.
            if index in missed_indices and not self._miss(frame, track):
                continue
            retained.append(track)
        self._tracks = retained

        if self.config.closed_world.enabled:
            existence_evidence = [
                track.belief.existence
                for track in self._tracks
                if track.belief is not None
                and track.lifecycle is not Lifecycle.REMOVED
            ]
            existence_evidence.extend(
                proposal.existence for proposal in self._birth_proposals
            )
            if existence_evidence:
                self.capacity.observe(existence_evidence)
                self._capacity_initialized = True
        self._last_frame = frame
        return [
            self._output(frame, track, track.track_id in observed_ids)
            for track in sorted(self._tracks, key=lambda item: item.track_id)
            if track.lifecycle is not Lifecycle.TENTATIVE or track.track_id in observed_ids
        ]


class NearClosedIdentityTracker(_NearClosedBase):
    """Single TGA+PSOI+NCIC online implementation for a near-closed arena."""


class PSICTracker:
    """Stable public facade over the single TGA+PSOI+NCIC online core."""

    def __init__(
        self,
        config: TrackerConfig,
        topology: SpeciesTopology | None = None,
        frame_size: tuple[int, int] | None = None,
    ):
        self.config = config
        self._implementation = NearClosedIdentityTracker(
            config, topology=topology, frame_size=frame_size
        )

    def update(self, frame: int, observations: Sequence[Observation]) -> list[TrackOutput]:
        return self._implementation.update(frame, observations)

    def __getattr__(self, name: str):
        return getattr(self._implementation, name)
