"""NCIC: near-closed arena, posterior capacity, and identity conservation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ClosedWorldConfig
from .types import Lifecycle


_GRID_SIZE = 16
_BETA_PRIOR = 1.0
_FIELD_DECAY = 0.995
_PROBABILITY_EPS = np.finfo(np.float64).eps


class ArenaVisibilityField:
    """Online Beta field from reliable hits and expected identity misses."""

    def __init__(self, frame_size: tuple[int, int], config: ClosedWorldConfig):
        config.validate()
        if len(frame_size) != 2 or min(frame_size) <= 0:
            raise ValueError("frame_size must be positive (width, height)")
        self.width = float(frame_size[0])
        self.height = float(frame_size[1])
        self.config = config
        shape = (_GRID_SIZE, _GRID_SIZE)
        self._hits = np.zeros(shape, dtype=np.float64)
        self._misses = np.zeros(shape, dtype=np.float64)
        self._support = np.zeros(shape, dtype=np.float64)
        self._last_frame = 0

    def advance(self, frame: int) -> None:
        if int(frame) != frame or frame < 1:
            raise ValueError("frame must be a positive integer")
        if self._last_frame and frame <= self._last_frame:
            raise ValueError("arena frames must be strictly increasing")
        delta = 1 if not self._last_frame else int(frame) - self._last_frame
        decay = _FIELD_DECAY**delta
        self._hits *= decay
        self._misses *= decay
        self._support *= decay
        self._last_frame = int(frame)

    def _cell(self, center: np.ndarray) -> tuple[int, int]:
        value = np.asarray(center, dtype=np.float64)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("arena center must be a finite two-vector")
        column = int(
            np.floor(
                np.clip(value[0] / self.width, 0.0, 1.0 - _PROBABILITY_EPS)
                * _GRID_SIZE
            )
        )
        row = int(
            np.floor(
                np.clip(value[1] / self.height, 0.0, 1.0 - _PROBABILITY_EPS)
                * _GRID_SIZE
            )
        )
        return row, column

    def record_hit(self, center: np.ndarray, weight: float = 1.0) -> None:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("arena hit weight must be finite and nonnegative")
        cell = self._cell(center)
        self._hits[cell] += weight
        self._support[cell] += weight

    def record_expected_miss(self, center: np.ndarray, weight: float = 1.0) -> None:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("arena miss weight must be finite and nonnegative")
        self._misses[self._cell(center)] += weight

    def visibility(self, center: np.ndarray) -> float:
        cell = self._cell(center)
        return float(
            (self._hits[cell] + _BETA_PRIOR)
            / (self._hits[cell] + self._misses[cell] + 2.0 * _BETA_PRIOR)
        )

    def support(self, center: np.ndarray) -> float:
        cell = self._cell(center)
        maximum = float(np.max(self._support))
        return float(
            (self._support[cell] + _BETA_PRIOR)
            / (maximum + 2.0 * _BETA_PRIOR)
        )

    def normalized_boundary_distance(self, center: np.ndarray) -> float:
        value = np.asarray(center, dtype=np.float64)
        distances = np.array(
            [value[0], self.width - value[0], value[1], self.height - value[1]]
        )
        return float(
            np.clip(
                np.min(distances) / max(min(self.width, self.height), _PROBABILITY_EPS),
                0.0,
                0.5,
            )
        )


def bbox_center(bbox_xywh: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox_xywh, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError("bbox must be a finite xywh vector")
    return bbox[:2] + bbox[2:] / 2.0


class DiscreteCapacityBelief:
    """Causal posterior over near-closed arena cardinality.

    Supplied probabilities describe existing identities or persistent birth
    proposals. Their Poisson-binomial count is a lower bound on capacity, so a
    temporary detector miss cannot directly erase an animal.
    """

    def __init__(self, config: ClosedWorldConfig):
        config.validate()
        self.config = config
        self._pmf = np.empty(0, dtype=np.float64)

    @staticmethod
    def _resize(values: np.ndarray, size: int) -> np.ndarray:
        result = np.zeros(size, dtype=np.float64)
        result[: min(size, values.size)] = values[:size]
        total = float(np.sum(result))
        return result / total if total > 0.0 else np.full(size, 1.0 / size)

    @staticmethod
    def _transition(values: np.ndarray) -> np.ndarray:
        """One-count symmetric near-closed transition with reflecting edges."""
        result = 0.5 * values
        result[:-1] += 0.25 * values[1:]
        result[1:] += 0.25 * values[:-1]
        result[0] += 0.25 * values[0]
        result[-1] += 0.25 * values[-1]
        return result / np.sum(result)

    @property
    def value(self) -> float:
        if not self._pmf.size:
            return 0.0
        return float(np.arange(self._pmf.size, dtype=np.float64) @ self._pmf)

    @property
    def uncertainty(self) -> float:
        if not self._pmf.size:
            return 1.0
        support = np.arange(self._pmf.size, dtype=np.float64)
        return float(np.sum(self._pmf * np.square(support - self.value)))

    @property
    def posterior(self) -> np.ndarray:
        result = self._pmf.copy()
        result.setflags(write=False)
        return result

    def observe(self, existence_probabilities) -> float:
        probabilities = np.asarray(tuple(existence_probabilities), dtype=np.float64)
        if probabilities.ndim != 1 or np.any(~np.isfinite(probabilities)):
            raise ValueError("existence probabilities must be a finite vector")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("existence probabilities must lie in [0, 1]")

        visible_pmf = np.asarray([1.0], dtype=np.float64)
        for probability in probabilities:
            visible_pmf = np.convolve(
                visible_pmf,
                np.asarray([1.0 - probability, probability], dtype=np.float64),
            )

        support_size = max(self._pmf.size, probabilities.size + 2)
        if self._pmf.size:
            prior = self._transition(self._resize(self._pmf, support_size))
        else:
            prior = np.full(support_size, 1.0 / support_size, dtype=np.float64)
        lower_bound_likelihood = np.ones(support_size, dtype=np.float64)
        cumulative = np.cumsum(visible_pmf)
        lower_bound_likelihood[: cumulative.size] = cumulative
        posterior = prior * np.maximum(lower_bound_likelihood, _PROBABILITY_EPS)
        self._pmf = posterior / np.sum(posterior)
        return self.value

    def slot_probability(self, current_count: int) -> float:
        if int(current_count) != current_count or current_count < 0:
            raise ValueError("current_count must be a nonnegative integer")
        if not self._pmf.size:
            return 1.0
        start = min(int(current_count) + 1, self._pmf.size)
        return float(np.sum(self._pmf[start:]))

    def birth_pressure(self, current_identity_mass: float) -> float:
        if not np.isfinite(current_identity_mass) or current_identity_mass < 0:
            raise ValueError("current_identity_mass must be finite and nonnegative")
        if not self._pmf.size:
            return 0.0
        occupied = min(int(np.floor(current_identity_mass)), self._pmf.size - 1)
        return float(np.sum(self._pmf[: occupied + 1]))


class SoftCapacityBelief(DiscreteCapacityBelief):
    """Compatibility name for older imports; new code uses discrete evidence."""

    def update(self, clean_count: float) -> float:
        if not np.isfinite(clean_count) or clean_count < 0:
            raise ValueError("clean_count must be finite and nonnegative")
        whole = int(np.floor(clean_count))
        fraction = float(clean_count - whole)
        evidence = [1.0] * whole
        if fraction > 0.0:
            evidence.append(fraction)
        return self.observe(evidence)


@dataclass
class IdentityBelief:
    config: ClosedWorldConfig
    existence: float
    lifecycle: Lifecycle
    hits: int
    misses: int = 0

    @classmethod
    def bootstrap(cls, config: ClosedWorldConfig) -> "IdentityBelief":
        initial = 1.0 - 1.0 / (config.birth_confirmation + 2.0)
        return cls(
            config=config,
            existence=float(initial),
            lifecycle=Lifecycle.ACTIVE,
            hits=max(1, config.birth_confirmation),
        )

    @classmethod
    def tentative(cls, config: ClosedWorldConfig) -> "IdentityBelief":
        initial = 1.0 - 1.0 / (config.birth_confirmation + 2.0)
        return cls(
            config=config,
            existence=float(initial),
            lifecycle=Lifecycle.TENTATIVE,
            hits=1,
        )

    def match(self, quality: float) -> None:
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("match quality must be in [0, 1]")
        was_dormant = self.lifecycle is Lifecycle.DORMANT
        self.existence = float(
            np.clip(1.0 - (1.0 - self.existence) * (1.0 - quality), 0.0, 1.0)
        )
        self.hits += 1
        self.misses = 0
        if was_dormant:
            self.lifecycle = Lifecycle.RECOVERED
        elif self.lifecycle is Lifecycle.TENTATIVE and self.hits < self.config.birth_confirmation:
            self.lifecycle = Lifecycle.TENTATIVE
        else:
            self.lifecycle = Lifecycle.ACTIVE

    def miss(self, *, visibility: float) -> None:
        if not np.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        self.misses += 1
        miss_if_present = 1.0 - visibility
        numerator = self.existence * miss_if_present
        denominator = numerator + (1.0 - self.existence)
        self.existence = float(
            np.clip(numerator / max(denominator, _PROBABILITY_EPS), 0.0, 1.0)
        )
        self.lifecycle = Lifecycle.DORMANT

    @property
    def removable(self) -> bool:
        return bool(
            self.misses >= self.config.birth_confirmation and self.existence < 0.5
        )

    def propagation_eligible(
        self, *, visibility: float, normalized_uncertainty: float
    ) -> bool:
        if not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        if not np.isfinite(normalized_uncertainty) or normalized_uncertainty < 0:
            raise ValueError("normalized_uncertainty must be finite and nonnegative")
        motion_certainty = 1.0 / (1.0 + normalized_uncertainty)
        posterior = self.existence * visibility * motion_certainty
        return bool(
            self.misses <= self.config.propagation_max_gap and posterior >= 0.5
        )


@dataclass(frozen=True)
class IdentityAssignment:
    matches: tuple[tuple[int, int], ...]
    misses: tuple[int, ...]
    births: tuple[int, ...]
    clutters: tuple[int, ...]
    total_cost: float


def _cost_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError(
            f"{name} must be a finite nonnegative vector with shape ({size},)"
        )
    return result


def solve_identity_conserving_assignment(
    pair_cost: np.ndarray,
    *,
    miss_cost: np.ndarray,
    birth_cost: np.ndarray,
    clutter_cost: np.ndarray,
    max_pair_cost: float,
    local_support_mask: np.ndarray | None = None,
    duplicate_affinity: np.ndarray | None = None,
) -> IdentityAssignment:
    """Minimize causal explanation energy while preserving local evidence first.

    Existing identities are favoured through their learned existence and arena
    miss costs, not by forcing every numerically feasible edge. Each link must
    beat its counterfactual miss-plus-birth/clutter explanation. Locally
    supported TGA/PSOI edges are resolved before persistent-proposal transport
    edges, so a global recovery cannot displace stronger local evidence.
    """

    pairs = np.asarray(pair_cost, dtype=np.float64)
    if pairs.ndim != 2:
        raise ValueError("pair_cost must be two-dimensional")
    track_count, observation_count = pairs.shape
    misses = _cost_vector(miss_cost, track_count, "miss_cost")
    births = _cost_vector(birth_cost, observation_count, "birth_cost")
    clutters = _cost_vector(clutter_cost, observation_count, "clutter_cost")
    if not np.isfinite(max_pair_cost) or max_pair_cost <= 0:
        raise ValueError("max_pair_cost must be positive and finite")

    feasible = np.isfinite(pairs) & (pairs <= max_pair_cost)
    if local_support_mask is None:
        local_support = feasible.copy()
    else:
        local_support = np.asarray(local_support_mask, dtype=bool)
        if local_support.shape != pairs.shape:
            raise ValueError("local_support_mask must match pair_cost shape")
        local_support = feasible & local_support

    if duplicate_affinity is None:
        affinity = np.zeros(
            (observation_count, observation_count), dtype=np.float64
        )
    else:
        affinity = np.asarray(duplicate_affinity, dtype=np.float64)
        if affinity.shape != (observation_count, observation_count):
            raise ValueError(
                "duplicate_affinity must be square over observations"
            )
        if np.any(~np.isfinite(affinity)) or np.any(
            (affinity < 0.0) | (affinity > 1.0)
        ):
            raise ValueError("duplicate_affinity must be finite and in [0, 1]")

    def staged_assignment(
        mask: np.ndarray,
        candidate_rows: tuple[int, ...],
        candidate_columns: tuple[int, ...],
        *,
        cost_matrix: np.ndarray = pairs,
        birth_vector: np.ndarray = births,
        clutter_vector: np.ndarray = clutters,
    ) -> list[tuple[int, int]]:
        if not candidate_rows or not candidate_columns:
            return []
        row_count = len(candidate_rows)
        column_count = len(candidate_columns)
        sub_cost = cost_matrix[np.ix_(candidate_rows, candidate_columns)]
        sub_mask = mask[np.ix_(candidate_rows, candidate_columns)]
        residual_cost = np.minimum(
            birth_vector[np.asarray(candidate_columns, dtype=int)],
            clutter_vector[np.asarray(candidate_columns, dtype=int)],
        )
        size = row_count + column_count
        augmented = np.full((size, size), np.inf, dtype=np.float64)
        augmented[:row_count, :column_count] = np.where(
            sub_mask, sub_cost, np.inf
        )
        for local_row, row in enumerate(candidate_rows):
            augmented[local_row, column_count + local_row] = misses[row]
        for local_column, column in enumerate(candidate_columns):
            augmented[row_count + local_column, local_column] = residual_cost[
                local_column
            ]
        augmented[row_count:, column_count:] = 0.0
        assigned_rows, assigned_columns = linear_sum_assignment(augmented)
        flow_count = sum(
            1
            for row, column in zip(
                assigned_rows.tolist(), assigned_columns.tolist()
            )
            if row < row_count
            and column < column_count
            and sub_mask[row, column]
        )
        if flow_count == 0:
            return []

        fixed_size = row_count + column_count - flow_count
        fixed_flow = np.full((fixed_size, fixed_size), np.inf, dtype=np.float64)
        fixed_flow[:row_count, :column_count] = np.where(
            sub_mask, sub_cost, np.inf
        )
        if row_count > flow_count:
            fixed_flow[:row_count, column_count:] = 0.0
        if column_count > flow_count:
            fixed_flow[row_count:, :column_count] = 0.0
        identity_rows, identity_columns = linear_sum_assignment(fixed_flow)
        return [
            (candidate_rows[row], candidate_columns[column])
            for row, column in zip(
                identity_rows.tolist(), identity_columns.tolist()
            )
            if row < row_count
            and column < column_count
            and sub_mask[row, column]
        ]

    all_rows = tuple(range(track_count))
    all_columns = tuple(range(observation_count))
    # Identity permutation is a global property: a locally mutual edge may
    # still worsen the joint explanation by forcing another identity onto a
    # much poorer observation. Resolve the complete local stage at once.
    matched = staged_assignment(local_support, all_rows, all_columns)
    locally_matched_rows = {row for row, _ in matched}
    locally_matched_columns = {column for _, column in matched}
    remaining_rows = {
        row for row in all_rows if row not in locally_matched_rows
    }
    remaining_columns = {
        column for column in all_columns if column not in locally_matched_columns
    }
    explained_columns = set(locally_matched_columns)
    probability_floor = np.finfo(np.float64).eps
    # Residual recovery is a sequential posterior. Miss/birth/clutter energy
    # decides whether one more recovery flow is supported; conditional pair
    # energy alone decides its identity permutation. After each accepted flow,
    # duplicate evidence is conditioned on the newly explained observation.
    while remaining_rows and remaining_columns:
        best_pair: tuple[tuple[float, int, int], int, int] | None = None
        explained_indices = np.asarray(
            sorted(explained_columns), dtype=int
        )
        for column in sorted(remaining_columns):
            explained = (
                float(np.max(affinity[column, explained_indices]))
                if explained_indices.size
                else 0.0
            )
            independent = float(np.clip(1.0 - explained, 0.0, 1.0))
            occupancy_energy = -np.log(
                max(independent, probability_floor)
            )
            base_clutter_probability = float(np.exp(-clutters[column]))
            conditional_clutter_probability = 1.0 - (
                (1.0 - base_clutter_probability) * (1.0 - explained)
            )
            residual_observation_cost = min(
                births[column] + occupancy_energy,
                -np.log(
                    max(
                        conditional_clutter_probability,
                        probability_floor,
                    )
                ),
            )
            for row in sorted(remaining_rows):
                conditional_pair = pairs[row, column] + occupancy_energy
                if (
                    not np.isfinite(conditional_pair)
                    or conditional_pair > max_pair_cost
                ):
                    continue
                gain = (
                    misses[row]
                    + residual_observation_cost
                    - conditional_pair
                )
                if gain <= 0.0:
                    continue
                pair_key = (-conditional_pair, -row, -column)
                if best_pair is None or pair_key > best_pair[0]:
                    best_pair = (pair_key, row, column)
        if best_pair is None:
            break
        _, row, column = best_pair
        matched.append((row, column))
        remaining_rows.remove(row)
        remaining_columns.remove(column)
        explained_columns.add(column)

    matched_rows = {row for row, _ in matched}
    matched_columns = {column for _, column in matched}
    missed = tuple(row for row in range(track_count) if row not in matched_rows)
    born: list[int] = []
    cluttered: list[int] = []
    for column in range(observation_count):
        if column in matched_columns:
            continue
        if births[column] <= clutters[column]:
            born.append(column)
        else:
            cluttered.append(column)

    total = float(
        sum(pairs[row, column] for row, column in matched)
        + sum(misses[row] for row in missed)
        + sum(births[column] for column in born)
        + sum(clutters[column] for column in cluttered)
    )
    return IdentityAssignment(
        matches=tuple(sorted(matched)),
        misses=missed,
        births=tuple(born),
        clutters=tuple(cluttered),
        total_cost=total,
    )
