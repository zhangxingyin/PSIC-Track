"""Small, validated configuration for the active TGA+PSOI+NCIC tracker."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, TypeVar


@dataclass(frozen=True)
class BaseTrackerConfig:
    min_score: float = 0.05
    min_hits: int = 1
    max_age: int = 30
    max_assignment_cost: float = 1.5

    def validate(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if self.min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        if self.max_age < 0:
            raise ValueError("max_age cannot be negative")
        if self.max_assignment_cost <= 0:
            raise ValueError("max_assignment_cost must be positive")


@dataclass(frozen=True)
class PoseConfig:
    enabled: bool = True

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("pose enabled must be boolean")


@dataclass(frozen=True)
class AssociationConfig:
    enabled: bool = True

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("association enabled must be boolean")


@dataclass(frozen=True)
class ObservationIntegrityConfig:
    enabled: bool = False

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("observation_integrity enabled must be boolean")


@dataclass(frozen=True)
class ClosedWorldConfig:
    enabled: bool = False
    birth_confirmation: int = 2
    propagation_max_gap: int = 3

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("closed_world enabled must be boolean")
        if self.birth_confirmation < 1:
            raise ValueError("birth_confirmation must be positive")
        if self.propagation_max_gap < 0:
            raise ValueError("propagation_max_gap must be nonnegative")


SectionT = TypeVar(
    "SectionT",
    BaseTrackerConfig,
    PoseConfig,
    AssociationConfig,
    ObservationIntegrityConfig,
    ClosedWorldConfig,
)


def _section(cls: type[SectionT], raw: Mapping[str, Any] | None) -> SectionT:
    values = dict(raw or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    result = cls(**values)
    result.validate()
    return result


@dataclass(frozen=True)
class TrackerConfig:
    base: BaseTrackerConfig = BaseTrackerConfig()
    pose: PoseConfig = PoseConfig()
    association: AssociationConfig = AssociationConfig()
    observation_integrity: ObservationIntegrityConfig = ObservationIntegrityConfig()
    closed_world: ClosedWorldConfig = ClosedWorldConfig()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.base.validate()
        self.pose.validate()
        self.association.validate()
        self.observation_integrity.validate()
        self.closed_world.validate()

    @property
    def near_closed_enabled(self) -> bool:
        return bool(self.observation_integrity.enabled or self.closed_world.enabled)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "TrackerConfig":
        values = dict(raw or {})
        preset = values.pop("preset", None)
        modules = values.pop("modules", None)
        if preset is not None:
            if preset not in {"tga", "psictrack_psoi_ncic"}:
                raise ValueError(f"unknown tracker preset: {preset}")
            enabled = preset == "psictrack_psoi_ncic"
            preset_values: dict[str, dict[str, Any]] = {
                "pose": {"enabled": enabled},
                "association": {"enabled": enabled},
                "observation_integrity": {"enabled": enabled},
                "closed_world": {"enabled": enabled},
            }
            for section, section_values in preset_values.items():
                explicit = values.get(section)
                if explicit is not None:
                    section_values.update(dict(explicit))
                values[section] = section_values
        if modules is not None:
            if preset is None:
                raise ValueError("module switches require a tracker preset")
            if not isinstance(modules, Mapping):
                raise ValueError("module switches must be a mapping")
            allowed_modules = {"psoi", "ncic"}
            unknown_modules = sorted(set(modules) - allowed_modules)
            if unknown_modules:
                raise ValueError(f"unknown module switches: {unknown_modules}")
            if any(not isinstance(value, bool) for value in modules.values()):
                raise ValueError("module switches must be boolean")
            psoi_enabled = bool(modules.get("psoi", values["pose"]["enabled"]))
            ncic_enabled = bool(modules.get("ncic", values["closed_world"]["enabled"]))
            values["pose"]["enabled"] = psoi_enabled
            values["observation_integrity"]["enabled"] = psoi_enabled
            values["association"]["enabled"] = psoi_enabled
            values["closed_world"]["enabled"] = ncic_enabled
        allowed = {
            "base",
            "pose",
            "association",
            "observation_integrity",
            "closed_world",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown TrackerConfig sections: {unknown}")
        return cls(
            base=_section(BaseTrackerConfig, values.get("base")),
            pose=_section(PoseConfig, values.get("pose")),
            association=_section(AssociationConfig, values.get("association")),
            observation_integrity=_section(
                ObservationIntegrityConfig, values.get("observation_integrity")
            ),
            closed_world=_section(ClosedWorldConfig, values.get("closed_world")),
        )
