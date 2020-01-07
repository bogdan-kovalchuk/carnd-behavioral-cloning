"""Immutable, dependency-free configuration for training experiments.

``TrainingConfig`` uses a frozen dataclass, giving named fields, generated
comparison, and constant-time field access with O(1) storage for this fixed
schema.  ``MappingTrainingConfig`` validates the same schema but stores a
copied ``MappingProxyType``.  Construction is O(k) for k fields and lookup is
O(1) on the underlying mapping.  The mapping form is convenient for external
key/value inputs; the dataclass form is clearer for Python callers.  The
adapter and signature functions make the two alternatives comparable without
coupling the training runtime to either storage choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral
from types import MappingProxyType
from typing import Mapping


def _is_finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class TrainingConfig:
    """Validated values needed to describe a reproducible training run."""

    batch_size: int = 16
    epochs: int = 5
    validation_split: float = 0.2
    learning_rate: float = 0.001
    steering_correction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, Integral) or isinstance(self.batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.epochs, Integral) or isinstance(self.epochs, bool):
            raise TypeError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not _is_finite_number(self.validation_split):
            raise ValueError("validation_split must be finite")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if not _is_finite_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not _is_finite_number(self.steering_correction) or self.steering_correction < 0:
            raise ValueError("steering_correction must not be negative")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "epochs", int(self.epochs))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "validation_split", float(self.validation_split))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "steering_correction", float(self.steering_correction))

    @classmethod
    def from_mapping(cls, values: Mapping) -> "TrainingConfig":
        """Build the dataclass representation from a field mapping."""

        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        return cls(**dict(values))


_CONFIG_FIELDS = frozenset(field.name for field in fields(TrainingConfig))


class MappingTrainingConfig:
    """Alternative immutable configuration backed by a read-only mapping."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping = None, **overrides) -> None:
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        unknown = (set(values) | set(overrides)) - _CONFIG_FIELDS
        if unknown:
            raise ValueError("unknown training fields: {}".format(sorted(unknown)))
        merged = asdict(TrainingConfig())
        merged.update(values)
        merged.update(overrides)
        validated = TrainingConfig(**merged)
        object.__setattr__(self, "_values", MappingProxyType(asdict(validated)))

    def __setattr__(self, name, value) -> None:
        if hasattr(self, "_values"):
            raise AttributeError("MappingTrainingConfig is immutable")
        object.__setattr__(self, name, value)

    def __getitem__(self, name):
        return self._values[name]

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name)

    def as_mapping(self) -> Mapping:
        return self._values

    def as_training_config(self) -> TrainingConfig:
        """Convert this representation to the canonical dataclass."""

        return TrainingConfig.from_mapping(self._values)


def to_training_config(config) -> TrainingConfig:
    """Normalize either supported representation to ``TrainingConfig``."""

    if isinstance(config, TrainingConfig):
        return config
    if isinstance(config, MappingTrainingConfig):
        return config.as_training_config()
    if isinstance(config, Mapping):
        return TrainingConfig.from_mapping(config)
    raise TypeError("config must be a TrainingConfig or mapping")


def configuration_signature(config) -> tuple:
    """Return a stable, hashable field/value signature for either form."""

    canonical = to_training_config(config)
    return tuple((field.name, getattr(canonical, field.name)) for field in fields(canonical))
