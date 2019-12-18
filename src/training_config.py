"""Immutable, dependency-free configuration for training experiments."""

from __future__ import annotations

from dataclasses import dataclass


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
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool):
            raise TypeError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.steering_correction < 0:
            raise ValueError("steering_correction must not be negative")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
