"""Deterministic batch helpers for driving-log rows.

The functions in this module deliberately stop after one pass.  This makes
them useful for tests and for callers that want to control epoch boundaries
without importing the training stack from ``model.py``.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def batch_generator(
    samples: Iterable[T],
    batch_size: int = 32,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> Iterator[Tuple[T, ...]]:
    """Yield deterministic tuple batches from ``samples``.

    A copied list protects the caller's collection from both shuffling and
    accidental mutation.  With ``shuffle=False`` the input order is retained;
    with shuffling enabled a local ``random.Random`` instance makes repeated
    calls with the same seed reproducible.
    """

    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    ordered: List[T] = list(samples)
    if shuffle:
        random.Random(seed).shuffle(ordered)

    for offset in range(0, len(ordered), batch_size):
        batch = tuple(ordered[offset : offset + batch_size])
        if drop_last and len(batch) < batch_size:
            break
        yield batch


def expand_camera_samples(
    rows: Sequence[Sequence[str]],
    correction: float = 0.2,
) -> Tuple[Tuple[str, float, str], ...]:
    """Expand driving-log rows into center, left, and right camera records."""

    expanded = []
    for row in rows:
        center, left, right, angle = row[:4]
        center_angle = float(angle)
        expanded.extend(
            (
                (center, center_angle, "center"),
                (left, center_angle + correction, "left"),
                (right, center_angle - correction, "right"),
            )
        )
    return tuple(expanded)
