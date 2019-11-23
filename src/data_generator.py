"""Deterministic batch helpers for driving-log rows.

The functions in this module deliberately stop after one pass.  This makes
them useful for tests and for callers that want to control epoch boundaries
without importing the training stack from ``model.py``.

Approach comparison
-------------------
``batch_generator`` copies the values into a list and optionally shuffles that
list.  It takes O(n) extra references and O(n) time, then slices each batch in
O(batch_size) time.  It is the simplest choice for general iterables.

``indexed_batch_generator`` materializes the source as a tuple, permutes an
O(n) integer index array, and fetches values lazily per batch.  Its asymptotic
time and memory costs are also O(n), but the temporary data is numeric indices
rather than a second rearranged value list.  It is the better fit for
indexable, array-like sources and makes the access strategy explicit.
"""

from __future__ import annotations

import random
import math
from numbers import Integral, Real
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def _validate_batch_arguments(batch_size: int, seed: int) -> None:
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a real number".format(name))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("{} must be finite".format(name))
    return number


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

    _validate_batch_arguments(batch_size, seed)

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

    correction = _finite_real(correction, "correction")
    expanded = []
    for row in rows:
        if len(row) < 4:
            raise ValueError("each row needs four camera fields")
        center, left, right, angle = row[:4]
        center_angle = _finite_real(float(angle), "steering angle")
        expanded.extend(
            (
                (center, center_angle, "center"),
                (left, center_angle + correction, "left"),
                (right, center_angle - correction, "right"),
            )
        )
    return tuple(expanded)


def indexed_batch_generator(
    samples: Sequence[T],
    batch_size: int = 32,
    *,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> Iterator[Tuple[T, ...]]:
    """Yield batches by permuting NumPy indices instead of sample values.

    This alternative is useful when the source is an indexable array-like
    object: only integer indices are rearranged, while the source values are
    fetched when each batch is formed.
    """

    _validate_batch_arguments(batch_size, seed)

    items = tuple(samples)
    indices = np.arange(len(items), dtype=np.int64)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)

    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        if drop_last and len(batch_indices) < batch_size:
            break
        yield tuple(items[int(index)] for index in batch_indices)
