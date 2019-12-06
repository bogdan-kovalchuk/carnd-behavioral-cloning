"""Deterministic image augmentation primitives."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _validate_inputs(image: np.ndarray, steering: float, flip: bool) -> Tuple[np.ndarray, float]:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be a 2-D or 3-D array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must contain numeric values")
    if not isinstance(flip, (bool, np.bool_)):
        raise TypeError("flip must be boolean")
    if isinstance(steering, (bool, np.bool_)):
        raise TypeError("steering must be numeric")
    try:
        steering = float(steering)
    except (TypeError, ValueError):
        raise TypeError("steering must be numeric")
    if not math.isfinite(steering):
        raise ValueError("steering must be finite")
    return array, steering


def augment_sample(
    image: np.ndarray,
    steering: float,
    *,
    flip: bool = False,
) -> Tuple[np.ndarray, float]:
    """Return a copied image and steering label, optionally mirrored."""

    array, steering = _validate_inputs(image, steering, flip)
    augmented = np.flip(array, axis=1).copy() if flip else array.copy()
    corrected_steering = -steering if flip else steering
    return augmented, corrected_steering


def augment_sample_indexed(
    image: np.ndarray,
    steering: float,
    *,
    flip: bool = False,
) -> Tuple[np.ndarray, float]:
    """Mirror an image by gathering columns in reverse index order."""

    array, steering = _validate_inputs(image, steering, flip)
    if flip:
        columns = np.arange(array.shape[1] - 1, -1, -1)
        augmented = np.take(array, columns, axis=1).copy()
        return augmented, -steering
    return array.copy(), steering
