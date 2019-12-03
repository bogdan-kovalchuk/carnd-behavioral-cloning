"""Deterministic image augmentation primitives."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def augment_sample(
    image: np.ndarray,
    steering: float,
    *,
    flip: bool = False,
) -> Tuple[np.ndarray, float]:
    """Return a copied image and steering label, optionally mirrored."""

    array = np.asarray(image)
    augmented = np.flip(array, axis=1).copy() if flip else array.copy()
    corrected_steering = -float(steering) if flip else float(steering)
    return augmented, corrected_steering


def augment_sample_indexed(
    image: np.ndarray,
    steering: float,
    *,
    flip: bool = False,
) -> Tuple[np.ndarray, float]:
    """Mirror an image by gathering columns in reverse index order."""

    array = np.asarray(image)
    if flip:
        columns = np.arange(array.shape[1] - 1, -1, -1)
        augmented = np.take(array, columns, axis=1).copy()
        return augmented, -float(steering)
    return array.copy(), float(steering)
