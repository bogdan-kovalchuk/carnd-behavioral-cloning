"""Camera-specific steering-label helpers.

Both correction functions are constant-time and use constant auxiliary
memory.  ``correct_steering`` makes the three cases visible as branches,
which is straightforward to read when only three camera positions exist.
``correct_steering_lookup`` stores the offsets in a table, so extending the
camera vocabulary changes data rather than control flow.  The lookup version
also makes the offset comparison easy to test across implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class CameraSample:
    """An image path paired with its corrected steering label."""

    image_path: str
    steering: float
    camera: str


_CAMERA_OFFSETS = {"center": 0.0, "left": 1.0, "right": -1.0}


def _validate_correction_inputs(angle: float, camera: str, correction: float):
    if isinstance(angle, bool) or isinstance(correction, bool):
        raise TypeError("angle and correction must be numeric")
    try:
        angle = float(angle)
        correction = float(correction)
    except (TypeError, ValueError):
        raise TypeError("angle and correction must be numeric")
    if not math.isfinite(angle) or not math.isfinite(correction):
        raise ValueError("angle and correction must be finite")
    if correction < 0:
        raise ValueError("correction must not be negative")
    if not isinstance(camera, str) or camera not in _CAMERA_OFFSETS:
        raise ValueError("camera must be center, left, or right")
    return angle, correction


def correct_steering(angle: float, camera: str, correction: float = 0.2) -> float:
    """Apply the conventional steering correction for a camera position."""

    angle, correction = _validate_correction_inputs(angle, camera, correction)
    if camera == "center":
        return angle
    if camera == "left":
        return angle + correction
    if camera == "right":
        return angle - correction
    raise ValueError("camera must be center, left, or right")


def correct_steering_lookup(
    angle: float,
    camera: str,
    correction: float = 0.2,
) -> float:
    """Apply the same correction through a camera-to-offset lookup table."""

    angle, correction = _validate_correction_inputs(angle, camera, correction)
    return angle + _CAMERA_OFFSETS[camera] * correction


def make_camera_samples(
    center_path: str,
    left_path: str,
    right_path: str,
    angle: float,
    correction: float = 0.2,
) -> Tuple[CameraSample, ...]:
    """Return center, left, and right samples in a stable order."""

    return tuple(
        CameraSample(path, correct_steering(angle, camera, correction), camera)
        for path, camera in (
            (center_path, "center"),
            (left_path, "left"),
            (right_path, "right"),
        )
    )
